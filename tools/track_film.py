"""
Evaluate BYTETracker + FiLM ReID on MOT benchmarks.

Usage:
    python tools/track_film.py \
        -f exps/example/mot/yolox_x_mix_det.py \
        -c pretrained/bytetrack_x_mot17.pth.tar \
        --film-ckpt checkpoints/film_reid/film_reid_best.pth \
        --fp16 --fuse

FiLM-specific args:
    --film-ckpt           path to FiLM ReID checkpoint (required)
    --film-num-classes    number of identities (auto-detected from ckpt when omitted)
    --film-device         device for FiLM model (default: cuda)
    --film-seq-len        trajectory history length (default: 20)
    --film-motion-hidden  motion encoder hidden dim (auto-detected from ckpt when omitted)
    --film-fastreid-config optional fast-reid config .yml for backbone
"""

import os
import sys

FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(FILE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger
from yolox.evaluators import MOTEvaluator

import argparse
import random
import warnings
import glob
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack + FiLM ReID Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument("--dist-backend", default="nccl", type=str)
    parser.add_argument("--dist-url", default=None, type=str)
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("-d", "--devices", default=None, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--num_machines", default=1, type=int)
    parser.add_argument("--machine_rank", default=0, type=int)

    # experiment
    parser.add_argument("-f", "--exp_file", default=None, type=str)
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true")
    parser.add_argument("--test", dest="test", default=False, action="store_true")
    parser.add_argument("--speed", dest="speed", default=False, action="store_true")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    # detection args
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="detector checkpoint")
    parser.add_argument("--conf", default=0.01, type=float)
    parser.add_argument("--nms", default=0.7, type=float)
    parser.add_argument("--tsize", default=None, type=int)
    parser.add_argument("--seed", default=None, type=int)

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.6)
    parser.add_argument("--track_buffer", type=int, default=30)
    parser.add_argument("--match_thresh", type=float, default=0.9)
    parser.add_argument("--min-box-area", type=float, default=100)
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true")

    # ReID fusion params (shared with existing trackers)
    parser.add_argument("--reid-weight", type=float, default=0.35,
                        help="appearance cost weight in IoU+ReID fusion")
    parser.add_argument("--reid-thresh", type=float, default=0.7,
                        help="max cosine distance before ReID cost is capped")
    parser.add_argument("--reid-alpha", type=float, default=0.9,
                        help="EMA momentum for track ReID feature smoothing")

    # FiLM-specific args
    parser.add_argument("--film-ckpt", required=True, type=str,
                        help="path to FiLM ReID checkpoint (.pth)")
    parser.add_argument("--film-num-classes", default=None, type=int,
                        help="number of identities — auto-detected from checkpoint if omitted")
    parser.add_argument("--film-device", default="cuda", type=str,
                        help="device for FiLM model (cuda / cpu)")
    parser.add_argument("--film-seq-len", default=20, type=int,
                        help="trajectory history length fed to motion encoder")
    parser.add_argument("--film-motion-hidden", default=None, type=int,
                        help="motion encoder hidden dim — auto-detected from checkpoint if omitted")
    parser.add_argument("--film-fastreid-config", default=None, type=str,
                        help="optional fast-reid config .yml for backbone construction")

    return parser


def compare_dataframes(gts, ts):
    accs, names = [], []
    for k, tsacc in ts.items():
        if k in gts:
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))
    return accs, names


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn("Seeding enabled — CUDNN deterministic mode is ON.")

    is_distributed = num_gpu > 1
    cudnn.benchmark = True
    rank = args.local_rank

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    results_folder = os.path.join(file_name, "track_results")
    os.makedirs(results_folder, exist_ok=True)
    video_folder = os.path.join(file_name, "videos")

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    val_loader = exp.get_eval_loader(args.batch_size, is_distributed, args.test)
    evaluator = MOTEvaluator(
        args=args,
        dataloader=val_loader,
        img_size=exp.test_size,
        confthre=exp.test_conf,
        nmsthre=exp.nmsthre,
        num_classes=exp.num_classes,
    )

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if not args.speed and not args.trt:
        ckpt_file = args.ckpt or os.path.join(file_name, "best_ckpt.pth.tar")
        logger.info("Loading detector checkpoint: {}".format(ckpt_file))
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        model.load_state_dict(ckpt["model"])
        logger.info("Detector checkpoint loaded.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.fuse:
        logger.info("Fusing conv+bn ...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse and not is_distributed and args.batch_size == 1
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(trt_file), "TensorRT model not found. Run tools/trt.py first."
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    *_, summary = evaluator.evaluate_film(
        model, is_distributed, args.fp16, trt_file, decoder,
        exp.test_size, results_folder, video_folder,
    )
    logger.info("\n" + summary)

    # ── MOTA evaluation ───────────────────────────────────────────────────
    mm.lap.default_solver = 'lap'

    gt_type = '_val_half' if exp.val_ann == 'val_half.json' else ''

    eval_dataset  = getattr(val_loader, "dataset", None)
    dataset_root  = getattr(eval_dataset, "data_dir", None)
    dataset_split = getattr(eval_dataset, "name", None)
    gt_root = None
    if dataset_root and dataset_split:
        gt_root = os.path.join(dataset_root, dataset_split)

    if args.mot20:
        gt_root = gt_root or os.path.join('datasets', 'MOT20', 'train')
    else:
        gt_root = gt_root or os.path.join('datasets', 'mot', 'train')

    gtfiles = glob.glob(os.path.join(gt_root, '*/gt/gt{}.txt'.format(gt_type)))
    tsfiles = [
        f for f in glob.glob(os.path.join(results_folder, '*.txt'))
        if not os.path.basename(f).startswith('eval')
    ]

    logger.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    logger.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
    logger.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    logger.info('Loading files.')

    mot_csv_sep = r'\s*,\s*'
    gt = OrderedDict([
        (Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1, sep=mot_csv_sep))
        for f in gtfiles
    ])
    ts = OrderedDict([
        (os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1, sep=mot_csv_sep))
        for f in tsfiles
    ])

    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts)

    logger.info('Running metrics')
    metrics = [
        'recall', 'precision', 'num_unique_objects', 'mostly_tracked',
        'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
        'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects',
    ]
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    div_dict = {
        'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
        'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost'],
    }
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = summary[divided] / summary[divisor]
    fmt = mh.formatters
    for k in ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations',
              'mostly_tracked', 'partially_tracked', 'mostly_lost']:
        fmt[k] = fmt['mota']
    print(mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))

    metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    logger.info('Completed')


if __name__ == "__main__":
    args = make_parser().parse_args()

    if not os.path.isfile(args.film_ckpt):
        raise FileNotFoundError("FiLM checkpoint not found: {}".format(args.film_ckpt))

    if not args.experiment_name:
        args.experiment_name = "eval_film_reid"

    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args, num_gpu),
    )
