"""Run MOT evaluation with BYTETrackerSLSTM (xLSTM trajectory corrector).

Entry point mirrors tools/track.py but swaps BYTETracker with
BYTETrackerSLSTM from new_pipeline.byte_tracker_slstm at runtime.

All evaluation metrics (MOTA, MOTP, IDF1, …) are computed identically
to tools/track.py — the only difference is the tracker used per frame.

Example:
    python new_pipeline/track_slstm.py \
        -f exps/example/mot/yolox_m_mix_det.py \
        -c pretrained/bytetrack_m_mot17.pth.tar \
        -b 1 -d 1 --fp16 --fuse \
        --experiment-name eval_slstm \
  --slstm_ckpt checkpoints/slstm_mot17/slstm_mot17.pt
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import random
import warnings
from collections import OrderedDict
from pathlib import Path

import motmetrics as mm
import torch
import torch.backends.cudnn as cudnn
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.evaluators import MOTEvaluator
from yolox.evaluators import mot_evaluator
from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger

from new_pipeline.byte_tracker_slstm import BYTETrackerSLSTM


# ── CLI ───────────────────────────────────────────────────────────────────────

def make_parser():
    import argparse
    parser = argparse.ArgumentParser("ByteTrack + sLSTM Eval")
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

    # detector args
    parser.add_argument("-c", "--ckpt", default=None, type=str)
    parser.add_argument("--conf", default=0.01, type=float)
    parser.add_argument("--nms", default=0.7, type=float)
    parser.add_argument("--tsize", default=None, type=int)
    parser.add_argument("--seed", default=None, type=int)

    # tracking args (identical to tools/track.py)
    parser.add_argument("--track_thresh", type=float, default=0.6)
    parser.add_argument("--track_buffer", type=int, default=30)
    parser.add_argument("--match_thresh", type=float, default=0.9)
    parser.add_argument("--min-box-area", type=float, default=100)
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true")

    # sLSTM args
    parser.add_argument(
        "--slstm_ckpt",
        default="checkpoints/slstm_mot17/slstm_mot17.pt",
        type=str,
        help="sLSTM checkpoint (.pt) saved from Kaggle training",
    )
    parser.add_argument("--slstm_vocab_size", default=256, type=int,
                        help="Token vocabulary size (must match training)")
    parser.add_argument("--slstm_context_length", default=256, type=int,
                        help="Token context window length (must match training)")
    parser.add_argument("--slstm_alpha0", default=0.5, type=float,
                        help="Max sLSTM blend weight (0=pure Kalman, 1=pure sLSTM)")
    parser.add_argument("--slstm_beta", default=0.3, type=float,
                        help="Decay rate for blend weight when track is missing")
    parser.add_argument("--slstm_device", default=None, type=str,
                        help="Device for sLSTM inference (default: same as detector)")
    return parser


# ── Evaluation helpers (identical to tools/track.py) ─────────────────────────

def compare_dataframes(gts, ts):
    accs, names = [], []
    for k, tsacc in ts.items():
        if k in gts:
            logger.info("Comparing {}...".format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, "iou", distth=0.5))
            names.append(k)
        else:
            logger.warning("No ground truth for {}, skipping.".format(k))
    return accs, names


# ── Main evaluation loop (mirrors tools/track.py exactly) ────────────────────

@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn("You have chosen to seed testing. This will turn on the CUDNN deterministic setting.")

    is_distributed = num_gpu > 1
    cudnn.benchmark = True
    rank = args.local_rank

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    results_folder = os.path.join(file_name, "track_results")
    os.makedirs(results_folder, exist_ok=True)

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
        ckpt_file = args.ckpt if args.ckpt else os.path.join(file_name, "best_ckpt.pth.tar")
        logger.info("loading checkpoint")
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse and not is_distributed and args.batch_size == 1
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(trt_file), "TensorRT model not found! Run tools/trt.py first."
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    # ── Run tracking ──────────────────────────────────────────────────────
    *_, summary = evaluator.evaluate(
        model, is_distributed, args.fp16, trt_file, decoder, exp.test_size, results_folder
    )
    logger.info("\n" + summary)

    # ── Compute MOTA / motchallenge metrics ───────────────────────────────
    mm.lap.default_solver = "lap"

    gt_type = "_val_half" if exp.val_ann == "val_half.json" else ""
    print("gt_type", gt_type)

    if args.mot20:
        gtfiles = glob.glob(
            os.path.join("datasets/MOT20/train", "*/gt/gt{}.txt".format(gt_type))
        )
    else:
        gtfiles = glob.glob(
            os.path.join("datasets/mot/train", "*/gt/gt{}.txt".format(gt_type))
        )
    print("gt_files", gtfiles)
    tsfiles = [
        f for f in glob.glob(os.path.join(results_folder, "*.txt"))
        if not os.path.basename(f).startswith("eval")
    ]

    logger.info("Found {} groundtruths and {} test files.".format(len(gtfiles), len(tsfiles)))
    logger.info("Available LAP solvers {}".format(mm.lap.available_solvers))
    logger.info("Default LAP solver '{}'".format(mm.lap.default_solver))
    logger.info("Loading files.")

    gt = OrderedDict(
        [(Path(f).parts[-3], mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=1))
         for f in gtfiles]
    )
    ts = OrderedDict(
        [(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=-1))
         for f in tsfiles]
    )

    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts)

    logger.info("Running metrics")
    metrics = [
        "recall", "precision", "num_unique_objects", "mostly_tracked",
        "partially_tracked", "mostly_lost", "num_false_positives", "num_misses",
        "num_switches", "num_fragmentations", "mota", "motp", "num_objects",
    ]
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    div_dict = {
        "num_objects": ["num_false_positives", "num_misses", "num_switches", "num_fragmentations"],
        "num_unique_objects": ["mostly_tracked", "partially_tracked", "mostly_lost"],
    }
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = summary[divided] / summary[divisor]
    fmt = mh.formatters
    for k in ["num_false_positives", "num_misses", "num_switches", "num_fragmentations",
               "mostly_tracked", "partially_tracked", "mostly_lost"]:
        fmt[k] = fmt["mota"]
    print(mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))

    metrics = mm.metrics.motchallenge_metrics + ["num_objects"]
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    logger.info("Completed")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    # Monkey-patch BYTETracker → BYTETrackerSLSTM inside the evaluator
    class _PatchedTracker(BYTETrackerSLSTM):
        def __init__(self, patch_args, frame_rate=30):
            super().__init__(
                patch_args,
                frame_rate=frame_rate,
                slstm_ckpt=patch_args.slstm_ckpt,
                vocab_size=patch_args.slstm_vocab_size,
                context_length=patch_args.slstm_context_length,
                alpha0=patch_args.slstm_alpha0,
                beta=patch_args.slstm_beta,
                device=patch_args.slstm_device,
            )

    mot_evaluator.BYTETracker = _PatchedTracker

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
