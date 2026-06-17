from collections import OrderedDict, defaultdict
from pathlib import Path

import argparse
import contextlib
import glob
import io
import json
import os
import random
import tempfile
import time
import warnings

import motmetrics as mm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from loguru import logger
from tqdm import tqdm

from yolox.evaluators.mot_evaluator import write_results
from yolox.exp import get_exp
from yolox.tracker.byte_tracker import BYTETracker
from yolox.utils import setup_logger


def make_parser():
    parser = argparse.ArgumentParser("YOLO + ByteTrack Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment file used for MOT eval loader and test split",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("-d", "--devices", default=None, type=int, help="device count")
    parser.add_argument("-c", "--ckpt", required=True, type=str, help="YOLO .pt checkpoint")
    parser.add_argument("--conf", default=0.01, type=float, help="YOLO detection confidence")
    parser.add_argument("--nms", default=0.7, type=float, help="YOLO NMS IoU threshold")
    parser.add_argument("--tsize", default=None, type=int, help="YOLO inference image size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument("--yolo-classes", nargs="+", type=int, default=None, help="class ids to keep")
    parser.add_argument("--device", default=None, type=str, help="YOLO device, e.g. 0 or cpu")
    parser.add_argument("--fp16", default=False, action="store_true", help="use half precision in YOLO")
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="frames to keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold")
    parser.add_argument("--min-box-area", type=float, default=100, help="filter out tiny boxes")
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test MOT20")
    parser.add_argument(
        "opts",
        help="modify exp config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            logger.info("Comparing {}...".format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, "iou", distth=0.5))
            names.append(k)
        else:
            logger.warning("No ground truth for {}, skipping.".format(k))
    return accs, names


def scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def first(value):
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def image_path_from_loader(dataloader, img_file_name):
    dataset = dataloader.dataset
    return os.path.join(dataset.data_dir, dataset.name, img_file_name)


def apply_sequence_tuning(args, video_name, original_track_thresh):
    if video_name == "MOT17-05-FRCNN" or video_name == "MOT17-06-FRCNN":
        args.track_buffer = 14
    elif video_name == "MOT17-13-FRCNN" or video_name == "MOT17-14-FRCNN":
        args.track_buffer = 25
    else:
        args.track_buffer = 30

    if video_name == "MOT17-01-FRCNN":
        args.track_thresh = 0.65
    elif video_name == "MOT17-06-FRCNN":
        args.track_thresh = 0.65
    elif video_name == "MOT17-12-FRCNN":
        args.track_thresh = 0.7
    elif video_name == "MOT17-14-FRCNN":
        args.track_thresh = 0.67
    elif video_name in ["MOT20-06", "MOT20-08"]:
        args.track_thresh = 0.3
    else:
        args.track_thresh = original_track_thresh


def yolo_dets(result, yolo_classes):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 5), dtype=np.float32)

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    if yolo_classes is not None:
        cls = boxes.cls.detach().cpu().numpy().astype(np.int64)
        keep = np.isin(cls, np.asarray(yolo_classes, dtype=np.int64))
        xyxy = xyxy[keep]
        conf = conf[keep]

    if xyxy.size == 0:
        return np.empty((0, 5), dtype=np.float32)
    return np.concatenate([xyxy, conf[:, None]], axis=1).astype(np.float32)


def image_id_from_ids(ids):
    if torch.is_tensor(ids):
        return int(ids.reshape(-1)[0].item())
    if isinstance(ids, np.ndarray):
        return int(ids.reshape(-1)[0])
    if isinstance(ids, (list, tuple)):
        return image_id_from_ids(ids[0])
    return int(ids)


def append_coco_predictions(data_list, dets, image_id, dataset):
    if dets.shape[0] == 0:
        return

    label = dataset.class_ids[0]
    bboxes = dets[:, :4].copy()
    bboxes[:, 2] -= bboxes[:, 0]
    bboxes[:, 3] -= bboxes[:, 1]
    scores = dets[:, 4]

    for bbox, score in zip(bboxes, scores):
        data_list.append(
            {
                "image_id": int(image_id),
                "category_id": int(label),
                "bbox": bbox.tolist(),
                "score": float(score),
                "segmentation": [],
            }
        )


def detection_summary(dataloader, data_list, inference_time, track_time, n_samples):
    a_infer_time = 1000 * inference_time / (n_samples * dataloader.batch_size)
    a_track_time = 1000 * track_time / (n_samples * dataloader.batch_size)
    info = ", ".join(
        [
            "Average {} time: {:.2f} ms".format(k, v)
            for k, v in zip(
                ["forward", "track", "inference"],
                [a_infer_time, a_track_time, a_infer_time + a_track_time],
            )
        ]
    )
    info += "\n"

    if len(data_list) == 0:
        return info

    coco_gt = dataloader.dataset.coco
    fd, tmp = tempfile.mkstemp()
    os.close(fd)
    try:
        with open(tmp, "w") as f:
            json.dump(data_list, f)
        coco_dt = coco_gt.loadRes(tmp)

        try:
            from yolox.layers import COCOeval_opt as COCOeval
        except ImportError:
            from pycocotools.cocoeval import COCOeval
            logger.warning("Use standard COCOeval.")

        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            coco_eval.summarize()
        info += redirect_string.getvalue()
    finally:
        os.remove(tmp)

    return info


def evaluate_yolo(exp, args):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: install Ultralytics with `pip install ultralytics` "
            "to evaluate YOLO .pt checkpoints."
        ) from exc

    if args.batch_size != 1:
        raise ValueError("YOLO eval currently expects `-b/--batch-size 1` so sequences stay ordered.")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn("You have chosen to seed testing. This turns on CUDNN deterministic mode.")

    cudnn.benchmark = True

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    results_folder = os.path.join(file_name, "track_results")
    os.makedirs(results_folder, exist_ok=True)
    setup_logger(file_name, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    val_loader = exp.get_eval_loader(args.batch_size, False, False)
    model = YOLO(args.ckpt)

    tracker = BYTETracker(args)
    original_track_thresh = args.track_thresh
    data_list = []
    results = []
    video_names = defaultdict()
    inference_time = 0.0
    track_time = 0.0
    n_samples = max(len(val_loader) - 1, 1)

    for cur_iter, (_, _, info_imgs, ids) in enumerate(tqdm(val_loader)):
        frame_id = int(scalar(info_imgs[2]))
        video_id = int(scalar(info_imgs[3]))
        img_file_name = first(info_imgs[4])
        video_name = img_file_name.split("/")[0]

        apply_sequence_tuning(args, video_name, original_track_thresh)

        if video_name not in video_names:
            video_names[video_id] = video_name
        if frame_id == 1:
            tracker = BYTETracker(args)
            if len(results) != 0:
                prev_name = video_names[video_id - 1]
                result_filename = os.path.join(results_folder, "{}.txt".format(prev_name))
                write_results(result_filename, results)
                results = []

        img_path = image_path_from_loader(val_loader, img_file_name)
        start = time.time()
        predict_kwargs = {
            "source": img_path,
            "conf": args.conf,
            "iou": args.nms,
            "classes": args.yolo_classes,
            "device": args.device,
            "half": args.fp16,
            "verbose": False,
        }
        if args.tsize is not None:
            predict_kwargs["imgsz"] = args.tsize
        yolo_results = model.predict(**predict_kwargs)
        infer_end = time.time()
        dets = yolo_dets(yolo_results[0], args.yolo_classes)
        append_coco_predictions(data_list, dets, image_id_from_ids(ids), val_loader.dataset)

        img_h = int(scalar(info_imgs[0]))
        img_w = int(scalar(info_imgs[1]))
        online_targets = tracker.update(dets, [img_h, img_w], [img_h, img_w])

        online_tlwhs = []
        online_ids = []
        online_scores = []
        for target in online_targets:
            tlwh = target.tlwh
            tid = target.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(target.score)
        results.append((frame_id, online_tlwhs, online_ids, online_scores))

        track_end = time.time()
        if cur_iter < len(val_loader) - 1:
            inference_time += infer_end - start
            track_time += track_end - infer_end

    if len(results) != 0:
        result_filename = os.path.join(results_folder, "{}.txt".format(video_names[video_id]))
        write_results(result_filename, results)

    logger.info("\n" + detection_summary(val_loader, data_list, inference_time, track_time, n_samples))
    return results_folder


def evaluate_mot_results(exp, args, results_folder):
    mm.lap.default_solver = "lap"

    if exp.val_ann == "val_half.json":
        gt_type = "_val_half"
    else:
        gt_type = ""

    print("gt_type", gt_type)
    if args.mot20:
        gtfiles = glob.glob(os.path.join("datasets/MOT20/train", "*/gt/gt{}.txt".format(gt_type)))
    else:
        gtfiles = glob.glob(os.path.join("datasets/mot/train", "*/gt/gt{}.txt".format(gt_type)))
    print("gt_files", gtfiles)
    tsfiles = [
        f for f in glob.glob(os.path.join(results_folder, "*.txt"))
        if not os.path.basename(f).startswith("eval")
    ]

    logger.info("Found {} groundtruths and {} test files.".format(len(gtfiles), len(tsfiles)))
    logger.info("Available LAP solvers {}".format(mm.lap.available_solvers))
    logger.info("Default LAP solver '{}'".format(mm.lap.default_solver))
    logger.info("Loading files.")

    gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=-1)) for f in tsfiles])

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
    change_fmt_list = [
        "num_false_positives", "num_misses", "num_switches", "num_fragmentations",
        "mostly_tracked", "partially_tracked", "mostly_lost",
    ]
    for key in change_fmt_list:
        fmt[key] = fmt["mota"]
    print(mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))

    metrics = mm.metrics.motchallenge_metrics + ["num_objects"]
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    logger.info("Completed")


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        base_name = exp.exp_name if exp.exp_name else "yolo_bytetrack"
        args.experiment_name = "{}_yolo".format(base_name)

    if args.devices is not None and args.devices > 1:
        raise ValueError("tools/track_yolo.py currently supports single-device evaluation only.")

    results_folder = evaluate_yolo(exp, args)
    evaluate_mot_results(exp, args, results_folder)
