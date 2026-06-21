from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import os.path as osp
import shutil
from collections import defaultdict

import cv2

from torchreid.data import ImageDataset


class MOTReID(ImageDataset):
    dataset_dir = "mot_reid"

    def __init__(self, root="", **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        meta_path = osp.join(self.dataset_dir, "meta.json")
        self.check_before_run([self.dataset_dir, meta_path])

        with open(meta_path, "r") as f:
            meta = json.load(f)

        train = self._load_split(meta["train"])
        query = self._load_split(meta["query"])
        gallery = self._load_split(meta["gallery"])

        super(MOTReID, self).__init__(train, query, gallery, **kwargs)

    def _load_split(self, rows):
        data = []
        for row in rows:
            img_path = osp.join(self.dataset_dir, row["path"])
            data.append((img_path, int(row["pid"]), int(row["camid"])))
        return data


def parse_args():
    parser = argparse.ArgumentParser("Build MOT17 ReID crops for torchreid")
    parser.add_argument("--mot-root", default="datasets/mot", help="MOT dataset root")
    parser.add_argument("--output", default="datasets/mot_reid", help="output ReID dataset directory")
    parser.add_argument("--detector", default="FRCNN", choices=["FRCNN", "DPM", "SDP", "all"], help="MOT17 detector split to use")
    parser.add_argument("--eval-bases", nargs="+", default=["MOT17-11", "MOT17-13"], help="base sequences reserved for query/gallery")
    parser.add_argument("--frame-stride", type=int, default=5, help="keep one crop every N frames per identity")
    parser.add_argument("--min-visibility", type=float, default=0.3, help="minimum MOT visibility")
    parser.add_argument("--min-height", type=int, default=40, help="minimum crop height")
    parser.add_argument("--min-width", type=int, default=12, help="minimum crop width")
    parser.add_argument("--min-crops-per-id", type=int, default=3, help="drop identities with too few kept crops")
    parser.add_argument("--max-crops-per-id", type=int, default=200, help="cap kept crops per identity; 0 disables cap")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing output images and metadata")
    return parser.parse_args()


def sequence_base(seq_name):
    parts = seq_name.split("-")
    return "-".join(parts[:2])


def list_sequences(mot_root, detector):
    train_root = osp.join(mot_root, "train")
    seqs = []
    for name in sorted(os.listdir(train_root)):
        seq_dir = osp.join(train_root, name)
        if not osp.isdir(seq_dir):
            continue
        if detector != "all" and not name.endswith("-" + detector):
            continue
        if not osp.isfile(osp.join(seq_dir, "gt", "gt.txt")):
            continue
        seqs.append(name)
    return seqs


def parse_gt(seq_dir, min_visibility, min_width, min_height, frame_stride):
    items_by_identity = defaultdict(list)
    gt_path = osp.join(seq_dir, "gt", "gt.txt")
    with open(gt_path, "r") as f:
        for line in f:
            values = line.strip().split(",")
            if len(values) < 9:
                continue
            frame_id = int(float(values[0]))
            track_id = int(float(values[1]))
            x = float(values[2])
            y = float(values[3])
            w = float(values[4])
            h = float(values[5])
            mark = int(float(values[6]))
            cls = int(float(values[7]))
            visibility = float(values[8])

            if frame_stride > 1 and frame_id % frame_stride != 0:
                continue
            if mark == 0 or cls != 1:
                continue
            if visibility < min_visibility or w < min_width or h < min_height:
                continue
            items_by_identity[track_id].append((frame_id, x, y, w, h, visibility))
    return items_by_identity


def crop_box(img, box):
    _, x, y, w, h, _ = box
    img_h, img_w = img.shape[:2]
    x1 = max(0, min(img_w - 1, int(round(x))))
    y1 = max(0, min(img_h - 1, int(round(y))))
    x2 = max(0, min(img_w, int(round(x + w))))
    y2 = max(0, min(img_h, int(round(y + h))))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def make_identity_tasks(seq_name, track_id, boxes, seq_dir, out_dir, split_name, pid, camid, max_crops):
    if max_crops > 0 and len(boxes) > max_crops:
        step = max(len(boxes) // max_crops, 1)
        boxes = boxes[::step][:max_crops]

    tasks = []
    split_dir = osp.join(out_dir, split_name)
    ensure_dir(split_dir)
    for box in boxes:
        frame_id = box[0]
        rel_path = osp.join(
            split_name,
            "{:05d}_c{:03d}_{}_t{:04d}_f{:06d}.jpg".format(
                pid, camid + 1, seq_name, track_id, frame_id
            ),
        )
        tasks.append(
            {
                "seq_dir": seq_dir,
                "frame_id": frame_id,
                "box": box,
                "out_path": osp.join(out_dir, rel_path),
                "row": {"path": rel_path, "pid": pid, "camid": camid},
            }
        )
    return tasks


def write_crop_tasks(tasks):
    tasks_by_frame = defaultdict(list)
    for task in tasks:
        tasks_by_frame[(task["seq_dir"], task["frame_id"])].append(task)

    rows = []
    for (seq_dir, frame_id), frame_tasks in sorted(tasks_by_frame.items()):
        img_path = osp.join(seq_dir, "img1", "{:06d}.jpg".format(frame_id))
        img = cv2.imread(img_path)
        if img is None:
            continue
        for task in frame_tasks:
            crop = crop_box(img, task["box"])
            if crop is None:
                continue
            cv2.imwrite(task["out_path"], crop)
            rows.append(task["row"])
    return rows


def validate_eval_split(query_rows, gallery_rows):
    gallery_by_pid = defaultdict(set)
    for row in gallery_rows:
        gallery_by_pid[row["pid"]].add(row["camid"])

    invalid = []
    for row in query_rows:
        gallery_camids = gallery_by_pid.get(row["pid"], set())
        valid_gallery_camids = [camid for camid in gallery_camids if camid != row["camid"]]
        if len(valid_gallery_camids) == 0:
            invalid.append((row["pid"], row["camid"]))
    if invalid:
        raise RuntimeError(
            "Invalid ReID eval split: {} query identities have no gallery image with a different camid. "
            "Example: {}".format(len(invalid), invalid[:5])
        )


def build_dataset(args):
    mot_root = osp.abspath(args.mot_root)
    out_dir = osp.abspath(args.output)
    if osp.exists(osp.join(out_dir, "meta.json")) and not args.overwrite:
        raise FileExistsError("{} already exists; pass --overwrite to rebuild".format(osp.join(out_dir, "meta.json")))

    ensure_dir(out_dir)
    if args.overwrite:
        for split in ["bounding_box_train", "query", "bounding_box_test"]:
            split_dir = osp.join(out_dir, split)
            if osp.isdir(split_dir):
                shutil.rmtree(split_dir)
    for split in ["bounding_box_train", "query", "bounding_box_test"]:
        ensure_dir(osp.join(out_dir, split))

    seqs = list_sequences(mot_root, args.detector)
    if len(seqs) == 0:
        raise RuntimeError("No MOT sequences found in {}".format(osp.join(mot_root, "train")))

    eval_bases = set(args.eval_bases)
    train_tasks, query_tasks, gallery_tasks = [], [], []
    pid_map = {}
    camid_map = {seq: idx for idx, seq in enumerate(seqs)}
    gallery_camid_offset = len(camid_map)

    for seq_name in seqs:
        seq_dir = osp.join(mot_root, "train", seq_name)
        identities = parse_gt(
            seq_dir,
            args.min_visibility,
            args.min_width,
            args.min_height,
            args.frame_stride,
        )
        is_eval = sequence_base(seq_name) in eval_bases
        for track_id, boxes in sorted(identities.items()):
            boxes = sorted(boxes, key=lambda item: item[0])
            if len(boxes) < args.min_crops_per_id:
                continue
            key = "{}:{:04d}".format(seq_name, track_id)
            pid = pid_map.setdefault(key, len(pid_map))
            camid = camid_map[seq_name]

            if is_eval:
                query = boxes[:1]
                gallery = boxes[1:]
                query_camid = camid
                gallery_camid = camid + gallery_camid_offset
                query_tasks.extend(
                    make_identity_tasks(seq_name, track_id, query, seq_dir, out_dir, "query", pid, query_camid, 1)
                )
                gallery_tasks.extend(
                    make_identity_tasks(
                        seq_name,
                        track_id,
                        gallery,
                        seq_dir,
                        out_dir,
                        "bounding_box_test",
                        pid,
                        gallery_camid,
                        args.max_crops_per_id,
                    )
                )
            else:
                train_tasks.extend(
                    make_identity_tasks(
                        seq_name,
                        track_id,
                        boxes,
                        seq_dir,
                        out_dir,
                        "bounding_box_train",
                        pid,
                        camid,
                        args.max_crops_per_id,
                    )
                )

    train_rows = write_crop_tasks(train_tasks)
    query_rows = write_crop_tasks(query_tasks)
    gallery_rows = write_crop_tasks(gallery_tasks)
    validate_eval_split(query_rows, gallery_rows)

    meta = {
        "name": "mot_reid",
        "source": mot_root,
        "detector": args.detector,
        "frame_stride": args.frame_stride,
        "min_visibility": args.min_visibility,
        "eval_camid_note": "Query and gallery from single-camera MOT sequences use different pseudo camids so Market1501-style evaluation has valid cross-camera matches.",
        "train": train_rows,
        "query": query_rows,
        "gallery": gallery_rows,
    }
    meta_path = osp.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved {}".format(meta_path))
    print("train images: {}".format(len(train_rows)))
    print("query images: {}".format(len(query_rows)))
    print("gallery images: {}".format(len(gallery_rows)))
    print("identities: {}".format(len(pid_map)))
    print("cameras/sequences: {}".format(len(camid_map)))


if __name__ == "__main__":
    build_dataset(parse_args())
