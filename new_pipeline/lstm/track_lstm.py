"""Run MOT evaluation with BYTETrackerLSTM without modifying ByteTrack files.

This entrypoint mirrors tools/track.py, but swaps the tracker created inside
MOTEvaluator from BYTETracker to new_pipeline.byte_tracker_lstm.BYTETrackerLSTM
at runtime.

Example:
python new_pipeline/track_lstm.py \
    -f exps/example/mot/yolox_m_mix_det.py \
    -c pretrained/bytetrack_m_mot17.pth.tar \
    -b 1 -d 1 --fp16 --fuse \
    --experiment-name eval_lstm \
    --lstm_ckpt checkpoints/lstm/best.pth
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from new_pipeline.lstm.byte_tracker_lstm import BYTETrackerLSTM
from tools.track import main as track_main
from tools.track import make_parser
from yolox.core import launch
from yolox.evaluators import mot_evaluator
from yolox.exp import get_exp


def _add_arg_once(parser, *names, **kwargs):
    existing = {opt for action in parser._actions for opt in action.option_strings}
    if any(name in existing for name in names):
        return
    parser.add_argument(*names, **kwargs)


def add_lstm_args(parser):
    _add_arg_once(
        parser,
        "--lstm_ckpt",
        default="checkpoints/lstm/best.pth",
        type=str,
        help="LSTM residual checkpoint",
    )
    _add_arg_once(parser, "--lstm_hidden_size", default=128, type=int)
    _add_arg_once(parser, "--lstm_num_layers", default=2, type=int)
    _add_arg_once(parser, "--lstm_alpha0", default=1.0, type=float)
    _add_arg_once(parser, "--lstm_beta", default=0.3, type=float)
    _add_arg_once(parser, "--lstm_device", default=None, type=str)
    _add_arg_once(parser, "--assoc_ckpt", default=None, type=str)
    _add_arg_once(parser, "--assoc_weight", default=0.35, type=float)
    _add_arg_once(parser, "--assoc_seq_len", default=16, type=int)
    _add_arg_once(parser, "--assoc_hidden_size", default=128, type=int)
    _add_arg_once(parser, "--assoc_num_layers", default=1, type=int)
    _add_arg_once(parser, "--assoc_dropout", default=0.1, type=float)
    _add_arg_once(parser, "--assoc_mlp_hidden", default=128, type=int)
    _add_arg_once(parser, "--assoc_min_history", default=2, type=int)
    return parser


def build_lstm_tracker(args):
    return BYTETrackerLSTM(
        args,
        lstm_ckpt=args.lstm_ckpt,
        hidden_size=args.lstm_hidden_size,
        num_layers=args.lstm_num_layers,
        alpha0=args.lstm_alpha0,
        beta=args.lstm_beta,
        assoc_ckpt=args.assoc_ckpt,
        assoc_weight=args.assoc_weight,
        assoc_seq_len=args.assoc_seq_len,
        assoc_hidden_size=args.assoc_hidden_size,
        assoc_num_layers=args.assoc_num_layers,
        assoc_dropout=args.assoc_dropout,
        assoc_mlp_hidden=args.assoc_mlp_hidden,
        assoc_min_history=args.assoc_min_history,
        device=args.lstm_device,
    )


def patch_evaluator_tracker():
    # MOTEvaluator.evaluate() calls the module-level BYTETracker symbol.
    # Replacing that symbol here keeps yolox/evaluators/mot_evaluator.py intact.
    mot_evaluator.BYTETracker = build_lstm_tracker


if __name__ == "__main__":
    patch_evaluator_tracker()

    parser = add_lstm_args(make_parser())
    args = parser.parse_args()

    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name + "_lstm"

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        track_main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args, num_gpu),
    )
