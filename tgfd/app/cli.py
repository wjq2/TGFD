"""Run TGFD denoising and evaluation on event recordings."""
import argparse
from dataclasses import fields, replace
import json
from pathlib import Path
import sys

import numpy as np

from tgfd.core.config import Config
from tgfd.app.runner import RunOptions, run_files


def common(parser):
    parser.add_argument("--input", required=True, nargs="+", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--columns", help="Actual NPY column names, e.g. t,x,y,p,label")
    parser.add_argument("--time-unit", choices=["s", "ms", "us"])
    parser.add_argument("--mode", choices=["full", "lite"], default="full")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--windowing",
        choices=["aow", "fti", "fec"],
        help="Override the configuration window strategy; default is AOW",
    )


def positive(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("threshold must be finite and positive")
    return value


def build_parser():
    """Construct the parser without running any file processing."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    denoise = sub.add_parser(
        "denoise", help="Save continuous scores and optionally filtered original rows"
    )
    common(denoise)
    denoise.add_argument("--threshold", type=positive)
    denoise.add_argument("--comparison", choices=["ge", "gt"], default="ge")

    evaluate = sub.add_parser(
        "evaluate", help="Evaluate per-file ROC AUC from continuous event scores"
    )
    common(evaluate)
    evaluate.add_argument("--signal-label", required=True, type=float)

    mesr = sub.add_parser("mesr", help="Compute MESR for raw and filtered events")
    common(mesr)
    mesr.add_argument("--threshold", required=True, type=positive)
    mesr.add_argument("--comparison", choices=["ge", "gt"], default="gt")
    mesr.add_argument("--block-size", type=int, default=30000)
    mesr.add_argument(
        "--reference-count",
        type=int,
        help="Optional override; default floor(2*block-size/3) follows E-MLB",
    )
    mesr.add_argument(
        "--legacy-fallback",
        action="store_true",
        help="Explicitly reproduce raw-MESR substitution when filtered MESR is undefined",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    config = Config.load(args.config)
    if args.windowing is not None:
        config = replace(config, windowing=args.windowing)

    options = {
        field.name: getattr(args, field.name)
        for field in fields(RunOptions)
        if hasattr(args, field.name)
    }
    invocation = {
        k: (
            [str(p) for p in v]
            if k == "input"
            else str(v) if isinstance(v, Path) else v
        )
        for k, v in vars(args).items()
    }

    run_files(
        args.input,
        config,
        args.output_dir,
        RunOptions(**options),
        invocation=invocation,
        report=lambda row: print(json.dumps(row, allow_nan=False)),
    )
    return 0


def entrypoint():
    """Command-line error handling shared by both entry points."""
    try:
        raise SystemExit(main())
    except (ValueError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
