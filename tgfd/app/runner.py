"""Reusable file execution, independent of command-line argument parsing."""
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Callable

from tgfd.core.config import Config
from tgfd.evaluation import evaluate_mesr
from tgfd.core.filtering import make_keep_mask
from tgfd.io.events import load_events, signal_labels
from tgfd.core.pipeline import score_events, warmup
from tgfd.io.results import (
    prepare_output_directory,
    save_filtered,
    save_json,
    save_roc,
    save_scores,
    save_summary,
)
from tgfd.evaluation.roc import compute_roc


@dataclass(frozen=True)
class RunOptions:
    """File-processing settings, separate from the numerical Config."""

    command: str = "denoise"
    columns: str | None = None
    time_unit: str | None = None
    mode: str = "full"
    signal_label: float | None = None
    threshold: float | None = None
    comparison: str = "ge"
    block_size: int = 30000
    reference_count: int | None = None
    legacy_fallback: bool = False


def _validate_options(options):
    """Validate settings for denoising and evaluation."""
    if options.command not in ("denoise", "evaluate", "mesr"):
        raise ValueError("command must be denoise, evaluate or mesr")
    if options.mode not in ("full", "lite"):
        raise ValueError("mode must be full or lite")
    if options.comparison not in ("ge", "gt"):
        raise ValueError("comparison must be ge or gt")
    if options.command == "evaluate" and options.signal_label is None:
        raise ValueError("ROC evaluation requires an explicit signal_label")
    if options.command == "mesr":
        if options.threshold is None:
            raise ValueError("MESR requires an explicit threshold")
        if options.block_size < 2 or (
            options.reference_count is not None
            and not 0 < options.reference_count < options.block_size
        ):
            raise ValueError("MESR requires block-size > reference-count > 0 and block-size >= 2")
    if options.threshold is not None:
        make_keep_mask([], options.threshold, options.comparison)


def run_files(inputs, config: Config, output_dir, options: RunOptions | None = None,
              *, invocation=None, report: Callable[[dict], None] | None = None):
    """Process explicit input paths, save outputs and return per-file records.

    No command-line Namespace is needed. Use score_events for an in-memory
    array, or this function for the same file workflow as the CLI.
    """
    options = RunOptions() if options is None else options
    paths = [Path(path) for path in inputs]
    if not paths:
        raise ValueError("at least one input file is required")
    _validate_options(options)
    destination = prepare_output_directory(output_dir, paths)
    if invocation is None:
        invocation = {
            "input": [str(path) for path in paths],
            "output_dir": str(destination),
            **asdict(options),
        }
    save_json(destination / "run_config.json", {
        "arguments": invocation,
        "config": config.to_dict(),
    })
    warmup()
    rows = []
    for path in paths:
        data = load_events(path, options.columns, options.time_unit)
        y = signal_labels(data.labels, options.signal_label) if options.command == "evaluate" else None
        start = time.perf_counter()
        scores = score_events(data.events, config, options.mode)
        seconds = time.perf_counter() - start
        row = {
            "file": str(path), "n_events": len(scores), "mode": options.mode,
            "seconds_scoring": seconds, "windowing": config.windowing,
        }
        if y is not None:
            roc = compute_roc(y, scores)
            row["auc"] = roc["auc"]
            save_roc(destination, path.stem, roc)
        if options.command == "denoise":
            save_scores(destination, path.stem, scores, data, config, options.mode)
            if options.threshold is not None:
                mask = make_keep_mask(scores, options.threshold, options.comparison)
                save_filtered(destination, path.stem, data.raw, mask)
                row.update(threshold=options.threshold, comparison=options.comparison,
                           n_kept=int(mask.sum()))
        if options.command == "mesr":
            row.update(evaluate_mesr(
                data.events, scores, (config.width, config.height),
                threshold=options.threshold, comparison=options.comparison,
                block_size=options.block_size, reference_count=options.reference_count,
                legacy_fallback=options.legacy_fallback,
            ))
        rows.append(row)
        if report is not None:
            report(row.copy())
    save_summary(destination, rows)
    return rows
