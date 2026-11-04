# Usage Notes

Run commands from the repository root. `python run.py` and `python -m tgfd`
use the same interface. Use `python run.py COMMAND --help` for command options.

## Input Formats

NPY files must contain a numeric array, with one event per row. Set the column
order and timestamp unit explicitly:

| Actual file columns | CLI argument |
| --- | --- |
| `[t, x, y, p]` | `--columns t,x,y,p` |
| `[t, x, y, p, label]` | `--columns t,x,y,p,label` |
| `[label, t, x, y, p]` | `--columns label,t,x,y,p` |

Use `--time-unit us`, `ms`, or `s`. Labels are used only by ROC/AUC evaluation;
denoising does not use them. Coordinates must be integer-valued and fit the
configured sensor resolution. Polarity may use `{-1, 1}` or `{0, 1}`, with `1`
representing positive polarity.

AEDAT4 files are read with the optional `aedat` package. Other event file formats
must first be converted to a numeric NPY array.

To inspect the values in an NPY recording:

```bash
python run.py inspect --input data/events.npy
```

## Parameters

Edit a JSON file in `configs/`, or create a `Config` in Python:

| Parameter | Meaning |
| --- | --- |
| `width`, `height` | Sensor resolution |
| `windowing` | `aow`, `fti`, or `fec` |
| `dt_ms` | Base time interval in milliseconds, used by AOW and FTI |
| `n_min` | Minimum event count used by AOW |
| `n_events` | Event count per FEC window |
| `alpha` | Edge pruning threshold factor |
| `min_component_size` | Minimum retained component size |
| `radius` | Spatial support radius for full TGFD |
| `tau_us` | Temporal weighting scale in microseconds |
| `eps` | Numerical stabilizer |

The three shipped configurations differ only in their window strategy. Full
TGFD and TGFD-Lite use the same windowing and topology implementation.

## Python Calls

```python
import numpy as np
from tgfd import Config, score_events, make_keep_mask
from tgfd.io import load_events

data = load_events("data/events.npy", columns="label,t,x,y,p", time_unit="us")
config = Config.load("configs/default.json")
scores = score_events(data.events, config, mode="full")
keep = make_keep_mask(scores, threshold=10)
np.save("events_denoised.npy", data.raw[keep])
```

Match `columns` to your file. The threshold is an example. Set `mode="lite"`
for TGFD-Lite. If the input is already an `(N, 4)` array containing
`[t_us, x, y, p]`, pass it directly to `score_events`.

For the file workflow from another Python program, use
`tgfd.app.runner.RunOptions` and `tgfd.app.runner.run_files`.

## Batch Processing and Outputs

`--input` accepts multiple recording paths. In Bash, replace one path with
`data/*.npy` to process all NPY recordings in that folder. Use a new output
directory; input filenames must have distinct stems within a run.

| File | Contents |
| --- | --- |
| `*_scores.npz` | Scores, original row indices, and input/configuration metadata |
| `*_denoised.npy` | Retained original rows, saved when a threshold is supplied |
| `*_keep.npy` | Boolean retention mask, saved with filtered events |
| `*_roc.csv` | ROC thresholds, false positive rates, and true positive rates |
| `results.csv`, `results.json` | Per-recording summaries |
| `run_config.json` | Configuration and command arguments |

The denoising score command does not choose an operating threshold. The
README's denoising example supplies one so that filtered events are saved.
`--comparison ge` means `>=`; `--comparison gt` means `>`.

## Evaluation

ROC/AUC requires both signal and noise labels. Pass the raw signal class using
`--signal-label`. Larger scores indicate stronger signal support; all input
events, including events with zero scores, are included.

MESR follows the E-MLB formula and uses complete 30000-event blocks by default.
The trailing incomplete block is omitted. When no complete block exists,
MESR is reported as `null`. These metric blocks are separate from the
algorithm's time windows. See [credits](credits.md) for metric sources and
the original third-party license.

For a parameter sweep, the same evaluation interface also provides `sweep`:

```bash
python run.py sweep \
  --input data/events.npy \
  --columns t,x,y,p,label \
  --time-unit us \
  --config configs/default.json \
  --signal-label 0 \
  --parameter alpha \
  --values 0 0.5 1.0 1.5 2.0 \
  --output-dir outputs/alpha_sweep
```

## Code Organization

| Folder | Responsibility |
| --- | --- |
| `tgfd/core/` | Configuration, windowing, topology prior, temporal scoring, and filtering |
| `tgfd/io/` | Event loading and result saving |
| `tgfd/evaluation/` | ROC/AUC and ESR/MESR |
| `tgfd/app/` | Command parsing and coordination of file processing |
