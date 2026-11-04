# Usage Notes

## Input

Use an NPY array with one event per row. Set `--columns` to match the column order and `--time-unit` to `us`, `ms`, or `s`. Labels are required only for ROC/AUC evaluation.

## Run

From the repository root, run:

```bash
python run.py denoise \
  --input data/events.npy \
  --columns label,t,x,y,p \
  --time-unit us \
  --config configs/default.json \
  --output-dir outputs/denoised
```

Add `--mode lite` for TGFD-Lite. Use `python run.py COMMAND --help` for additional options.

## Output

By default, denoising saves per-event scores. Supply `--threshold` with a numeric value to also save retained events and their retention mask. Use a new output directory for each run.
