# TGFD

Companion repository for **Topology-Guided Fast Denoising (TGFD)** for event streams.

## Overview

TGFD extracts a topology prior from 2D event aggregations and maps it back to
the original event stream to guide event-by-event denoising.

- **TGFD-Lite:** filters events using only the mapped topology prior.
- **TGFD:** uses the topology prior to guide event-by-event denoising with temporal support.

This release includes the complete TGFD and TGFD-Lite denoising pipeline,
windowing strategies and evaluation tools.

## Environment

Use Python 3.10 or newer, with NumPy, SciPy, Numba and scikit-learn.

```bash
pip install -r requirements.txt
```

For AEDAT4 files, also install `aedat` with `pip install aedat`.

## Datasets

- **DND21:** available through [EDformer](https://github.com/NJUVISION/EDformer#datasets-preparation).
- **E-MLB:** available from the [original project](https://github.com/KugaMaxx/cuke-emlb#datasets).
- **DVSNOISE20:** available through [EDnCNN](https://github.com/bald6354/edncnn#dvsnoise20-dataset-optional).

Obtain recordings from their original providers and place them in `data/`.

## Usage

```bash
python run.py denoise \
  --input data/events.npy \
  --columns label,t,x,y,p \
  --time-unit us \
  --config configs/default.json \
  --output-dir outputs/denoised
```

Add `--mode lite` for TGFD-Lite.

See [usage instructions](docs/usage.md) for more details.

## Contact

If you have any questions about our work or code, please email [Jiaqi Wang](mailto:wangjiaqi02@hotmail.com).
