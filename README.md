# TGFD

Companion repository for **Topology-Guided Fast Denoising (TGFD)** for event streams.

## Overview

TGFD extracts a topology prior from 2D event aggregations and maps it back to
the original event stream to guide event-by-event denoising.

- **TGFD-Lite:** filters events using only the mapped topology prior.
- **TGFD:** uses the topology prior to guide event-by-event denoising with temporal support.


## Environment

Use Python 3.10 or newer, with NumPy, SciPy, Numba and scikit-learn.

```bash
pip install -r requirements.txt
```

For AEDAT4 files, also install `aedat` with `pip install aedat`.


## Datasets

- **DND21:** [Dataset and project](https://github.com/SensorsINI/dnd21).  
  Paper: [Low Cost and Latency Event Camera Background Activity Denoising](https://doi.org/10.1109/TPAMI.2022.3152999).

- **E-MLB:** [Dataset and project](https://github.com/KugaMaxx/cuke-emlb#datasets).  
  Paper: [E-MLB: Multilevel Benchmark for Event-Based Camera Denoising](https://doi.org/10.1109/TMM.2023.3260638).

- **DVSNOISE20:** [Dataset and project](https://github.com/bald6354/edncnn#dvsnoise20-dataset-optional).\
  Paper: [Event Probability Mask (EPM) and Event Denoising Convolutional Neural Network (EDnCNN) for Neuromorphic Cameras](https://doi.org/10.1109/CVPR42600.2020.00177).



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



## Contact

If you have any questions about our work or code, please email wangjiaqi02@hotmail.com.
