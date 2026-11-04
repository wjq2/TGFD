# Metric Sources and Third-party License

## DND21 ROC/AUC

ROC/AUC uses scikit-learn and follows the signal-positive definition in
Guo and Delbruck, *Low Cost and Latency Event Camera Background Activity
Denoising*, TPAMI 45(1), 785-795, 2023.
https://doi.org/10.1109/TPAMI.2022.3152999

## E-MLB ESR/MESR

`tgfd/evaluation/metrics.py` adapts the original `EventStructuralRatio`
formula to NumPy event arrays. The source is:

https://github.com/KugaMaxx/cuke-emlb/blob/105a1c36c368f10ff63b4a6f862992ae8842628d/python/src/utils/metric.py

Copyright (c) 2023 Kuga Maxx. The original MIT License is reproduced below.
This third-party license covers the adapted metric; it does not assign a
license to the authors' TGFD algorithm.

```text
MIT License

Copyright (c) 2023 Kuga Maxx

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
