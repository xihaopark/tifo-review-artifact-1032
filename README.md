# TIFO

Reference implementation of the Time-Invariant Frequency Operator for
long-term time-series forecasting.

Supported backbones: DLinear, iTransformer, and PatchTST.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Data

```text
dataset/
  ETT-small/ETTh1.csv
  ETT-small/ETTh2.csv
  ETT-small/ETTm1.csv
  ETT-small/ETTm2.csv
  electricity/electricity.csv
  traffic/traffic.csv
  weather/weather.csv
```

## Run

```bash
bash scripts/run_ettm2_h96.sh 2022 0
```

The first argument is the random seed and the second is the GPU index.
