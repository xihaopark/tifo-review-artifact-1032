# TIFO: Time-Invariant Frequency Operator

This repository contains the anonymous reference implementation of TIFO. TIFO
is a dataset-conditioned spectral input adapter: it computes a
frequency/channel statistic from training windows, maps that statistic to
globally shared spectral gains, reweights each input in the frequency domain,
and passes the reconstructed sequence to an otherwise unchanged forecasting
backbone.

The implementation includes both the historical full-FFT path retained for
result reproducibility and the real-reconstruction rFFT/iRFFT variants used in
the revised evaluation.

## Artifact scope

The release is intentionally limited to the code needed to inspect and run the
proposed operator:

- the standalone TIFO and matched `ori` control paths;
- DLinear, iTransformer, and PatchTST forecasting backbones;
- the seven long-term forecasting datasets described in the paper; and
- run-manifest generation for seeds, resolved arguments, dataset hashes, and
  final metrics.

The comparison methods in the paper are third-party baselines and are not
redistributed here. `TIFO*` in the paper denotes the submitted composition of
TIFO with SAN; it is not a separate proposed operator. This repository is the
reference implementation for standalone TIFO and does not claim to regenerate
third-party baseline cells.

Datasets, checkpoints, generated logs, internal experiment orchestration,
review material, manuscript sources, and Git history are excluded.

## Environment

The reported runs use Python 3.10 and CUDA-enabled PyTorch. Install the remaining
dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Data

Download the standard ETT, Electricity, Traffic, and Weather forecasting data
used by Time-Series-Library and arrange them as follows:

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

Dataset files are not redistributed in this archive.

## Quick verification

The command-line entry point can be checked without downloading data:

```bash
python run.py --help
```

## Frozen representative run

The following is the validation-selected ETTm2, horizon-96 standalone TIFO
configuration used for the reported three-seed evaluation. Replace `2022` by
`2021` or `2023` for the other reported seeds.

```bash
python run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --model_id ettm2_h96_tifo_seed2022 \
  --model iTransformer \
  --method tifo \
  --data ETTm2 \
  --root_path ./dataset/ETT-small \
  --data_path ETTm2.csv \
  --features M \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 128 --d_ff 128 --e_layers 2 --n_heads 8 --factor 3 \
  --filter_dim 512 \
  --tifo_variant hermitian_raw \
  --tifo_dropout 0.5 \
  --tifo_residual_alpha 1.0 \
  --tifo_zero_pad_ratio 0.0 \
  --train_epochs 30 --patience 5 \
  --batch_size 32 --learning_rate 0.0001 \
  --random_seed 2022 --itr 1 --gpu 0 --num_workers 0
```

The same command is available as `scripts/run_ettm2_h96.sh`; pass a seed and GPU
index as its first and second arguments, respectively:

```bash
bash scripts/run_ettm2_h96.sh 2022 0
```

For the matched backbone control, use the identical command with
`--method ori`. The `ori` path bypasses TIFO without advancing the random-number
stream used to initialize the shared backbone.

## TIFO controls

- `--tifo_variant historical`: result-preserving full-FFT implementation.
- `--tifo_variant hermitian_raw`: one-sided rFFT/iRFFT with raw training-window
  statistics.
- `--tifo_variant hermitian_aligned`: rFFT/iRFFT with statistics aligned to the
  backbone input normalization.
- `--tifo_score_mode data`: use the training-set statistic.
- `--tifo_score_mode permuted`: preserve per-channel score marginals while
  breaking frequency correspondence with a local fixed RNG.
- `--tifo_score_mode ones`: remove score variation.

The statistic is computed from the training loader only. TIFO parameters are
optimized jointly with the backbone under forecasting MSE; they are spectral
gains and are not kernel eigenvalues.

## Output and reproducibility

Every run writes the resolved command, random seed, dataset hash, code state,
and final metrics to a run manifest. Reported values use seeds 2021--2023 and
sample standard deviation. Validation-selected configurations are frozen before
the corresponding final three-seed aggregation.
