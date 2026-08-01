#!/usr/bin/env bash
set -euo pipefail

seed="${1:-2022}"
gpu="${2:-0}"

python run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --model_id "ettm2_h96_tifo_seed${seed}" \
  --model iTransformer \
  --method tifo \
  --data ETTm2 \
  --root_path ./dataset/ETT-small \
  --data_path ETTm2.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 2 \
  --n_heads 8 \
  --factor 3 \
  --filter_dim 512 \
  --tifo_variant hermitian_raw \
  --tifo_dropout 0.5 \
  --tifo_residual_alpha 1.0 \
  --tifo_zero_pad_ratio 0.0 \
  --train_epochs 30 \
  --patience 5 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --random_seed "${seed}" \
  --itr 1 \
  --gpu "${gpu}" \
  --num_workers 0
