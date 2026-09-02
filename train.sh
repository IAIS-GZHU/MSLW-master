#!/usr/bin/env bash
set -euo pipefail

python /path/to/MSLW/train.py \
  --model-id /path/to/stable-diffusion-v1-4 \
  --image-dir /path/to/Flickr8k/Images \
  --log-dir /path/to/training-logs \
  --log-name exp1 \
  --checkpoint-dir /path/to/training-checkpoints/exp1 \
  --max-epochs 6 \
  --attack-start-epoch 2 \
  --batch-size 4 \
  --bit-length 48 \
  --image-size 512 \
  --num-workers 4 \
  --train-ratio 0.95 \
  --learning-rate 0.0001 \
  --seed 2026 \
  --noise-transforms Identity Jpeg random_crop GaussianBlur GaussianNoise ColorJitter \
  --initial-noise-probabilities 1.0 0.0 0.0 0.0 0.0 0.0 \
  --attack-probabilities 0.1 0.25 0.25 0.1 0.15 0.15
