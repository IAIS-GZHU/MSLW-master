#!/usr/bin/env bash
set -euo pipefail

python /path/to/MSLW/generate.py \
  --model-id /path/to/stable-diffusion-v1-4 \
  --prompt-path /path/to/prompts/Stable-Diffusion-Prompts.csv \
  --prompt-column Prompt \
  --generator-path /path/to/MSLW/checkpoints/encoder_v1_48.pth \
  --image-dir /path/to/watermarked-images/exp1 \
  --message-dir /path/to/watermark-messages/exp1 \
  --start 0 \
  --end 100 \
  --batch-size 8 \
  --bit-length 48 \
  --tau 150 \
  --phi 0.6 \
  --guidance-scale 7.5 \
  --num-steps 50 \
  --seed 2026 \
  --prediction-type epsilon
