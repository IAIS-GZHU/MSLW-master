#!/usr/bin/env bash
set -euo pipefail

python /path/to/MSLW/decode.py \
  --experiment-id exp1 \
  --model-id /path/to/stable-diffusion-v1-4 \
  --image-dir /path/to/watermarked-images/exp1 \
  --message-dir /path/to/watermark-messages/exp1 \
  --extractor-path /path/to/MSLW/checkpoints/decoder_v1_48.pth \
  --result-path /path/to/decoding-results/decode.txt \
  --batch-size 8 \
  --bit-length 48 \
  --image-size 512 \
  --num-workers 4
