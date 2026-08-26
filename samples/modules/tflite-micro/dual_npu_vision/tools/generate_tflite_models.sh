#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
APP_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
ET_APP_DIR=$(cd -- "$APP_DIR/../../executorch/dual_npu_vision" && pwd)
MLEK_ROOT=$(west list -f '{abspath}' alif-mlek)
DOWNLOADS_DIR=${1:-"$(west topdir)/tflm-dual-model-resources"}
PYTHON=${PYTHON:-python3}
VELA=${VELA:-vela}
mkdir -p "$DOWNLOADS_DIR/u55" "$DOWNLOADS_DIR/u85"

curl -L --fail \
  https://raw.githubusercontent.com/emza-vs/ModelZoo/master/Models/Object_detection/SSD/ssd_slim_120x160x1_v1_int8.tflite \
  --output "$DOWNLOADS_DIR/ssd_slim_source.tflite"
curl -L --fail \
  https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth \
  --output "$DOWNLOADS_DIR/mobilenet_v2.pth"

# Create a common PyTorch SSD checkpoint from the trained public artifact.
# Both the TFLite and ExecuTorch exporters consume this checkpoint.
"$PYTHON" "$ET_APP_DIR/tools/import_ssd_slim_tflite.py" \
  --source "$DOWNLOADS_DIR/ssd_slim_source.tflite" \
  --output "$DOWNLOADS_DIR/comparable_ssd_slim.pth"

"$PYTHON" "$ET_APP_DIR/tools/export_comparable_tflite.py" ssd-slim \
  --weights "$DOWNLOADS_DIR/comparable_ssd_slim.pth" \
  --output "$DOWNLOADS_DIR/comparable_ssd_slim.tflite"
"$PYTHON" "$ET_APP_DIR/tools/export_mobilenet_v2_keras_tflite.py" \
  --weights "$DOWNLOADS_DIR/mobilenet_v2.pth" \
  --output "$DOWNLOADS_DIR/mobilenet_v2_imagenet.tflite" \
  --labels-output "$APP_DIR/models/labels_imagenet_1000.txt"

"$VELA" "$DOWNLOADS_DIR/comparable_ssd_slim.tflite" \
  --output-dir "$DOWNLOADS_DIR/u55" --accelerator-config ethos-u55-256 \
  --optimise Performance --config "$MLEK_ROOT/scripts/vela/ensemble_vela.ini" \
  --system-config RTSS_HP_SRAM_MRAM --memory-mode Shared_Sram
"$VELA" "$DOWNLOADS_DIR/mobilenet_v2_imagenet.tflite" \
  --output-dir "$DOWNLOADS_DIR/u85" --accelerator-config ethos-u85-256 \
  --optimise Performance --config "$MLEK_ROOT/scripts/vela/ensemble_vela.ini" \
  --system-config Ethos_U85_SRAM_MRAM --memory-mode Shared_Sram

cp "$DOWNLOADS_DIR/u55/comparable_ssd_slim_vela.tflite" \
   "$APP_DIR/models/comparable_ssd_slim_u55_256.tflite"
cp "$DOWNLOADS_DIR/u85/mobilenet_v2_imagenet_vela.tflite" \
   "$APP_DIR/models/mobilenet_v2_imagenet_u85_256.tflite"

echo "Updated the Vela TFLite models in $APP_DIR/models"
