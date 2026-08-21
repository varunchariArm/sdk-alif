#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
APP_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
WORKSPACE=$(west topdir)
MLEK_ROOT=$(west list -f '{abspath}' alif-mlek)
DOWNLOADS_DIR=${1:-"$WORKSPACE/tflm-model-resources"}
PYTHON=${PYTHON:-python3}

"$PYTHON" "$MLEK_ROOT/set_up_default_resources.py" \
  --downloads-dir "$DOWNLOADS_DIR" \
  --ml-frameworks tflm \
  --use-case object_detection \
  --use-case vww \
  --parallel 4

cp "$DOWNLOADS_DIR/object_detection/yolo-fastest_192_face_v4_vela_H256.tflite" \
   "$APP_DIR/models/yolo_fastest_face_u55_256.tflite"
cp "$DOWNLOADS_DIR/vww/vww4_128_128_INT8_vela_Z256.tflite" \
   "$APP_DIR/models/vww_u85_256.tflite"

echo "Updated the Vela TFLite models in $APP_DIR/models"
