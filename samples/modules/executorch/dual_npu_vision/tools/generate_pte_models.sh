#!/bin/sh
set -eu

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <vela.ini> <ssd-slim.pth> <mobilenet-v2.pth>" >&2
  exit 2
fi

sample_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
workspace=$(CDPATH= cd -- "$sample_dir/../../../../.." && pwd)
python="$workspace/.venv-executorch/bin/python"

if [ ! -x "$python" ]; then
  echo "ExecuTorch Python environment not found: $python" >&2
  exit 1
fi

export PYTHONPATH="$workspace/modules/lib${PYTHONPATH:+:$PYTHONPATH}"

"$python" "$sample_dir/tools/export_torchvision_models.py" comparable-ssd \
  --config "$1" \
  --weights "$2" \
  --output "$sample_dir/models/comparable_ssd_slim_u55.pte"

"$python" "$sample_dir/tools/export_torchvision_models.py" mobilenet-v2 \
  --config "$1" \
  --weights "$3" \
  --mv2-classes 1000 \
  --labels-output "$sample_dir/models/labels_imagenet_1000.txt" \
  --output "$sample_dir/models/mobilenet_v2_imagenet_u85.pte"

echo "Generated both PTE files in $sample_dir/models"
