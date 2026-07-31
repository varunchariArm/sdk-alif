#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <yolo-u55-vela.npz> <vww-u85-vela.npz>" >&2
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

"$python" "$sample_dir/tools/export_yolo_u55_pte.py" \
  --npz "$1" \
  --output "$sample_dir/models/yolo_fastest_face_u55_256.pte"

"$python" "$sample_dir/tools/export_vww_u85_pte.py" \
  --npz "$2" \
  --output "$sample_dir/models/vww_u85_256.pte"

echo "Generated both PTE files in $sample_dir/models"
