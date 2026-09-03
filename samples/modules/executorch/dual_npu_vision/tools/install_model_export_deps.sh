#!/bin/sh
set -eu

if [ "$#" -gt 1 ]; then
  echo "usage: $0 [python]" >&2
  exit 2
fi

sample_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
workspace=$(CDPATH= cd -- "$sample_dir/../../../../.." && pwd)
python=${1:-"$workspace/.venv-executorch/bin/python"}

if [ ! -x "$python" ]; then
  echo "ExecuTorch Python environment not found: $python" >&2
  exit 1
fi

# ExecuTorch's pinned Vela 5.0.0 needs FlatBuffers 24.3.25, while the pinned
# TOSA Tools wheel declares FlatBuffers 25.2.10. The serializer APIs used by
# this exporter work with Vela's version. Install TOSA Tools without resolving
# its FlatBuffers dependency so pip does not reject the otherwise validated
# combination.
"$python" -m pip install \
  "numpy==2.1.3" \
  "flatbuffers==24.3.25" \
  "ethos-u-vela==5.0.0" \
  "jsonschema==4.24.0" \
  "ml-dtypes==0.5.1" \
  "semver==3.0.4" \
  "tensorflow==2.20.0"
"$python" -m pip install --no-deps --force-reinstall "tosa-tools==2026.2.1"

"$python" -c \
  'import flatbuffers, tosa_serializer; import ethosu.vela.api; import tensorflow; print("Model export dependencies are ready; FlatBuffers", flatbuffers.__version__)'
