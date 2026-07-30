#!/bin/sh
set -eu

sample_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
workspace=$(CDPATH= cd -- "$sample_dir/../../../../.." && pwd)
executorch="$workspace/modules/lib/executorch"
core_driver="$workspace/modules/ethos-u-core-driver-src"
patch="$sample_dir/patches/executorch-dual-npu.patch"
expected_core_driver=fe1efeff3381032d459216b63a78f658f100ad75

if [ ! -d "$executorch/.git" ]; then
  echo "ExecuTorch checkout not found at $executorch" >&2
  echo "Run 'west update' and initialize its submodules first." >&2
  exit 1
fi

if git -C "$executorch" apply --reverse --check "$patch" 2>/dev/null; then
  echo "ExecuTorch dual-NPU patch is already applied."
elif git -C "$executorch" apply --check "$patch"; then
  git -C "$executorch" apply "$patch"
  echo "Applied ExecuTorch dual-NPU patch."
else
  echo "ExecuTorch patch does not apply cleanly; inspect module changes." >&2
  exit 1
fi

if [ ! -d "$core_driver/.git" ]; then
  echo "Missing $core_driver" >&2
  echo "Clone the Ethos-U core driver at $expected_core_driver there." >&2
  exit 1
fi

actual_core_driver=$(git -C "$core_driver" rev-parse HEAD)
if [ "$actual_core_driver" != "$expected_core_driver" ]; then
  echo "Warning: Ethos-U core driver is $actual_core_driver" >&2
  echo "Validated revision is $expected_core_driver" >&2
fi

if [ ! -d "$workspace/modules/cmsis-nn-src" ]; then
  echo "Missing CMSIS-NN checkout at $workspace/modules/cmsis-nn-src" >&2
  exit 1
fi

echo "Workspace dependencies are ready."
