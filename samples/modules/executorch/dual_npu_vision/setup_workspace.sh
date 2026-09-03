#!/bin/sh
set -eu

sample_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
workspace=$(CDPATH= cd -- "$sample_dir/../../../../.." && pwd)
executorch="$workspace/modules/lib/executorch"
core_driver="$workspace/modules/ethos-u-core-driver-src"
zephyr="$workspace/zephyr"
executorch_patch="$sample_dir/patches/executorch-dual-npu.patch"
zephyr_patch="$sample_dir/patches/zephyr-sram1-placement.patch"
required_core_driver=b7cd193afde80afe8bbae9a26d2ca6586554f054

# west currently supplies Zephyr's older hal_ethos_u snapshot. This separate
# main checkout is required until that manifest revision contains the merged
# multi-variant implementation. The ancestor test accepts newer main commits.

if [ ! -d "$executorch/.git" ]; then
  echo "ExecuTorch checkout not found at $executorch" >&2
  echo "Run 'west update' and initialize its submodules first." >&2
  exit 1
fi

if git -C "$executorch" apply --reverse --check "$executorch_patch" 2>/dev/null; then
  echo "ExecuTorch dual-NPU patch is already applied."
elif git -C "$executorch" apply --check "$executorch_patch"; then
  git -C "$executorch" apply "$executorch_patch"
  echo "Applied ExecuTorch dual-NPU patch."
else
  echo "ExecuTorch patch does not apply cleanly; inspect module changes." >&2
  exit 1
fi

if [ ! -d "$zephyr/.git" ]; then
  echo "Zephyr checkout not found at $zephyr" >&2
  echo "Run 'west update' before setting up the sample." >&2
  exit 1
fi

if git -C "$zephyr" apply --unidiff-zero --reverse --check "$zephyr_patch" 2>/dev/null; then
  echo "Zephyr SRAM1 placement patch is already applied."
elif git -C "$zephyr" apply --unidiff-zero --check "$zephyr_patch"; then
  git -C "$zephyr" apply --unidiff-zero "$zephyr_patch"
  echo "Applied Zephyr SRAM1 placement patch."
else
  echo "Zephyr SRAM1 placement patch does not apply cleanly; inspect module changes." >&2
  exit 1
fi

if [ ! -d "$core_driver/.git" ]; then
  echo "Missing $core_driver" >&2
  echo "Clone the Ethos-U core driver main branch there." >&2
  exit 1
fi

actual_core_driver=$(git -C "$core_driver" rev-parse HEAD)
actual_core_driver_branch=$(git -C "$core_driver" branch --show-current)
if [ "$actual_core_driver_branch" != "main" ]; then
  echo "Ethos-U core driver must be checked out on main, not $actual_core_driver_branch" >&2
  exit 1
fi
if ! git -C "$core_driver" merge-base --is-ancestor \
  "$required_core_driver" "$actual_core_driver"; then
  echo "Ethos-U core driver main does not contain $required_core_driver" >&2
  echo "Fetch the current main branch and try again." >&2
  exit 1
fi

echo "Ethos-U core driver main: $actual_core_driver"

if [ ! -d "$workspace/modules/cmsis-nn-src" ]; then
  echo "Missing CMSIS-NN checkout at $workspace/modules/cmsis-nn-src" >&2
  exit 1
fi

echo "Workspace dependencies are ready."
