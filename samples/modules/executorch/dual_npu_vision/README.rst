Dual-NPU live vision
####################

This RTSS-HP sample runs a trained SSD-Slim face detector on Ethos-U55 and a
trained torchvision MobileNetV2 image classifier on
Ethos-U85. Both PyTorch models are quantized and lowered directly through the
ExecuTorch Ethos-U backend; no TFLite model or Vela NPZ wrapper is used. It
uses the native Zephyr MT9M114, ISP and MW405
drivers on the Alif SDK ``main`` branch. The MT9M114 and MW405 support landed
through Alif PR #879; no MLEK camera or display bridge is linked.

Hardware
********

* Alif Ensemble E8 development kit
* MT9M114 camera on J16
* MW405 480x800 display
* SERAM 1.110.0 and SEToolkit 1.10

Create a clean workspace
************************

Use a new, empty directory so that west modules, generated files, and CMake
caches from an older SDK or core-driver branch cannot affect the build:

.. code-block:: console

   mkdir alif-dual-npu-main
   cd alif-dual-npu-main
   python3 -m venv .venv
   . .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install west pyelftools fdt ninja

   west init -m https://github.com/varunchariArm/sdk-alif.git --mr main
   west config manifest.project-filter +executorch
   west update
   python -m pip install -r zephyr/scripts/requirements.txt
   git -C modules/lib/executorch submodule update --init --recursive

   git clone --branch main \
     https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-core-driver.git \
     modules/ethos-u-core-driver-src
   git -C modules/ethos-u-core-driver-src merge-base --is-ancestor \
     b7cd193afde80afe8bbae9a26d2ca6586554f054 HEAD

   git clone https://github.com/ARM-software/CMSIS-NN.git \
     modules/cmsis-nn-src
   git -C modules/cmsis-nn-src checkout \
     d933672e7ca97eec70ef43230baee7b20c2a28ae

Create ``.venv-executorch`` with Python 3.12 and install the ExecuTorch Python
requirements as described under `Generating the PTE models`_. Do not copy a
previous build directory into this workspace.

Build
*****

Run from the west workspace root (the ``clone`` directory):

First apply the small ExecuTorch integration patch. It enables the Alif
``CONFIG_ARM_ETHOS_U`` Kconfig name, permits schema-declared delegated tensor
inputs, and provides the per-NPU command/weight preparation hooks used by this
sample.

.. code-block:: console

   ./sdk-alif/samples/modules/executorch/dual_npu_vision/setup_workspace.sh

The setup script is idempotent. It also verifies that the Ethos-U core driver
checkout is on ``main`` and contains the multi-variant merge commit.

.. code-block:: console

   APP=$PWD/sdk-alif/samples/modules/executorch/dual_npu_vision
   OD=$PWD/sdk-alif/samples/modules/tflite-micro/alif_object_detection
   MODULES=$(./.venv/bin/west list -f '{abspath}' | grep -v '/modules/lib/executorch$' | paste -sd';' -)
   MODULES="$MODULES;$PWD/modules/lib/executorch;$PWD/modules/ethos-u-core-driver-src"

   ./.venv/bin/west build \
     -b alif_e8_dk/ae822fa0e5597xx0/rtss_hp \
     -d build-dual-npu-vision \
     "$APP" --pristine -- \
     -DZEPHYR_MODULES="$MODULES" \
     -DPYTHON_EXECUTABLE="$PWD/.venv-executorch/bin/python" \
     -DPython3_EXECUTABLE="$PWD/.venv-executorch/bin/python" \
     -DCMSIS_NN_LOCAL_PATH="$PWD/modules/cmsis-nn-src" \
     -DDTC_OVERLAY_FILE="$APP/boards/dual_npu_e8.overlay;$OD/boards/alif_e8_dk_ae822fa0e5597xx0_rtss_hp.overlay;$OD/serial_camera.overlay;$OD/serial_camera_mt9m114.overlay;$OD/serial_camera_isp.overlay;$OD/serial_camera_mt9m114_isp.overlay;$APP/isp_route.overlay" \
     -DOVERLAY_CONFIG="$APP/vision.conf"

The final overlay must remain last so CPI routes exclusively to the ISP.
The validated path selects 1288x728 Y10P input, crops to the PR-defined square
ROI, and requests 192x192 planar RGB888 output from the ISP. Five video
buffers are circulated continuously. The preview is scaled to 352x352 at
display position (64,184).

Outputs
*******

* ``build-dual-npu-vision/zephyr/zephyr.bin``
* ``build-dual-npu-vision/zephyr/zephyr.elf``
* ``build-dual-npu-vision/model_assets.bin`` (both PTEs, labels, startup BMP)

Package and flash
*****************

Set ``ALIF_SE_TOOLS_DIR`` to the SEToolkit 1.10 application directory. Copy
the generated application images and the sample's package configuration into
the toolkit:

.. code-block:: console

   export ALIF_SE_TOOLS_DIR=/path/to/alif_se_toolkit/app-release-exec-macos
   APP=$PWD/sdk-alif/samples/modules/executorch/dual_npu_vision

   cp build-dual-npu-vision/zephyr/zephyr.bin \
      build-dual-npu-vision/model_assets.bin \
      "$ALIF_SE_TOOLS_DIR/build/images/"
   cp "$APP/flash/dual-npu-vision.json" \
      "$ALIF_SE_TOOLS_DIR/build/config/"

Put the E8 DevKit in SE mode and close any terminal connected to the SE UART.
Generate the application table of contents and write the package to MRAM:

.. code-block:: console

   cd "$ALIF_SE_TOOLS_DIR"
   ./app-gen-toc -f build/config/dual-npu-vision.json
   ./app-write-mram -p

After the write completes, move the board switch to U4, open the U4 serial
port at 115200 baud, and reset the board.

Runtime behavior
****************

At startup the application runs both models once using the bundled Grace
Hopper test image. After five seconds it switches to the live MT9M114 stream.
The ISP produces RGB888 frames, which are resized separately for SSD-Slim
(160x120 grayscale) and MobileNetV2 (224x224 RGB). Dedicated Zephyr worker threads
submit U55 and U85 work in parallel.

The MW405 UI shows a 352x352 live preview, face-detection boxes, the current
classification label and confidence, and rolling U55/U85/span timing
summaries. Video buffers are returned to the ISP queue after processing so
capture can run continuously.

Workspace dependencies
**********************

The validated workspace uses:

* ExecuTorch at the revision selected by ``west.yml`` plus
  ``patches/executorch-dual-npu.patch``.
* Ethos-U core driver ``main`` at ``modules/ethos-u-core-driver-src``. The
  checkout must contain commit
  ``b7cd193afde80afe8bbae9a26d2ca6586554f054``, which renamed and merged the
  multi-variant feature.
* CMSIS-NN at ``modules/cmsis-nn-src``.

Re-run ``setup_workspace.sh`` after ``west update`` because west may restore
the ExecuTorch module checkout.

Generating the PTE models
*************************

The checked-in PTE files are generated directly from trained torchvision
checkpoints. ``export_torchvision_models.py`` performs PT2E quantization,
calibration, ExecuTorch Ethos-U partitioning, and Vela compilation. The
exporter rejects a model unless its graph contains exactly one Ethos-U
delegate and no CPU fallback operators.

Prepare ExecuTorch's Python tools from the workspace root. The exact setup
options can vary with the pinned ExecuTorch revision; the following is the
standard setup for this workspace:

.. code-block:: console

   python3.12 -m venv .venv-executorch
   cd modules/lib/executorch
   ../../../.venv-executorch/bin/python -m pip install -r requirements-examples.txt
   env -u DEBUG ./install_executorch.sh
   cd ../../../

The ``env -u DEBUG`` prefix avoids treating an unrelated host ``DEBUG`` shell
variable as ExecuTorch's numeric build option. The firmware build does not
need the optional ``ethos_u`` Python dependency group; Vela is invoked
separately when regenerating the models.

Download the trained public SSD-Slim artifact and the official torchvision
MobileNetV2 checkpoint. Importing the SSD constants produces the common
PyTorch checkpoint consumed by both backend exporters:

.. code-block:: console

   curl -L -o ssd_slim_source.tflite \
     https://raw.githubusercontent.com/emza-vs/ModelZoo/master/Models/Object_detection/SSD/ssd_slim_120x160x1_v1_int8.tflite
   curl -L -o mobilenet_v2.pth \
     https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth

   APP=$PWD/sdk-alif/samples/modules/executorch/dual_npu_vision
   ./.venv-executorch/bin/python "$APP/tools/import_ssd_slim_tflite.py" \
     --source ssd_slim_source.tflite \
     --output comparable_ssd_slim.pth

Generate both PTE files using Alif MLEK's ``ensemble_vela.ini`` memory-system
definitions:

.. code-block:: console

   APP=$PWD/sdk-alif/samples/modules/executorch/dual_npu_vision
   "$APP/tools/generate_pte_models.sh" \
     /path/to/alif-mlek/scripts/vela/ensemble_vela.ini \
     comparable_ssd_slim.pth \
     mobilenet_v2.pth

Expected outputs are:

* ``models/comparable_ssd_slim_u55.pte``: input ``1x1x120x160`` int8 and raw
  box-delta/two-class-logit tensors targeting Ethos-U55-256.
* ``models/mobilenet_v2_imagenet_u85.pte``: input ``1x3x224x224`` int8 and a
  1000-class probability tensor targeting Ethos-U85-256.
* ``models/labels_imagenet_1000.txt``: torchvision's matching ImageNet labels.

Both PTEs expose int8 inputs and outputs and contain one Ethos-U delegate with
no CPU fallback. Rebuild after replacing either PTE; CMake watches the model
files and regenerates the compiled MRAM offsets automatically.

Runtime and artifact size
*************************

The application PTEs are completely delegated, so the CMake target does not
link ExecuTorch's Cortex-M fallback-kernel catalogue. The libraries may still
be compiled as part of the module build, but the linker discards them. This is
safe only while every production PTE remains fully delegated.

For the validated comparable models, the current pristine build measures:

.. list-table:: ExecuTorch artifacts
   :header-rows: 1

   * - Artifact
     - Size
   * - Zephyr firmware ``zephyr.bin``
     - 203,084 bytes
   * - Combined model/test payload
     - 3,888,654 bytes
   * - U55 SSD-Slim PTE
     - 296,896 bytes
   * - U85 MobileNetV2 PTE
     - 3,470,624 bytes

An ``nm`` check of the resulting ELF reports no ``cortex_m::native``
fallback-op symbols.

Both exported memory plans and their retained int8 inputs fit the statically
reserved method pools; the five-buffer Zephyr video pool remains separately
allocated in SRAM0.

The model payload starts at ``0x80008000`` and the HP firmware executes from
``0x80400000``. The package keeps the payload below the firmware and within
the validated E8 MRAM window.
