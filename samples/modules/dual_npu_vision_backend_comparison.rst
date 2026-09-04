ExecuTorch and TFLite Micro dual-NPU comparison
################################################

This guide creates a clean Alif Ensemble E8 workspace, generates comparable
SSD-Slim and MobileNetV2 artifacts, builds both runtime variants, deploys each
image with SEToolkit, and captures comparable timing data.

The two applications use the same:

* SSD-Slim checkpoint and 120x160 grayscale camera input on Ethos-U55-256.
* torchvision MobileNetV2 checkpoint and 224x224 RGB input on Ethos-U85-256.
* MT9M114, ISP, MW405 display, Zephyr configuration, core driver, and compiler.
* startup image, ImageNet labels, application postprocessing, and live camera
  pipeline.

ExecuTorch packages the networks as PTE programs. TFLite Micro packages Vela
compiled TFLite files. Both applications dispatch one worker thread per NPU
from the high-performance Cortex-M55.

Validated configuration
***********************

Hardware:

* Alif Ensemble E8 development kit.
* MT9M114 camera connected to the bottom-side J16 selfie connector.
* MW405 480x800 display.
* SEROM 1.105.65, SERAM 1.110.0, and SEToolkit 1.10.

Software:

* SDK fork branch ``dual-npu-main-integration`` at commit
  ``8e0877cb61ee4c1d4ffde008692fd14ad7030e7d``.
* Zephyr ``97fddffd316f63f9325545ec5c3dfc9a5034d831`` from the SDK manifest.
* ExecuTorch ``45fe55c49c70a5fc741833eeb99fb023ffb4646f`` from the west manifest.
* Ethos-U core driver ``main`` containing the multi-variant merge
  ``b7cd193afde80afe8bbae9a26d2ca6586554f054``. The validated checkout was
  ``3469ca818c188ead5c29fa31437a3378ac2f7de7``.
* CMSIS-NN ``d933672e7ca97eec70ef43230baee7b20c2a28ae``.
* Python 3.12 and Zephyr SDK 0.17.0.

The firmware build works on macOS or Linux. The complete model-regeneration
flow was validated on a 64-bit Arm Linux host.

Install host prerequisites
**************************

Install Git, CMake, Python 3.12, and the platform tools needed to compile
Python packages. On macOS, install the Xcode Command Line Tools and the
packages with Homebrew:

.. code-block:: console

   xcode-select --install
   brew install git cmake python@3.12

On a Debian or Ubuntu Linux host, install the corresponding packages:

.. code-block:: console

   sudo apt update
   sudo apt install -y git cmake ninja-build python3.12 python3.12-venv \
     python3.12-dev build-essential

Create an empty west workspace and its lightweight build environment:

.. code-block:: console

   mkdir -p $HOME/alif-dual-npu-compare
   cd $HOME/alif-dual-npu-compare
   python3.12 -m venv .venv
   . .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install west pyelftools fdt ninja

Clone and initialize the SDK workspace
**************************************

Clone the branch containing both comparison applications and pin the
validated revision:

.. code-block:: console

   cd $HOME/alif-dual-npu-compare
   git clone --branch dual-npu-main-integration --single-branch \
     https://github.com/varunchariArm/sdk-alif.git sdk-alif
   git -C sdk-alif checkout 8e0877cb61ee4c1d4ffde008692fd14ad7030e7d
   west init -l sdk-alif
   west config manifest.project-filter +executorch,+tflite-micro,+alif-mlek
   west update
   python -m pip install -r zephyr/scripts/requirements.txt
   west sdk install

Initialize the ExecuTorch submodules:

.. code-block:: console

   git -C modules/lib/executorch submodule update --init --recursive

Add the upstream multi-variant Ethos-U core driver. The ancestor test accepts
a newer ``main`` commit but rejects a checkout that predates multi-variant
support:

.. code-block:: console

   git clone --branch main \
     https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-core-driver.git \
     modules/ethos-u-core-driver-src
   git -C modules/ethos-u-core-driver-src merge-base --is-ancestor \
     b7cd193afde80afe8bbae9a26d2ca6586554f054 HEAD

Clone the CMSIS-NN revision used by the ExecuTorch build:

.. code-block:: console

   git clone https://github.com/ARM-software/CMSIS-NN.git \
     modules/cmsis-nn-src
   git -C modules/cmsis-nn-src checkout \
     d933672e7ca97eec70ef43230baee7b20c2a28ae

Create the ExecuTorch Python environment. Explicitly select its interpreter
when running the installer so it does not use the active west environment:

.. code-block:: console

   python3.12 -m venv .venv-executorch
   .venv-executorch/bin/python -m pip install --upgrade pip
   .venv-executorch/bin/python -m pip install \
     -r modules/lib/executorch/requirements-examples.txt
   cd modules/lib/executorch
   env -u DEBUG \
     PYTHON_EXECUTABLE=../../../.venv-executorch/bin/python \
     ./install_executorch.sh
   cd ../../..

Apply the sample's ExecuTorch integration patch and Zephyr SRAM1 placement
patch, then verify the external dependencies:

.. code-block:: console

   ./sdk-alif/samples/modules/executorch/dual_npu_vision/setup_workspace.sh

The expected output includes:

.. code-block:: output

   Applied ExecuTorch dual-NPU patch.
   Applied Zephyr SRAM1 placement patch.
   Ethos-U core driver main: ...
   Workspace dependencies are ready.

The script is idempotent. Re-run it after ``west update`` because west can
restore the ExecuTorch or Zephyr checkout.

Generate the common source checkpoints
**************************************

Model regeneration is optional because validated PTE and TFLite files are
checked into both samples. Complete this and the next two sections when you
need to reproduce or modify the deployed models.

Install the validated ExecuTorch export dependencies. This supplies
TensorFlow 2.20.0, Vela 5.0.0, TOSA Tools 2026.2.1, and FlatBuffers 24.3.25:

.. code-block:: console

   ./sdk-alif/samples/modules/executorch/dual_npu_vision/tools/install_model_export_deps.sh

Vela 5.0.0 and TOSA Tools 2026.2.1 declare conflicting FlatBuffers versions.
The helper installs the combination validated for these exporters. A later
``pip check`` can still report the TOSA Tools metadata conflict even though
TOSA serialization and both Vela compilations work.

Download the public trained SSD-Slim source and official torchvision
MobileNetV2 checkpoint:

.. code-block:: console

   mkdir -p model-weights
   curl -L --fail \
     https://raw.githubusercontent.com/emza-vs/ModelZoo/59fcdb2aab865a8a8d93a9d419b3c5490a5508e4/Models/Object_detection/SSD/ssd_slim_120x160x1_v1_int8.tflite \
     -o model-weights/ssd_slim_120x160x1_v1_int8.tflite
   curl -L --fail \
     https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth \
     -o model-weights/mobilenet_v2-7ebf99e0.pth

Verify the source files before importing them:

.. code-block:: console

   printf '%s  %s\n' \
     64fcc31aa517798d0e798551418c85bc0a5ed03a75c45c4e47fc7ee41e5ea51f \
     model-weights/ssd_slim_120x160x1_v1_int8.tflite | sha256sum -c -
   printf '%s  %s\n' \
     7ebf99e03e254b273379b23edca7ec0da9f48273b23a332b93c1c99d49e86e8f \
     model-weights/mobilenet_v2-7ebf99e0.pth | sha256sum -c -

On macOS, replace ``sha256sum -c -`` with ``shasum -a 256 -c -``.

Import the trained SSD-Slim constants into the PyTorch topology shared by
both backends:

.. code-block:: console

   APP_ET=$PWD/sdk-alif/samples/modules/executorch/dual_npu_vision
   .venv-executorch/bin/python \
     "$APP_ET/tools/import_ssd_slim_tflite.py" \
     --source model-weights/ssd_slim_120x160x1_v1_int8.tflite \
     --output model-weights/ssd_slim_common.pth

With TensorFlow 2.20.0 the importer accepts per-channel scales accompanied by
a scalar zero point. It prints the checkpoint size and the maximum box and
score differences against the source TFLite model.

Generate the ExecuTorch PTE models
**********************************

Generate the U55 and U85 PTE programs from the common checkpoints:

.. code-block:: console

   APP_ET=$PWD/sdk-alif/samples/modules/executorch/dual_npu_vision
   "$APP_ET/tools/generate_pte_models.sh" \
     "$PWD/modules/alif-mlek/scripts/vela/ensemble_vela.ini" \
     "$PWD/model-weights/ssd_slim_common.pth" \
     "$PWD/model-weights/mobilenet_v2-7ebf99e0.pth"

The exporter performs PT2E calibration and quantization, partitions the graph
for Ethos-U, and invokes Vela. It rejects CPU fallback operators and applies
ExecuTorch ``QuantizeInputs`` and ``QuantizeOutputs`` after lowering. The
runtime ``forward`` boundaries are therefore int8 instead of the default
ExecuTorch FP32 boundaries.

Successful output reports one fully delegated graph for each target:

.. code-block:: output

   model=comparable-ssd target=ethos-u55-256 fully_delegated=yes
   runtime_boundary=input:int8 outputs:int8,int8 ...
   PTE: .../comparable_ssd_slim_u55.pte (301408 bytes)
   model=mobilenet-v2 target=ethos-u85-256 fully_delegated=yes
   runtime_boundary=input:int8 outputs:int8 ...
   PTE: .../mobilenet_v2_imagenet_u85.pte (3473776 bytes)

PTE byte hashes can differ across host environments even when model size,
delegation, tensor contracts, and Vela output agree. Do not use the PTE hash
as the only regeneration check.

Generate the TFLite Micro models
********************************

Use a separate environment for AI Edge Torch and TensorFlow Lite conversion.
This avoids changing the packages in the validated ExecuTorch environment:

.. code-block:: console

   python3.12 -m venv .venv-tflm-models
   .venv-tflm-models/bin/python -m pip install --upgrade pip
   .venv-tflm-models/bin/python -m pip install \
     numpy==2.1.3 tensorflow==2.19.1 torch==2.6.0 torchvision==0.21.0 \
     ai-edge-torch==0.4.0

Export SSD-Slim to full-int8 TFLite. Export MobileNetV2 through the equivalent
native-NHWC Keras topology; that exporter checks its output against the
torchvision model before conversion:

.. code-block:: console

   APP_ET=$PWD/sdk-alif/samples/modules/executorch/dual_npu_vision
   APP_TFLM=$PWD/sdk-alif/samples/modules/tflite-micro/dual_npu_vision
   MODEL_PY=$PWD/.venv-tflm-models/bin/python
   "$MODEL_PY" "$APP_ET/tools/export_comparable_tflite.py" ssd-slim \
     --weights model-weights/ssd_slim_common.pth \
     --output model-weights/comparable_ssd_slim.tflite
   "$MODEL_PY" "$APP_ET/tools/export_mobilenet_v2_keras_tflite.py" \
     --weights model-weights/mobilenet_v2-7ebf99e0.pth \
     --output model-weights/mobilenet_v2_imagenet.tflite \
     --labels-output "$APP_TFLM/models/labels_imagenet_1000.txt"

Compile both graphs with the same Vela 5.0.0 installation and memory-system
configuration used for PTE export:

.. code-block:: console

   VELA=$PWD/.venv-executorch/bin/vela
   VELA_INI=$PWD/modules/alif-mlek/scripts/vela/ensemble_vela.ini
   mkdir -p model-weights/u55 model-weights/u85
   "$VELA" model-weights/comparable_ssd_slim.tflite \
     --output-dir model-weights/u55 --accelerator-config ethos-u55-256 \
     --optimise Performance --config "$VELA_INI" \
     --system-config RTSS_HP_SRAM_MRAM --memory-mode Shared_Sram
   "$VELA" model-weights/mobilenet_v2_imagenet.tflite \
     --output-dir model-weights/u85 --accelerator-config ethos-u85-256 \
     --optimise Performance --config "$VELA_INI" \
     --system-config Ethos_U85_SRAM_MRAM --memory-mode Shared_Sram

Copy the compiled files into the TFLite Micro sample:

.. code-block:: console

   cp model-weights/u55/comparable_ssd_slim_vela.tflite \
     "$APP_TFLM/models/comparable_ssd_slim_u55_256.tflite"
   cp model-weights/u85/mobilenet_v2_imagenet_vela.tflite \
     "$APP_TFLM/models/mobilenet_v2_imagenet_u85_256.tflite"

The validated output sizes are 296,816 bytes for SSD-Slim and 3,489,616 bytes
for MobileNetV2.

Build the ExecuTorch application
********************************

Define the camera overlays and module list from the workspace root:

.. code-block:: console

   APP_ET=$PWD/sdk-alif/samples/modules/executorch/dual_npu_vision
   OD=$PWD/sdk-alif/samples/modules/tflite-micro/alif_object_detection
   MODULES=$(./.venv/bin/west list -f '{abspath}' | \
     grep -v '/modules/lib/executorch$' | paste -sd';' -)
   MODULES="$MODULES;$PWD/modules/lib/executorch;$PWD/modules/ethos-u-core-driver-src"

Build the PTE application:

.. code-block:: console

   ./.venv/bin/west build \
     -b alif_e8_dk/ae822fa0e5597xx0/rtss_hp \
     -d build-dual-npu-et "$APP_ET" --pristine -- \
     -DZEPHYR_MODULES="$MODULES" \
     -DPYTHON_EXECUTABLE="$PWD/.venv-executorch/bin/python" \
     -DPython3_EXECUTABLE="$PWD/.venv-executorch/bin/python" \
     -DCMSIS_NN_LOCAL_PATH="$PWD/modules/cmsis-nn-src" \
     -DDTC_OVERLAY_FILE="$APP_ET/boards/dual_npu_e8.overlay;$OD/boards/alif_e8_dk_ae822fa0e5597xx0_rtss_hp.overlay;$OD/serial_camera.overlay;$OD/serial_camera_mt9m114.overlay;$OD/serial_camera_isp.overlay;$OD/serial_camera_mt9m114_isp.overlay;$APP_ET/isp_route.overlay" \
     -DOVERLAY_CONFIG="$APP_ET/vision.conf"

Verify the deployable files:

.. code-block:: console

   ls -lh build-dual-npu-et/zephyr/zephyr.bin \
     build-dual-npu-et/zephyr/zephyr.elf build-dual-npu-et/model_assets.bin

The validated firmware is 203,352 bytes and the combined PTE, image, and label
payload is 3,896,318 bytes.

Build the TFLite Micro application
**********************************

Create a module list that omits ExecuTorch but retains the external driver:

.. code-block:: console

   APP_TFLM=$PWD/sdk-alif/samples/modules/tflite-micro/dual_npu_vision
   MODULES_TFLM=$(./.venv/bin/west list -f '{abspath}' | \
     grep -v '/modules/lib/executorch$' | paste -sd';' -)
   MODULES_TFLM="$MODULES_TFLM;$PWD/modules/ethos-u-core-driver-src"

Build with the same overlays and configuration:

.. code-block:: console

   ./.venv/bin/west build \
     -b alif_e8_dk/ae822fa0e5597xx0/rtss_hp \
     -d build-dual-npu-tflm "$APP_TFLM" --pristine -- \
     -DZEPHYR_MODULES="$MODULES_TFLM" \
     -DPython3_EXECUTABLE="$PWD/.venv/bin/python" \
     -DDTC_OVERLAY_FILE="$APP_TFLM/boards/dual_npu_e8.overlay;$OD/boards/alif_e8_dk_ae822fa0e5597xx0_rtss_hp.overlay;$OD/serial_camera.overlay;$OD/serial_camera_mt9m114.overlay;$OD/serial_camera_isp.overlay;$OD/serial_camera_mt9m114_isp.overlay;$APP_TFLM/isp_route.overlay" \
     -DOVERLAY_CONFIG="$APP_TFLM/vision.conf"

Verify the deployable files:

.. code-block:: console

   ls -lh build-dual-npu-tflm/zephyr/zephyr.bin \
     build-dual-npu-tflm/zephyr/zephyr.elf build-dual-npu-tflm/u85_model.bin

The validated firmware is 162,540 bytes and the combined TFLite, image, and
label payload is 3,907,566 bytes. Exact firmware sizes can vary slightly with
the host toolchain.

Package and deploy ExecuTorch
*****************************

Set the SEToolkit 1.10 path and copy the ExecuTorch build products:

.. code-block:: console

   export ALIF_SE_TOOLS_DIR=/path/to/alif_se_toolkit_110/app-release-exec-macos
   cp build-dual-npu-et/zephyr/zephyr.bin \
     build-dual-npu-et/model_assets.bin "$ALIF_SE_TOOLS_DIR/build/images/"
   cp "$APP_ET/flash/dual-npu-vision.json" \
     "$ALIF_SE_TOOLS_DIR/build/config/dual-npu-et.json"

Move the boot switch to **SE**, close serial terminals connected to the SE
UART, and program the package:

.. code-block:: console

   cd "$ALIF_SE_TOOLS_DIR"
   ./app-gen-toc -f build/config/dual-npu-et.json
   ./app-write-mram -p

Move the switch to **U4**, open the U4 serial port at 115200 baud, and reset
the board.

Package and deploy TFLite Micro
*******************************

Return to the workspace root and copy the TFLite Micro products:

.. code-block:: console

   cd $HOME/alif-dual-npu-compare
   cp build-dual-npu-tflm/zephyr/zephyr.bin \
     build-dual-npu-tflm/u85_model.bin "$ALIF_SE_TOOLS_DIR/build/images/"
   cp "$APP_TFLM/flash/dual-npu-vision.json" \
     "$ALIF_SE_TOOLS_DIR/build/config/dual-npu-tflm.json"

Move the switch to **SE** and deploy the second package:

.. code-block:: console

   cd "$ALIF_SE_TOOLS_DIR"
   ./app-gen-toc -f build/config/dual-npu-tflm.json
   ./app-write-mram -p

Move the switch to **U4**, reconnect the U4 terminal, and reset the board.

Collect comparable results
**************************

Each application first runs both models on ``grace_hopper.bmp``, waits five
seconds, and then starts live camera inference. Confirm that the startup image
produces a face box and an ImageNet classification before recording results.

Use the same scene, camera position, lighting, warm-up period, and frame count
for both runs. Record these values from the ``dual-et:`` and ``dual-tflm:``
lines:

.. list-table:: Comparison metrics
   :header-rows: 1

   * - Metric
     - Meaning
   * - ``U55`` and ``U85``
     - Wall-clock microseconds around ExecuTorch ``method->execute()`` or
       TFLite Micro ``Invoke()``. These include runtime and delegate overhead.
   * - ``NPU-active U55/U85``
     - Ethos-U PMU cycles counted from ``NPU_ACTIVE`` to ``NPU_IDLE``.
   * - ``span``
     - Time from the first worker submission until both workers complete.
   * - ``overlap``
     - The interval during which the two invocation windows overlap.
   * - IRQ counters
     - Confirmation that both NPU instances continue completing work.

Record the artifact sizes as well:

.. code-block:: console

   stat -f '%N %z' build-dual-npu-et/zephyr/zephyr.bin \
     build-dual-npu-et/model_assets.bin \
     build-dual-npu-tflm/zephyr/zephyr.bin \
     build-dual-npu-tflm/u85_model.bin

On Linux, use ``stat -c '%n %s'`` instead.

PMU cycles are the primary comparison of NPU work. Wall-clock values show
total backend invocation cost. Camera capture, resize, tensor population,
display drawing, and serial logging occur outside the invocation windows.

The graphs have the same trained weights and input shapes, but one operator
boundary differs: TFLite MobileNetV2 keeps Softmax inside its delegated graph,
while the ExecuTorch application computes classification Softmax after
``method->execute()``. Report this distinction with benchmark results.

Troubleshooting
***************

If Kconfig reports undefined ``VIDEO_BUFFER_POOL_SRAM1`` or
``CDC200_FB_USES_SRAM1``, re-run ``setup_workspace.sh`` and rebuild with
``--pristine``. Do not remove the settings from ``vision.conf``; doing so
places the camera pool and framebuffer in SRAM0 and causes a linker overflow.

If SSD import reports ``cannot reshape array of size 1``, use commit
``8e0877c`` or newer and this importer:

.. code-block:: console

   sdk-alif/samples/modules/executorch/dual_npu_vision/tools/import_ssd_slim_tflite.py

If the firmware reports an invalid MRAM payload, deploy ``zephyr.bin`` and its
matching ``model_assets.bin`` or ``u85_model.bin`` from the same build.

If the camera reports chip ID ``0000`` or I2C error ``-5``, power off the
board, reseat the MT9M114 cable on J16, and verify the contact orientation.
The supplied overlays do not target J22.
