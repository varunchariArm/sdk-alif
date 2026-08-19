Dual-NPU live vision
####################

This RTSS-HP sample runs YOLO-Fastest face detection on Ethos-U55 and Visual
Wake Words on Ethos-U85. It uses the native Zephyr MT9M114, ISP and MW405
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
* ``build-dual-npu-vision/u85_model.bin`` (both PTEs and startup BMP)

Package and flash
*****************

Set ``ALIF_SE_TOOLS_DIR`` to the SEToolkit 1.10 application directory. Copy
the generated application images and the sample's package configuration into
the toolkit:

.. code-block:: console

   export ALIF_SE_TOOLS_DIR=/path/to/alif_se_toolkit/app-release-exec-macos
   APP=$PWD/sdk-alif/samples/modules/executorch/dual_npu_vision

   cp build-dual-npu-vision/zephyr/zephyr.bin \
      build-dual-npu-vision/u85_model.bin \
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

At startup the application runs both models once using the bundled
``man_and_baby.bmp`` test image. After five seconds it switches to the live
MT9M114 stream. The ISP produces RGB888 frames, which are resized separately
for YOLO (192x192 grayscale) and VWW (128x128 RGB). Dedicated Zephyr worker
threads submit U55 and U85 work in parallel.

The MW405 UI shows a 352x352 live preview, YOLO face boxes, person/face status,
and rolling U55/U85/span timing summaries. Video buffers are returned to the
ISP queue after processing so capture can run continuously.

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

The checked-in PTE files are generated in two stages. First, Vela compiles the
quantized TFLite network for the target NPU and emits an ``*_vela.npz`` file.
Second, the scripts in ``tools/`` serialize that command stream and its tensor
contract as an ExecuTorch program.

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

Generate Vela artifacts from the original fully-int8 TFLite models. Use the
``ensemble_vela.ini`` supplied by Alif MLEK and preserve these target settings:

.. code-block:: console

   vela yolo-fastest_192_face_v4.tflite \
     --accelerator-config ethos-u55-256 \
     --optimise Performance \
     --config /path/to/alif-mlek/scripts/vela/ensemble_vela.ini \
     --system-config RTSS_HP_SRAM_MRAM \
     --memory-mode Shared_Sram \
     --output-dir model-artifacts/yolo-u55

   vela vww4_128_128_INT8.tflite \
     --accelerator-config ethos-u85-256 \
     --optimise Performance \
     --config /path/to/alif-mlek/scripts/vela/ensemble_vela.ini \
     --system-config Ethos_U85_SRAM_Only \
     --memory-mode Sram_Only \
     --output-dir model-artifacts/vww-u85

Create both PTE files from Vela's NPZ files:

.. code-block:: console

   APP=$PWD/sdk-alif/samples/modules/executorch/dual_npu_vision
   "$APP/tools/generate_pte_models.sh" \
     model-artifacts/yolo-u55/yolo-fastest_192_face_v4_vela.npz \
     model-artifacts/vww-u85/vww4_128_128_INT8_vela.npz

Expected outputs are:

* ``models/yolo_fastest_face_u55_256.pte``: 370,080 bytes, input
  ``1x1x192x192`` int8 and two YOLO output tensors.
* ``models/vww_u85_256.pte``: 478,560 bytes, input ``1x1x128x128`` int8 and
  one two-class output tensor.

The exporters deliberately accept Vela NPZ files instead of silently
downloading training models. This keeps the original model license and model
selection explicit. Rebuild the Zephyr application after replacing either PTE
so ``u85_model.bin`` is repacked.
