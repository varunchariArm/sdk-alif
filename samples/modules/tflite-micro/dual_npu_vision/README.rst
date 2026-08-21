TFLite Micro dual-NPU live vision
#################################

This sample is the TFLite Micro counterpart of
``samples/modules/executorch/dual_npu_vision``. Both applications run the
same quantized networks and native Zephyr camera, ISP, display, and Ethos-U
core-driver paths:

* YOLO-Fastest face detection on Ethos-U55-256
* Visual Wake Words person classification on Ethos-U85-256
* MT9M114 camera on J16 and MW405 480x800 display
* one Zephyr worker thread per NPU, released concurrently

The difference under test is the runtime backend and model container. This
sample uses Vela-compiled ``.tflite`` files and TFLite Micro. The sibling
sample uses ExecuTorch ``.pte`` files.

Model provenance
****************

The ExecuTorch PTE files are not produced independently of TFLite. The model
flow used by the sibling sample is:

``quantized .tflite -> Vela command stream -> ExecuTorch .pte``

Therefore TFLite is an offline source artifact for PTE generation. At run
time, however, the ExecuTorch firmware neither links TFLite Micro nor reads a
``.tflite`` file. This sample deliberately uses the same Vela-compiled
networks so timing and memory comparisons isolate the runtime backend as much
as possible.

Create a clean workspace
************************

.. code-block:: console

   mkdir alif-dual-npu-tflm
   cd alif-dual-npu-tflm
   python3 -m venv .venv
   . .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install west pyelftools fdt ninja

   west init -m https://github.com/varunchariArm/sdk-alif.git \
     --mr dual-npu-main-integration
   west config manifest.project-filter +tflite-micro
   west update
   python -m pip install -r zephyr/scripts/requirements.txt

   git clone --branch main \
     https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-core-driver.git \
     modules/ethos-u-core-driver-src

The fork branch supplies this application. Camera/display support comes from
the Alif SDK main history, and the external Ethos-U checkout uses the upstream
multi-variant core driver from its main branch.

Generate the models
*******************

The validated Vela outputs are checked into ``models/``. To regenerate them
from the MLEK resource definitions, enable the ``alif-mlek`` west project and
run the helper with a Python environment suitable for MLEK (Python 3.12 was
used for the checked-in artifacts):

.. code-block:: console

   west config manifest.project-filter +tflite-micro,+alif-mlek
   west update
   PYTHON=/path/to/python3.12 \
     ./sdk-alif/samples/modules/tflite-micro/dual_npu_vision/tools/generate_tflite_models.sh

The helper selects ``H256`` for YOLO/U55 and ``Z256`` for VWW/U85. Vela embeds
the target product configuration in each TFLite custom operator. The
multi-variant core driver reads that metadata and reserves the matching NPU;
the application does not select a device by thread order.

Build
*****

Run from the west workspace root:

.. code-block:: console

   APP=$PWD/sdk-alif/samples/modules/tflite-micro/dual_npu_vision
   OD=$PWD/sdk-alif/samples/modules/tflite-micro/alif_object_detection
   MODULES=$(./.venv/bin/west list -f '{abspath}' | paste -sd';' -)
   MODULES="$MODULES;$PWD/modules/ethos-u-core-driver-src"

   ./.venv/bin/west build \
     -b alif_e8_dk/ae822fa0e5597xx0/rtss_hp \
     -d build-dual-npu-tflm \
     "$APP" --pristine -- \
     -DZEPHYR_MODULES="$MODULES" \
     -DPython3_EXECUTABLE="$PWD/.venv/bin/python" \
     -DDTC_OVERLAY_FILE="$APP/boards/dual_npu_e8.overlay;$OD/boards/alif_e8_dk_ae822fa0e5597xx0_rtss_hp.overlay;$OD/serial_camera.overlay;$OD/serial_camera_mt9m114.overlay;$OD/serial_camera_isp.overlay;$OD/serial_camera_mt9m114_isp.overlay;$APP/isp_route.overlay" \
     -DOVERLAY_CONFIG="$APP/vision.conf"

The overlay order matters: ``isp_route.overlay`` must remain last so CPI is
routed only through the ISP.

Package and flash
*****************

.. code-block:: console

   export ALIF_SE_TOOLS_DIR=/path/to/alif_se_toolkit/app-release-exec-macos
   APP=$PWD/sdk-alif/samples/modules/tflite-micro/dual_npu_vision

   cp build-dual-npu-tflm/zephyr/zephyr.bin \
      build-dual-npu-tflm/u85_model.bin \
      "$ALIF_SE_TOOLS_DIR/build/images/"
   cp "$APP/flash/dual-npu-vision.json" \
      "$ALIF_SE_TOOLS_DIR/build/config/dual-npu-tflm-vision.json"

   cd "$ALIF_SE_TOOLS_DIR"
   ./app-gen-toc -f build/config/dual-npu-tflm-vision.json
   ./app-write-mram -p

Runtime and comparison
**********************

At startup both models run once on ``man_and_baby.bmp``. Five seconds later
the application starts the live pipeline. The display shows the camera
preview, face boxes, person/face state, and a 16-frame rolling timing summary.
The serial log uses the ``dual-tflm:`` prefix.

For a backend comparison, build the sibling ExecuTorch sample with the same
SDK, overlays, core-driver revision, compiler, and optimization level. Compare:

* U55, U85, span, and overlap timings printed by each application
* ``zephyr.bin`` and ``zephyr.elf`` sizes
* the TFLite payload versus PTE payload size
* build-time and runtime dependencies

Do not interpret total frame period as NPU latency: capture, resizing, display
updates, and logging occur outside the measured ``Invoke()`` interval.
