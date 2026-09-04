TFLite Micro dual-NPU live vision
#################################

This sample demonstrates two different vision workloads running concurrently
through TFLite Micro while sharing the native Zephyr camera, ISP, display, and
multi-variant Ethos-U core-driver paths:

* SSD-Slim face detection (120x160 grayscale) on Ethos-U55-256
* MobileNetV2 ImageNet classification (224x224 RGB) on Ethos-U85-256
* MT9M114 camera on J16 and MW405 480x800 display
* one Zephyr worker thread per NPU, released concurrently

The U55 receives the smaller detector, while the higher-throughput U85 runs
the substantially larger classifier. This pairing demonstrates heterogeneous
workload placement rather than assigning both NPUs copies of one task.

For a greenfield, side-by-side TFLite Micro and ExecuTorch build, model
generation, deployment, and measurement procedure, see
``samples/modules/dual_npu_vision_backend_comparison.rst``.

Model provenance
****************

Both artifacts are generated from the same PyTorch checkpoints used by the
ExecuTorch comparison build. SSD-Slim imports the trained weights from the
emza-vs model and exposes raw box deltas and two-class logits. MobileNetV2 uses
the official torchvision ImageNet checkpoint and exposes 1000 raw logits.
TensorFlow Lite conversion performs full integer quantization, and Vela then
compiles the models for U55-256 and U85-256 respectively. Detector softmax,
anchor decoding, and NMS remain in common application postprocessing. The
TFLite MobileNetV2 carries its output Softmax in the delegated graph, while
the ExecuTorch application applies Softmax after ``method->execute()``; use
NPU PMU counters when isolating that operator from backend overhead.

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

The validated Vela outputs are checked into ``models/``. To regenerate them,
enable the ``alif-mlek`` west project and use Python 3.12 with PyTorch,
torchvision, TensorFlow, and AI Edge Torch installed:

.. code-block:: console

   west config manifest.project-filter +tflite-micro,+alif-mlek
   west update
   PYTHON=/path/to/model-venv/bin/python VELA=/path/to/vela \
     ./sdk-alif/samples/modules/tflite-micro/dual_npu_vision/tools/generate_tflite_models.sh

The helper downloads the public trained SSD-Slim artifact and official
torchvision MobileNetV2 checkpoint. It imports SSD-Slim into the shared
PyTorch topology, transfers MobileNetV2 into an equivalent native-NHWC Keras
graph, performs full-int8 conversion, and compiles the models for
``ethos-u55-256`` and ``ethos-u85-256``. Vela embeds the target product
configuration in each TFLite custom operator. The multi-variant core driver
reads that metadata and reserves the matching NPU.

The production artifacts are:

* ``comparable_ssd_slim_u55_256.tflite``: 296,816 bytes
* ``mobilenet_v2_imagenet_u85_256.tflite``: 3,489,616 bytes
* ``labels_imagenet_1000.txt`` and ``grace_hopper.bmp``

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

At startup both models run once on ``grace_hopper.bmp``, a square crop of
TensorFlow's standard Grace Hopper classification example. It provides one
large frontal face for SSD and a naval uniform that MobileNetV2 can classify
reliably. The source photograph is a public-domain U.S. Navy image; the test
asset is distributed by TensorFlow under its Apache-2.0 repository.

Five seconds later the application starts the live pipeline. The display
shows the camera preview, SSD face boxes and count, the top MobileNetV2 class
and confidence, and a 16-frame rolling timing summary. The serial log uses the
``dual-tflm:`` prefix.

For a backend comparison, use equivalent networks and preprocessing with the
same SDK, overlays, core-driver revision, compiler, and optimization level.
Compare:

* U55, U85, span, and overlap timings printed by each application
* ``NPU-active U55/U85`` PMU cycles printed by each application
* ``zephyr.bin`` and ``zephyr.elf`` sizes
* the TFLite payload versus PTE payload size
* build-time and runtime dependencies

Do not interpret total frame period as NPU latency: capture, resizing, display
updates, and logging occur outside the measured ``Invoke()`` interval. The
``U55`` and ``U85`` values measure wall time around ``Invoke()`` and include
TFLite Micro delegate/runtime overhead. The ``NPU-active`` values come from
the Ethos-U PMU counter in the same core-driver inference hooks used by the
ExecuTorch build, so they isolate accelerator-active work from host backend
overhead.
