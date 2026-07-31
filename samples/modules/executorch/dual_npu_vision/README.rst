Dual-NPU live vision
####################

This RTSS-HP sample runs YOLO-Fastest face detection on Ethos-U55 and Visual
Wake Words on Ethos-U85. It uses the native Zephyr MT9M114, ISP and MW405
drivers introduced by the updated Alif PR #879. No MLEK camera or display
bridge is linked.

Hardware
********

* Alif Ensemble E8 development kit
* MT9M114 camera on J16
* MW405 480x800 display
* SERAM 1.110.0 and SEToolkit 1.10

Build
*****

Run from the west workspace root (the ``clone`` directory):

First apply the small ExecuTorch integration patch. It enables the Alif
``CONFIG_ARM_ETHOS_U`` Kconfig name, permits schema-declared delegated tensor
inputs, and provides the per-NPU command/weight preparation hooks used by this
sample.

.. code-block:: console

   ./sdk-alif/samples/modules/executorch/dual_npu_vision/setup_workspace.sh

The setup script is idempotent and also checks that the validated Ethos-U core
driver checkout is present.

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
* Ethos-U core driver commit ``fe1efeff3381032d459216b63a78f658f100ad75``
  at ``modules/ethos-u-core-driver-src``.
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

   cd modules/lib/executorch
   ../../../.venv-executorch/bin/python -m pip install -r requirements-examples.txt
   ./install_executorch.sh
   cd ../../../

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
