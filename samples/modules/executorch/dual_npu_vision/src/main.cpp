/* SPDX-License-Identifier: Apache-2.0 */

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <new>
#include <vector>

#include <executorch/examples/arm/executor_runner/arm_memory_allocator.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>
#include <zephyr/cache.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include "model_layout.h"

#if defined(CONFIG_DUAL_ET_LEGACY_NATIVE_VISION)
#include <zephyr/device.h>
#include <zephyr/drivers/display.h>
#include <zephyr/drivers/display/cdc200.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/i2c.h>
#include <zephyr/drivers/mipi_dsi/dsi_dw.h>
#include <zephyr/drivers/video.h>
#include <zephyr/init.h>
#include <soc_common.h>
#include <se_service.h>
#endif

using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::aten::ScalarType;
using executorch::extension::BufferDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::Tag;

extern "C" Error executorch_delegate_EthosUBackend_registered(void);
extern "C" int dual_ethosu_init(void);
extern "C" unsigned dual_ethosu_u55_irqs(void);
extern "C" unsigned dual_ethosu_u85_irqs(void);
extern "C" uint64_t dual_ethosu_u55_submit_cycle(void);
extern "C" uint64_t dual_ethosu_u85_submit_cycle(void);
extern "C" uint64_t dual_ethosu_u55_irq_cycle(void);
extern "C" uint64_t dual_ethosu_u85_irq_cycle(void);
extern "C" void dual_report_checkpoint(unsigned) {}

extern "C" {
size_t ethosu_fast_scratch_size = 1536;
alignas(16) unsigned char ethosu_dtcm_scratch[1536];
unsigned char* ethosu_fast_scratch = ethosu_dtcm_scratch;
}

namespace {

/* Retained inputs are allocated after the method-planned buffers. */
constexpr size_t kU55MethodPoolSize = 640 * 1024;
constexpr size_t kMethodMetadataPoolSize = 4096;
// Leave headroom for the exported memory plans, retained inputs, and
// allocator alignment.
/* MobileNetV2 needs 150,912 bytes of planned storage plus its 150,528-byte
 * retained input. */
constexpr size_t kU85MethodPoolSize = 304 * 1024;
constexpr size_t kU55TempPoolSize = 3686400;
constexpr size_t kU85TempPoolSize = 1509968;
enum class Workload { FaceDetection, ImageClassification };

#if defined(CONFIG_DUAL_ET_LIVE_VISION)
/* Match the proven TFLM package: one payload below the XIP firmware. */
constexpr uintptr_t kU85PteAddress = 0x80008000U;
#else
constexpr uintptr_t kU85PteAddress = 0x02000000U;
constexpr uintptr_t kPayloadStoreAddress = kU85PteAddress;
#endif
constexpr size_t kU85PteSize = U85_PTE_SIZE;
#if defined(CONFIG_DUAL_ET_LIVE_VISION)
constexpr uintptr_t kU55PteAddress = kU85PteAddress + kU85PteSize;
#else
constexpr uintptr_t kU55PteAddress = kU85PteAddress + kU85PteSize;
#endif
constexpr size_t kU55PteSize = U55_PTE_SIZE;
constexpr uintptr_t kFaceBmpAddress = kU55PteAddress + kU55PteSize;
constexpr size_t kFaceBmpSize = FACE_BMP_SIZE;
constexpr uintptr_t kClassLabelsAddress = kFaceBmpAddress + kFaceBmpSize;
constexpr size_t kClassLabelsSize = CLASS_LABELS_SIZE;
#if defined(CONFIG_DUAL_ET_LIVE_VISION)
constexpr size_t kModelPayloadSize = kU85PteSize + kU55PteSize +
                                     kFaceBmpSize + kClassLabelsSize;
constexpr size_t kLiveWidth = 480;
constexpr size_t kLiveHeight = 352;
constexpr size_t kLiveFrameSize = kLiveWidth * kLiveHeight * 2;
constexpr size_t kCameraWidth = 560;
constexpr size_t kCameraHeight = 560;
constexpr size_t kCameraFrameSize = kCameraWidth * kCameraHeight * 2;
/* SRAM1 layout: 1 MiB U85 temp, 2 MiB U85 method, then camera frame. */
#else
constexpr uint32_t kFaceBmpHeader = 0xb0364d42U;
constexpr size_t kPayloadSize = kU85PteSize + kU55PteSize + kFaceBmpSize;
#endif
const unsigned char* const model_u55_pte =
    reinterpret_cast<const unsigned char*>(kU55PteAddress);

alignas(16)
unsigned char u55_method_pool[kU55MethodPoolSize];
alignas(16) unsigned char u55_metadata_pool[kMethodMetadataPoolSize];
alignas(16) unsigned char u85_metadata_pool[kMethodMetadataPoolSize];
__attribute__((section(".alif_sram1.tensor_arena"), aligned(16)))
unsigned char u85_method_pool[kU85MethodPoolSize];

/* Keep each delegate scratch arena in memory visible to its NPU. */
__attribute__((section(".alif_sram0.npu_scratch"), aligned(32)))
unsigned char u55_temp_pool[kU55TempPoolSize];
__attribute__((section(".alif_sram1.tensor_arena"), aligned(16)))
unsigned char u85_temp_pool[kU85TempPoolSize];

#if defined(CONFIG_DUAL_ET_LIVE_VISION)
/* ExecuTorch parses MobileNetV2 directly from U85-visible MRAM. */
const unsigned char* const model_u85_pte =
    reinterpret_cast<const unsigned char*>(kU85PteAddress);
#else
const unsigned char* const model_u85_pte =
    reinterpret_cast<const unsigned char*>(kU85PteAddress);
#endif

#if defined(CONFIG_DUAL_ET_LEGACY_NATIVE_VISION)
uint16_t* live_frame;
__attribute__((section(".alif_sram1.camera_frame"), aligned(32)))
uint16_t camera_frame_storage[kCameraWidth * kCameraHeight];
uint16_t* camera_frame = camera_frame_storage;
struct video_buffer live_video_buffer;
const struct device* live_display;
volatile uint32_t run_memory_before;
volatile uint32_t run_memory_after;
volatile int run_profile_status;
alignas(32) uint16_t display_line[kLiveWidth];

int publish_display_test_pattern() {
  struct display_buffer_descriptor desc = {
      .buf_size = sizeof(display_line), .width = kLiveWidth,
      .height = 1, .pitch = kLiveWidth};
  int rc = 0;
  for (size_t y = 0; y < kLiveHeight; ++y) {
    for (size_t x = 0; x < kLiveWidth; ++x) {
      display_line[x] = (x < kLiveWidth / 3) ? 0xf800 :
                        (x < 2 * kLiveWidth / 3) ? 0x07e0 : 0x001f;
    }
    rc = cdc200_display_write(live_display, 0, 0, y, &desc, display_line);
    if (rc != 0) break;
  }
  struct cdc200_fb_desc fb = {};
  cdc200_get_framebuffer(live_display, 0, &fb);
  const volatile uint16_t* pixels =
      reinterpret_cast<const volatile uint16_t*>(fb.fb_addr);
  printk("dual-et: display RGB test rc=%d readback=%04x/%04x/%04x\n",
         rc, pixels[0], pixels[kLiveWidth / 2], pixels[kLiveWidth - 1]);
  return rc;
}
int enable_live_display() {
  const struct device* panel = DEVICE_DT_GET(DT_ALIAS(panel));
  const struct device* dsi = DEVICE_DT_GET(DT_ALIAS(mipi_dsi));
  live_display = DEVICE_DT_GET(DT_CHOSEN(zephyr_display));
  if (!device_is_ready(panel) || !device_is_ready(dsi) ||
      !device_is_ready(live_display)) {
    printk("dual-et: display readiness panel=%d dsi=%d cdc200=%d\n",
           device_is_ready(panel), device_is_ready(dsi),
           device_is_ready(live_display));
    return -ENODEV;
  }
  int rc = dsi_dw_set_mode(dsi, DSI_DW_VIDEO_MODE);
  if (rc != 0) {
    printk("dual-et: DSI video mode failed rc=%d\n", rc);
    return rc;
  }
  rc = display_blanking_off(panel);
  if (rc != 0) {
    printk("dual-et: MW405 blanking off failed rc=%d\n", rc);
    return rc;
  }
  cdc200_set_enable(live_display, true);
  struct cdc200_display_caps caps = {};
  struct cdc200_fb_desc fb = {};
  cdc200_get_capabilities(live_display, &caps);
  cdc200_get_framebuffer(live_display, 0, &fb);
  printk("dual-et: MW405-C enabled panel=%ux%u L1=%d %ux%u fmt=%u fb=%p/%zu\n",
         caps.x_panel_resolution, caps.y_panel_resolution,
         caps.layer[0].layer_en, caps.layer[0].x_resolution,
         caps.layer[0].y_resolution,
         caps.layer[0].current_pixel_format, fb.fb_addr, fb.fb_size);
  return publish_display_test_pattern();
}

static int configure_camera_display_mux(void) {
  const struct gpio_dt_spec mux =
      GPIO_DT_SPEC_GET(DT_NODELABEL(mipi_dsi), cam_disp_mux_gpios);
  int rc = gpio_pin_configure_dt(&mux, GPIO_OUTPUT_ACTIVE);
  if (rc != 0) return rc;

  sys_set_bits(CGU_CLK_ENA, BIT(23) | BIT(21) | BIT(7));
  run_profile_t runp = {};
  int attempts = 0;
  do {
    rc = se_service_get_run_cfg(&runp);
    if (rc != 0) k_busy_wait(10000);
  } while (rc != 0 && ++attempts < 5);
  if (rc != 0) {
    run_profile_status = rc;
    return rc;
  }
  run_memory_before = runp.memory_blocks;
  /* Match Alif MLEK's E8 run profile.  MRAM must remain explicitly enabled
   * for non-CPU bus masters: the HP core can continue fetching its boot image
   * while U85's EXT AXI reads otherwise stall without ever raising an IRQ.
   */
  runp.memory_blocks |= SRAM0_MASK | SRAM1_MASK | MRAM_MASK |
                        FWRAM_MASK | BACKUP4K_MASK;
  runp.power_domains |= PD_VBAT_AON_MASK | PD_SYST_MASK |
                        PD_SSE700_AON_MASK | PD_SESS_MASK | PD_DBSS_MASK;
  runp.phy_pwr_gating |= MIPI_TX_DPHY_MASK | MIPI_RX_DPHY_MASK |
                         MIPI_PLL_DPHY_MASK | LDO_PHY_MASK;
  runp.ip_clock_gating |= NPU_HP_MASK | NPU_HE_MASK | CDC200_MASK |
                          CAMERA_MASK | MIPI_DSI_MASK | MIPI_CSI_MASK |
                          LP_PERIPH_MASK;
  attempts = 0;
  do {
    rc = se_service_set_run_cfg(&runp);
    if (rc != 0) k_busy_wait(10000);
  } while (rc != 0 && ++attempts < 5);
  run_memory_after = runp.memory_blocks;
  run_profile_status = rc;
  if (rc != 0) return rc;
  sys_write32(0x140001, CLKCTRL_PER_MST_CAMERA_PIXCLK_CTRL);
  return 0;
}
SYS_INIT(configure_camera_display_mux, PRE_KERNEL_1, 46);

int capture_live_frame() {
  const struct device* camera = DEVICE_DT_GET_ONE(alif_cam);
  if (live_display == nullptr || !device_is_ready(camera)) {
    printk("dual-et: vision device readiness display=%d camera=%d\n",
           live_display != nullptr, device_is_ready(camera));
    return -ENODEV;
  }
  struct cdc200_fb_desc fb = {};
  cdc200_get_framebuffer(live_display, 0, &fb);
  if (fb.fb_addr == nullptr || fb.fb_size < kLiveFrameSize) {
    printk("dual-et: CDC200 framebuffer failed addr=%p size=%zu need=%zu\n",
           fb.fb_addr, fb.fb_size, kLiveFrameSize);
    return -ENOMEM;
  }
  live_frame = reinterpret_cast<uint16_t*>(fb.fb_addr);
  memset(live_frame, 0, kLiveFrameSize);
  sys_cache_data_flush_range(live_frame, kLiveFrameSize);

  struct video_format format = {
      .pixelformat = VIDEO_PIX_FMT_Y10P,
      .width = kCameraWidth,
      .height = kCameraHeight,
      .pitch = kCameraWidth * 2,
  };
  int rc = video_set_format(camera, VIDEO_EP_OUT, &format);
  if (rc != 0) {
    printk("dual-et: ARX3A0 format failed rc=%d\n", rc);
    return rc;
  }
  live_video_buffer = {};
  live_video_buffer.buffer = reinterpret_cast<uint8_t*>(camera_frame);
  live_video_buffer.size = kCameraFrameSize;
  rc = video_enqueue(camera, VIDEO_EP_OUT, &live_video_buffer);
  if (rc != 0) {
    printk("dual-et: camera enqueue failed rc=%d\n", rc);
    return rc;
  }
  printk("dual-et: ARX3A0 capture buffer=%p size=%zu\n",
         camera_frame, kCameraFrameSize);
  struct video_buffer* captured = nullptr;
  /* This driver has a single-buffer queue. Warm up with independent capture
   * cycles rather than recycling a buffer while streaming. */
  for (int frame = 0; frame < 3; ++frame) {
    captured = nullptr;
    rc = video_stream_start(camera);
    if (rc != 0) break;
    rc = video_dequeue(camera, VIDEO_EP_OUT, &captured, K_SECONDS(5));
    video_stream_stop(camera);
    if (rc != 0 || captured != &live_video_buffer) break;
    sys_cache_data_invd_range(camera_frame, kCameraFrameSize);
    const volatile uint16_t* probe = camera_frame;
    printk("dual-et: camera frame %d timestamp=%u samples=%04x/%04x/%04x\n",
           frame, captured->timestamp, probe[0],
           probe[kCameraWidth * kCameraHeight / 2],
           probe[kCameraWidth * kCameraHeight - 1]);
    if (frame != 2) {
      rc = video_enqueue(camera, VIDEO_EP_OUT, &live_video_buffer);
      if (rc != 0) break;
    }
  }
  if (rc != 0 || captured != &live_video_buffer) {
    printk("dual-et: camera capture failed rc=%d buffer=%p\n", rc, captured);
    return rc != 0 ? rc : -EIO;
  }
  sys_cache_data_invd_range(camera_frame, kCameraFrameSize);
  printk("dual-et: captured ARX3A0 RAW10 frame bytes=%u timestamp=%u ms\n",
         captured->bytesused, captured->timestamp);

  constexpr size_t preview_source_height =
      kCameraWidth * kLiveHeight / kLiveWidth;
  constexpr size_t preview_y0 =
      (kCameraHeight - preview_source_height) / 2;
  uint16_t min_raw = 0x03ff;
  uint16_t max_raw = 0;
  for (size_t y = 0; y < kLiveHeight; ++y) {
    const size_t source_y = preview_y0 + y * preview_source_height / kLiveHeight;
    for (size_t x = 0; x < kLiveWidth; ++x) {
      const size_t source_x = x * kCameraWidth / kLiveWidth;
      const uint16_t raw10 = camera_frame[source_y * kCameraWidth + source_x] & 0x03ff;
      min_raw = raw10 < min_raw ? raw10 : min_raw;
      max_raw = raw10 > max_raw ? raw10 : max_raw;
      const uint8_t gray = static_cast<uint8_t>((raw10 * 255U + 511U) / 1023U);
      display_line[x] =
          static_cast<uint16_t>(((gray >> 3) << 11) |
                                ((gray >> 2) << 5) | (gray >> 3));
      if (y == 0 || y == kLiveHeight - 1 || x == 0 ||
          x == kLiveWidth - 1 || x == kLiveWidth / 2) {
        display_line[x] = 0xf800;
      }
    }
    struct display_buffer_descriptor row = {
        .buf_size = sizeof(display_line), .width = kLiveWidth,
        .height = 1, .pitch = kLiveWidth};
    rc = cdc200_display_write(live_display, 0, 0, y, &row, display_line);
    if (rc != 0) return rc;
  }
  printk("dual-et: rendered preview raw-range=%u..%u publish-rc=%d\n",
         min_raw, max_raw, rc);
  if (rc != 0) return rc;
  return 0;
}

inline void live_rgb(size_t x, size_t y, uint8_t& red, uint8_t& green,
                     uint8_t& blue) {
  constexpr size_t source_height = kCameraWidth * kLiveHeight / kLiveWidth;
  constexpr size_t source_y0 = (kCameraHeight - source_height) / 2;
  const size_t source_y = source_y0 + y * source_height / kLiveHeight;
  const size_t source_x = x * kCameraWidth / kLiveWidth;
  const uint16_t raw10 =
      camera_frame[source_y * kCameraWidth + source_x] & 0x03ff;
  const uint8_t gray = static_cast<uint8_t>((raw10 * 255U + 511U) / 1023U);
  red = green = blue = gray;
}
#endif

#if defined(CONFIG_DUAL_ET_LIVE_VISION)
extern "C" int zephyr_dual_display_init(void);
extern "C" int zephyr_dual_capture_preview(void);
extern "C" int zephyr_dual_release_frame(void);
extern "C" uint16_t* zephyr_dual_preview_buffer(void);
extern "C" void zephyr_dual_get_rgb(
    uint32_t x, uint32_t y, uint8_t* red, uint8_t* green, uint8_t* blue);
extern "C" void zephyr_dual_show_ssd_result(int faces);
extern "C" void zephyr_dual_show_classification(
    int class_id, int confidence, const char* label);
extern "C" void zephyr_dual_reset_results(void);
extern "C" void zephyr_dual_clear_display(void);
extern "C" void zephyr_dual_show_parallel_summary(
    uint32_t samples, uint32_t u55_us, uint32_t u85_us,
    uint32_t span_us, uint32_t overlap_us);
extern "C" void zephyr_dual_show_bmp(const uint8_t* bmp);
uint16_t* live_frame;
/* Written by the coordinator and read by both inference workers.  This must
 * not be an ordinary bool: otherwise the compiler may retain the initial
 * true value in the worker loops and keep feeding the startup BMP forever. */
volatile bool use_test_image = true;

int enable_live_display() {
  const int rc = zephyr_dual_display_init();
  live_frame = zephyr_dual_preview_buffer();
  return rc;
}
int capture_live_frame() { return zephyr_dual_capture_preview(); }

inline void live_rgb(size_t x, size_t y, uint8_t& red, uint8_t& green,
                     uint8_t& blue) {
  if (use_test_image) {
    const auto* bmp = reinterpret_cast<const uint8_t*>(kFaceBmpAddress);
    const size_t bx = x * 192U / kLiveWidth;
    const size_t by = 191U - y * 192U / kLiveHeight;
    const size_t offset = 54U + (by * 192U + bx) * 3U;
    blue = bmp[offset];
    green = bmp[offset + 1U];
    red = bmp[offset + 2U];
    return;
  }
  zephyr_dual_get_rgb(static_cast<uint32_t>(x), static_cast<uint32_t>(y),
                      &red, &green, &blue);
}
#endif

K_THREAD_STACK_DEFINE(u55_thread_stack, 32768);
K_THREAD_STACK_DEFINE(u85_thread_stack, 32768);
struct k_thread u55_thread;
struct k_thread u85_thread;
K_SEM_DEFINE(u55_start_sem, 0, 1);
K_SEM_DEFINE(u85_start_sem, 0, 1);
K_SEM_DEFINE(execute_ready_sem, 0, 2);
K_SEM_DEFINE(u55_execute_go_sem, 0, 1);
K_SEM_DEFINE(u85_execute_go_sem, 0, 1);
/* CPU-side resize/copy is performed before these gates.  Releasing both gates
 * together lets the two independent Ethos-U delegates submit to U55 and U85
 * without the higher-priority input-preparation worker serializing them. */
K_SEM_DEFINE(u55_npu_launch_sem, 0, 1);
K_SEM_DEFINE(u85_npu_launch_sem, 0, 1);
K_SEM_DEFINE(done_sem, 0, 2);
K_SEM_DEFINE(input_copied_sem, 0, 2);
volatile int execute_status[2] = {-1, -1};
volatile uint64_t execute_start_cycles[2];
volatile uint64_t execute_end_cycles[2];
volatile uint64_t prepare_start_cycles[2];
volatile uint64_t prepare_ready_cycles[2];
volatile int latest_class_id = -1;
volatile int latest_class_confidence;
char latest_class_label[20];

#if defined(CONFIG_DUAL_ET_LIVE_VISION)
void populate_model_input(int8_t* image, size_t bytes, Workload workload) {
  const size_t width = workload == Workload::FaceDetection ? 160U : 224U;
  const size_t height = workload == Workload::FaceDetection ? 120U : 224U;
  const size_t channels = workload == Workload::FaceDetection ? 1U : 3U;
  if (bytes != width * height * channels) return;
  const size_t plane = width * height;
  for (size_t y = 0; y < height; ++y) {
    const size_t source_y = y * kLiveHeight / height;
    for (size_t x = 0; x < width; ++x) {
      const size_t source_x = x * kLiveWidth / width;
      uint8_t red, green, blue;
      live_rgb(source_x, source_y, red, green, blue);
      const size_t index = y * width + x;
      if (workload == Workload::FaceDetection) {
        const int gray = static_cast<int>(
            (77U * red + 150U * green + 29U * blue) >> 8);
        image[index] = static_cast<int8_t>(gray - 128);
      } else {
        image[index] = static_cast<int8_t>(static_cast<int>(red) - 128);
        image[plane + index] =
            static_cast<int8_t>(static_cast<int>(green) - 128);
        image[2U * plane + index] =
            static_cast<int8_t>(static_cast<int>(blue) - 128);
      }
    }
  }
}

void refresh_live_input(EValue& retained_input, Workload workload) {
  if (!retained_input.isTensor()) return;
  Tensor tensor = retained_input.toTensor();
  if (tensor.scalar_type() != ScalarType::Char) return;
  int8_t* image = tensor.mutable_data_ptr<int8_t>();
  populate_model_input(image, tensor.nbytes(), workload);
  /* The Ethos-U masters do not snoop the M55 data cache.  Without cleaning
   * this range, method->execute() can keep consuming the startup-test tensor
   * even though the CPU has populated it from a new camera frame. */
  sys_cache_data_flush_range(image, tensor.nbytes());
}
#endif

Error prepare_inputs(Method& method,
                     executorch::runtime::MemoryAllocator& data_allocator,
                     executorch::runtime::MemoryAllocator& metadata_allocator,
                     Workload workload,
                     EValue* retained_input) {
  MethodMeta meta = method.method_meta();
  for (size_t i = 0; i < meta.num_inputs(); ++i) {
    auto tag = meta.input_tag(i);
    if (!tag.ok() || tag.get() != Tag::Tensor) {
      continue;
    }
    auto info = meta.input_tensor_meta(i);
    if (!info.ok()) {
      return info.error();
    }
    void* data = data_allocator.allocate(info->nbytes(), 16);
    if (data == nullptr) {
      printk("dual-et: input allocation failed, need=%zu\n", info->nbytes());
      return Error::MemoryAllocationFailed;
    }
    printk("dual-et: input[%zu] data=%p bytes=%zu elements=%zu\n", i, data,
           info->nbytes(), [&]() {
             size_t count = 1;
             for (size_t dim = 0; dim < info->sizes().size(); ++dim) {
               count *= info->sizes()[dim];
             }
             return count;
           }());

    size_t elements = 1;
    for (size_t dim = 0; dim < info->sizes().size(); ++dim) {
      elements *= info->sizes()[dim];
    }
#if !defined(CONFIG_DUAL_ET_LIVE_VISION)
    const uint8_t* bmp = reinterpret_cast<const uint8_t*>(kFaceBmpAddress);
    if (bmp[0] != 'B' || bmp[1] != 'M') {
      printk("dual-et: shared face BMP missing from SRAM0\n");
      return Error::InvalidProgram;
    }
    const uint8_t* pixels = bmp + 54;
#endif
    const size_t width = workload == Workload::FaceDetection ? 160U : 224U;
    const size_t height = workload == Workload::FaceDetection ? 120U : 224U;
    const size_t channels = workload == Workload::FaceDetection ? 1U : 3U;
    if (info->scalar_type() == ScalarType::Char &&
        elements == channels * width * height) {
      int8_t* image = static_cast<int8_t*>(data);
      populate_model_input(image, info->nbytes(), workload);
      printk("dual-et: prepared input for %s (%zux%zux%zu)\n",
             workload == Workload::FaceDetection ? "U55 SSD" : "U85 MV2",
             width, height, channels);
    } else {
      printk("dual-et: input size/type mismatch, elements=%zu bytes=%zu type=%u\n",
             elements, info->nbytes(),
             static_cast<unsigned>(info->scalar_type()));
      return Error::InvalidArgument;
    }
    void* impl_storage = metadata_allocator.allocate(
        sizeof(TensorImpl), alignof(TensorImpl));
    if (impl_storage == nullptr) {
      return Error::MemoryAllocationFailed;
    }
    auto* impl = new (impl_storage) TensorImpl(
        info->scalar_type(), info->sizes().size(),
        const_cast<TensorImpl::SizesType*>(info->sizes().data()), data,
        const_cast<TensorImpl::DimOrderType*>(info->dim_order().data()));
    *retained_input = EValue(Tensor(impl));
    Error status = method.set_input(*retained_input, i);
    if (status != Error::Ok) {
      return status;
    }
  }
  return Error::Ok;
}

struct Detection {
  float x0, y0, x1, y1, score;
};

float sigmoid(float value) { return 1.0f / (1.0f + expf(-value)); }

void render_ssd_detections(Method& method, bool log_results) {
  std::vector<EValue> outputs(method.outputs_size());
  if (method.get_outputs(outputs.data(), outputs.size()) != Error::Ok ||
      outputs.size() != 2) {
    printk("dual-et: U55 SSD output retrieval failed\n");
    return;
  }
  Tensor boxes = outputs[0].toTensor();
  Tensor logits = outputs[1].toTensor();
  if (boxes.numel() == 1118U * 2U && logits.numel() == 1118U * 4U) {
    Tensor swap = boxes;
    boxes = logits;
    logits = swap;
  }
  if (boxes.scalar_type() != ScalarType::Char ||
      logits.scalar_type() != ScalarType::Char ||
      boxes.numel() != 1118U * 4U || logits.numel() != 1118U * 2U) {
    printk("dual-et: U55 SSD output contract mismatch boxes=%zu logits=%zu\n",
           boxes.numel(), logits.numel());
    return;
  }
  const int8_t* box_data = boxes.const_data_ptr<int8_t>();
  const int8_t* score_data = logits.const_data_ptr<int8_t>();
  sys_cache_data_invd_range(const_cast<int8_t*>(box_data), boxes.nbytes());
  sys_cache_data_invd_range(const_cast<int8_t*>(score_data), logits.nbytes());
  Detection candidates[64];
  size_t count = 0;
  const int feature_h[4] = {15, 8, 4, 2};
  const int feature_w[4] = {20, 10, 5, 3};
  const int anchor_count[4] = {3, 2, 2, 3};
  const float min_boxes[4][3] = {{10, 16, 24}, {32, 48, 0},
                                 {64, 96, 0}, {128, 192, 256}};
  size_t anchor_index = 0;
  for (size_t level = 0; level < 4; ++level) {
    for (int y = 0; y < feature_h[level]; ++y) {
      for (int x = 0; x < feature_w[level]; ++x) {
        for (int anchor = 0; anchor < anchor_count[level];
             ++anchor, ++anchor_index) {
          constexpr float kScoreScale = 0.026616256684064865f;
          constexpr int kScoreZeroPoint = 1;
          const float bg = (static_cast<int>(score_data[anchor_index * 2U]) -
                            kScoreZeroPoint) * kScoreScale;
          const float person =
              (static_cast<int>(score_data[anchor_index * 2U + 1U]) -
               kScoreZeroPoint) * kScoreScale;
          const float score = sigmoid(person - bg);
          if (score < 0.70f || count == ARRAY_SIZE(candidates)) continue;
          auto delta = [&](size_t field) {
            return (static_cast<int>(box_data[anchor_index * 4U + field]) - 71) *
                   0.05725916847586632f;
          };
          const float aw = min_boxes[level][anchor] / 160.0f;
          const float ah = min_boxes[level][anchor] / 120.0f;
          const float anchor_cx =
              (static_cast<float>(x) + 0.5f) / feature_w[level];
          const float anchor_cy =
              (static_cast<float>(y) + 0.5f) / feature_h[level];
          const float cx = anchor_cx + delta(0) * 0.1f * aw;
          const float cy = anchor_cy + delta(1) * 0.1f * ah;
          const float width = expf(delta(2) * 0.2f) * aw;
          const float height = expf(delta(3) * 0.2f) * ah;
          candidates[count++] = {(cx - width / 2.0f) * 192.0f,
                                 (cy - height / 2.0f) * 192.0f,
                                 (cx + width / 2.0f) * 192.0f,
                                 (cy + height / 2.0f) * 192.0f, score};
        }
      }
    }
  }
  for (size_t i = 0; i < count; ++i) {
    size_t best = i;
    for (size_t j = i + 1; j < count; ++j)
      if (candidates[j].score > candidates[best].score) best = j;
    Detection swap = candidates[i];
    candidates[i] = candidates[best];
    candidates[best] = swap;
  }
  Detection selected[10];
  size_t selected_count = 0;
  for (size_t i = 0; i < count && selected_count < 10; ++i) {
    bool suppressed = false;
    for (size_t j = 0; j < selected_count; ++j) {
      const float ix0 = fmaxf(candidates[i].x0, selected[j].x0);
      const float iy0 = fmaxf(candidates[i].y0, selected[j].y0);
      const float ix1 = fminf(candidates[i].x1, selected[j].x1);
      const float iy1 = fminf(candidates[i].y1, selected[j].y1);
      const float intersection = fmaxf(0, ix1 - ix0) * fmaxf(0, iy1 - iy0);
      const float area_a = (candidates[i].x1 - candidates[i].x0) *
                           (candidates[i].y1 - candidates[i].y0);
      const float area_b = (selected[j].x1 - selected[j].x0) *
                           (selected[j].y1 - selected[j].y0);
      const float iou = intersection / (area_a + area_b - intersection);
      const float overlap_min = intersection / fminf(area_a, area_b);
      if (iou > 0.30f || overlap_min > 0.60f) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) selected[selected_count++] = candidates[i];
  }
  if (log_results)
    printk("dual-et: SSD persons=%zu candidates=%zu\n", selected_count, count);
#if defined(CONFIG_DUAL_ET_LIVE_VISION)
  zephyr_dual_show_ssd_result(static_cast<int>(selected_count));
#endif
  for (size_t i = 0; i < selected_count; ++i) {
    if (log_results)
      printk("dual-et: face %zu score=%d%% box=(%d,%d)-(%d,%d)\n", i,
             static_cast<int>(selected[i].score * 100),
             static_cast<int>(selected[i].x0), static_cast<int>(selected[i].y0),
             static_cast<int>(selected[i].x1), static_cast<int>(selected[i].y1));
#if defined(CONFIG_DUAL_ET_LIVE_VISION)
    /* The startup BMP is shown as a centered 272-pixel square. Live camera
     * preview and model input both use the same center crop. */
    const int box_width = static_cast<int>(kLiveHeight);
    const int box_xoff = (static_cast<int>(kLiveWidth) - box_width) / 2;
    int x0 = box_xoff + static_cast<int>(selected[i].x0 * box_width / 192);
    int x1 = box_xoff + static_cast<int>(selected[i].x1 * box_width / 192);
    int y0 = static_cast<int>(selected[i].y0 * kLiveHeight / 192);
    int y1 = static_cast<int>(selected[i].y1 * kLiveHeight / 192);
    x0 = x0 < 0 ? 0 : (x0 >= static_cast<int>(kLiveWidth) ? kLiveWidth - 1 : x0);
    x1 = x1 < 0 ? 0 : (x1 >= static_cast<int>(kLiveWidth) ? kLiveWidth - 1 : x1);
    y0 = y0 < 0 ? 0 : (y0 >= static_cast<int>(kLiveHeight) ? kLiveHeight - 1 : y0);
    y1 = y1 < 0 ? 0 : (y1 >= static_cast<int>(kLiveHeight) ? kLiveHeight - 1 : y1);
    if (x1 > x0 && y1 > y0) {
      constexpr uint16_t kBoxColor = 0x07e0;  // RGB565 green.
      for (int thickness = 0; thickness < 3; ++thickness) {
        for (int x = x0; x <= x1; ++x) {
          live_frame[(y0 + thickness) * kLiveWidth + x] = kBoxColor;
          live_frame[(y1 - thickness) * kLiveWidth + x] = kBoxColor;
        }
        for (int y = y0; y <= y1; ++y) {
          live_frame[y * kLiveWidth + x0 + thickness] = kBoxColor;
          live_frame[y * kLiveWidth + x1 - thickness] = kBoxColor;
        }
      }
    }
#endif
  }
#if defined(CONFIG_DUAL_ET_LIVE_VISION)
  sys_cache_data_flush_range(live_frame, kLiveFrameSize);
#endif
}

void lookup_class_label(size_t class_id, char* label, size_t capacity) {
  const char* text = reinterpret_cast<const char*>(kClassLabelsAddress);
  size_t current = 0;
  while (current < class_id &&
         static_cast<size_t>(text - reinterpret_cast<const char*>(kClassLabelsAddress)) <
             kClassLabelsSize) {
    if (*text++ == '\n') ++current;
  }
  size_t used = 0;
  while (used + 1U < capacity && *text != '\n' && *text != ',' && *text != '\0') {
    const char c = *text++;
    label[used++] = c >= 'a' && c <= 'z' ? static_cast<char>(c - 'a' + 'A') : c;
  }
  label[used] = '\0';
}

void render_classification(Method& method, size_t iteration) {
  std::vector<EValue> outputs(method.outputs_size());
  if (method.get_outputs(outputs.data(), outputs.size()) != Error::Ok ||
      outputs.size() != 1 || !outputs[0].isTensor()) return;
  Tensor output = outputs[0].toTensor();
  if (output.scalar_type() != ScalarType::Char || output.numel() != 1000U) return;
  const int8_t* values = output.const_data_ptr<int8_t>();
  sys_cache_data_invd_range(const_cast<int8_t*>(values), output.nbytes());
  size_t best = 0;
  float denominator = 0.0f;
  for (size_t i = 1; i < 1000U; ++i)
    if (values[i] > values[best]) best = i;
  for (size_t i = 0; i < 1000U; ++i)
    denominator += expf((static_cast<int>(values[i]) - values[best]) *
                        0.019969256594777107f);
  const int confidence = static_cast<int>(100.0f / denominator + 0.5f);
  lookup_class_label(best, latest_class_label, sizeof(latest_class_label));
  latest_class_id = static_cast<int>(best);
  latest_class_confidence = confidence;
  if (iteration < 12U || (iteration % 30U) == 0U)
    printk("dual-et: frame=%zu CLASS=%zu %s confidence=%d%% raw=%d\n",
           iteration, best, latest_class_label, confidence, values[best]);
#if defined(CONFIG_DUAL_ET_LIVE_VISION)
  zephyr_dual_show_classification(static_cast<int>(best), confidence,
                                  latest_class_label);
#endif
}

int run_model(const char* name, const unsigned char* pte, size_t pte_size,
              unsigned char* method_pool, size_t method_pool_size,
              unsigned char* temp_pool, size_t temp_pool_size,
              Workload workload, size_t worker_id) {
  struct k_sem* execute_go = worker_id == 0 ? &u55_execute_go_sem
                                             : &u85_execute_go_sem;
  struct k_sem* npu_launch = worker_id == 0 ? &u55_npu_launch_sem
                                             : &u85_npu_launch_sem;
  prepare_start_cycles[worker_id] = k_cycle_get_64();
  auto setup_failure = [execute_go, worker_id](int result) {
    /* Always satisfy the coordinator barrier. The healthy worker may still
     * execute, and both threads will terminate instead of deadlocking.
     */
    prepare_ready_cycles[worker_id] = k_cycle_get_64();
    for (;;) {
      k_sem_give(&execute_ready_sem);
      k_sem_take(execute_go, K_FOREVER);
      k_sem_give(&done_sem);
    }
    return result;
  };
  printk("dual-et: %s loading PTE (%zu bytes)\n", name, pte_size);
  BufferDataLoader loader(pte, pte_size);
  Result<Program> program = Program::load(&loader);
  if (!program.ok()) {
    printk("dual-et: %s Program::load failed 0x%08x\n", name,
           static_cast<unsigned>(program.error()));
    return setup_failure(1);
  }

  auto method_name = program->get_method_name(0);
  if (!method_name.ok()) {
    return setup_failure(2);
  }
  auto meta = program->method_meta(*method_name);
  if (!meta.ok()) {
    return setup_failure(3);
  }

  ArmMemoryAllocator data_allocator(method_pool_size, method_pool);
  unsigned char* metadata_pool = worker_id == 0 ? u55_metadata_pool
                                                 : u85_metadata_pool;
  ArmMemoryAllocator method_allocator(kMethodMetadataPoolSize, metadata_pool);
  printk("dual-et: %s data-pool=%p..%p (%zu bytes) metadata=%p/%zu\n", name,
         method_pool, method_pool + method_pool_size, method_pool_size,
         metadata_pool, kMethodMetadataPoolSize);
  std::vector<uint8_t*> buffers;
  std::vector<Span<uint8_t>> spans;
  for (size_t i = 0; i < meta->num_memory_planned_buffers(); ++i) {
    size_t size = meta->memory_planned_buffer_size(i).get();
    auto* buffer = static_cast<uint8_t*>(data_allocator.allocate(size, 16));
    if (buffer == nullptr) {
      printk("dual-et: %s planned buffer %zu failed (%zu)\n", name, i, size);
      return setup_failure(4);
    }
    buffers.push_back(buffer);
    spans.emplace_back(buffer, size);
  }

  HierarchicalAllocator planned({spans.data(), spans.size()});
  printk("dual-et: %s planned buffers=%zu data-used=%zu free=%zu\n",
         name, spans.size(), data_allocator.used_size(),
         data_allocator.free_size());
  ArmMemoryAllocator temp_allocator(temp_pool_size, temp_pool);
  MemoryManager manager(&method_allocator, &planned, &temp_allocator);
  Result<Method> method = program->load_method(*method_name, &manager, nullptr);
  if (!method.ok()) {
    printk("dual-et: %s load_method failed 0x%08x\n", name,
           static_cast<unsigned>(method.error()));
    return setup_failure(5);
  }
  printk("dual-et: %s method=%p metadata-used=%zu free=%zu\n", name,
         &method.get(), method_allocator.used_size(),
         method_allocator.free_size());
  printk("dual-et: %s temp after-load used=%zu free=%zu\n", name,
         temp_allocator.used_size(), temp_allocator.free_size());
  EValue retained_input;
  Error status = prepare_inputs(*method, data_allocator, method_allocator,
                                workload, &retained_input);
  if (status != Error::Ok) {
    printk("dual-et: %s input failed 0x%08x\n", name,
           static_cast<unsigned>(status));
    return setup_failure(6);
  }
  printk("dual-et: %s after-input data-used=%zu metadata-used=%zu\n", name,
         data_allocator.used_size(), method_allocator.used_size());

  prepare_ready_cycles[worker_id] = k_cycle_get_64();
  size_t unused_stack = 0;
  int stack_rc = k_thread_stack_space_get(k_current_get(), &unused_stack);
  printk("dual-et: %s prepared stack-unused=%zu rc=%d\n", name,
         unused_stack, stack_rc);
  Error aggregate_status = Error::Ok;
  for (size_t iteration = 0;; ++iteration) {
    k_sem_give(&execute_ready_sem);
    k_sem_take(execute_go, K_FOREVER);
#if defined(CONFIG_DUAL_ET_LIVE_VISION)
    refresh_live_input(retained_input, workload);
    /* set_input() made an initial copy into the Method's memory-planned input
     * during preparation. Refreshing only retained_input therefore changes
     * the caller buffer, not necessarily the tensor consumed by CALL_DELEGATE.
     * Rebind/copy it for every frame, then clean the actual bound buffer. */
    status = method->set_input(retained_input, 0);
    if (status != Error::Ok) {
      printk("dual-et: %s live set_input failed 0x%08x\n", name,
             static_cast<unsigned>(status));
      execute_status[worker_id] = 8;
      if (!use_test_image) k_sem_give(&input_copied_sem);
      k_sem_give(&done_sem);
      continue;
    }
    EValue bound_input;
    if (method->get_inputs(&bound_input, 1) == Error::Ok &&
        bound_input.isTensor()) {
      Tensor bound = bound_input.toTensor();
      sys_cache_data_flush_range(bound.mutable_data_ptr<int8_t>(),
                                 bound.nbytes());
      if (iteration < 12U || (iteration % 30U) == 0U) {
        const int8_t *bound_data = bound.const_data_ptr<int8_t>();
        uint32_t checksum = 2166136261U;
        for (size_t i = 0; i < bound.nbytes(); ++i)
          checksum = (checksum ^ static_cast<uint8_t>(bound_data[i])) *
                     16777619U;
        printk("dual-et: %s rebound src=%p dst=%p bytes=%zu checksum=%08x samples=%d/%d/%d\n", name,
               retained_input.toTensor().mutable_data_ptr<int8_t>(),
               bound.mutable_data_ptr<int8_t>(), bound.nbytes(), checksum,
               bound_data[0], bound_data[bound.nbytes() / 2U],
               bound_data[bound.nbytes() - 1U]);
      }
    }
    /* Both model tensors now own a copy of the camera pixels, so the
     * coordinator may immediately return the capture buffer to CPI. */
    if (!use_test_image) {
      k_sem_give(&input_copied_sem);
      /* Do not let whichever worker finishes preprocessing first submit and
       * finish its NPU job before the other input is ready. */
      k_sem_take(npu_launch, K_FOREVER);
    }
#endif
    uint64_t start = k_cycle_get_64();
    execute_start_cycles[worker_id] = start;
    status = method->execute();
    execute_status[worker_id] = status == Error::Ok ? 0 : 7;
    uint64_t end = k_cycle_get_64();
    execute_end_cycles[worker_id] = end;
    if (status != Error::Ok) {
      aggregate_status = status;
    }

    if (status == Error::Ok && workload == Workload::ImageClassification)
      render_classification(*method, iteration);
    if (status == Error::Ok &&
        workload == Workload::FaceDetection) {
      render_ssd_detections(*method, iteration < 2U || (iteration % 30U) == 0U);
    }
    k_sem_give(&done_sem);
  }
  return aggregate_status == Error::Ok ? 0 : 7;
}

void u55_thread_entry(void*, void*, void*) {
  k_sem_take(&u55_start_sem, K_FOREVER);
  (void)run_model("U55-SSD", model_u55_pte, kU55PteSize,
                  u55_method_pool, sizeof(u55_method_pool),
                  u55_temp_pool, sizeof(u55_temp_pool),
                  Workload::FaceDetection, 0);
}

void u85_thread_entry(void*, void*, void*) {
  k_sem_take(&u85_start_sem, K_FOREVER);
  (void)run_model("U85-MV2", model_u85_pte, kU85PteSize,
                  u85_method_pool, sizeof(u85_method_pool),
                  u85_temp_pool, sizeof(u85_temp_pool),
                  Workload::ImageClassification, 1);
}

}  // namespace

int main(void) {
  printk("*** dual ExecuTorch parallel SSD(U55) + MobileNetV2(U85) ***\n");
#if defined(CONFIG_DUAL_ET_LIVE_VISION)
#if defined(CONFIG_DUAL_ET_LEGACY_NATIVE_VISION)
  printk("dual-et: RUN profile rc=%d memory=%08x->%08x\n",
         run_profile_status, run_memory_before, run_memory_after);
  if (run_profile_status != 0) {
    printk("dual-et: RUN profile setup failed; model staging skipped\n");
    return 1;
  }
#endif
  if (enable_live_display() != 0) {
    printk("dual-et: display initialization failed\n");
    return 1;
  }
#endif
  constexpr uint32_t kPteOffset = 0x00000024U;
  constexpr uint32_t kPteMagic = 0x32315445U;
#if defined(CONFIG_DUAL_ET_LIVE_VISION)
  const volatile uint32_t* stored_u85 =
      reinterpret_cast<const volatile uint32_t*>(kU85PteAddress);
  const volatile uint32_t* stored_u55 =
      reinterpret_cast<const volatile uint32_t*>(kU55PteAddress);
  const volatile uint16_t* stored_bmp =
      reinterpret_cast<const volatile uint16_t*>(kFaceBmpAddress);
  sys_cache_data_invd_range(reinterpret_cast<void*>(kU85PteAddress),
                            kU85PteSize);
  sys_cache_data_invd_range(reinterpret_cast<void*>(kU85PteAddress),
                            kModelPayloadSize);
  if (stored_u85[0] != kPteOffset || stored_u85[1] != kPteMagic ||
      stored_u55[0] != kPteOffset || stored_u55[1] != kPteMagic ||
      stored_bmp[0] != 0x4d42U) {
    printk("dual-et: MRAM payload invalid U85=%08x/%08x U55=%08x/%08x BMP=%04x\n",
           stored_u85[0], stored_u85[1], stored_u55[0], stored_u55[1],
           stored_bmp[0]);
    return 1;
  }
  int cache_rc = 0;
#else
  int cache_rc = sys_cache_data_invd_range(
      reinterpret_cast<void*>(kU85PteAddress), kPayloadSize);
#endif
  const volatile uint32_t* u85_header =
      reinterpret_cast<const volatile uint32_t*>(kU85PteAddress);
  printk("dual-et: U85 MobileNetV2 PTE=%p/%zu\n",
         model_u85_pte, kU85PteSize);
#if defined(CONFIG_DUAL_ET_LIVE_VISION)
  printk("dual-et: U55 PTE=%p size=%zu live MT9M114 input enabled\n",
         model_u55_pte, kU55PteSize);
#else
  printk("dual-et: U55 PTE=%p size=%zu face BMP=%p size=%zu\n",
         model_u55_pte, kU55PteSize,
         reinterpret_cast<const void*>(kFaceBmpAddress), kFaceBmpSize);
#endif
  if (dual_ethosu_init() != 0) {
    printk("dual-et: NPU initialization failed\n");
    return 1;
  }
  ARG_UNUSED(cache_rc);
  ARG_UNUSED(u85_header);
  executorch::runtime::runtime_init();
  if (executorch_delegate_EthosUBackend_registered() != Error::Ok) {
    printk("dual-et: Ethos-U delegate registration failed\n");
    return 1;
  }
#if defined(CONFIG_DUAL_ET_LIVE_VISION)
  sys_cache_data_invd_range(reinterpret_cast<void*>(kFaceBmpAddress),
                            kFaceBmpSize);
  zephyr_dual_show_bmp(reinterpret_cast<const uint8_t*>(kFaceBmpAddress));
  printk("dual-et: startup self-test image displayed\n");
#elif defined(CONFIG_DUAL_ET_LIVE_VISION)
  if (capture_live_frame() != 0) return 1;
#endif

  printk("dual-et: starting parallel worker threads\n");
  k_thread_create(&u55_thread, u55_thread_stack,
                  K_THREAD_STACK_SIZEOF(u55_thread_stack), u55_thread_entry,
                  nullptr, nullptr, nullptr, 5, 0, K_NO_WAIT);
  k_thread_create(&u85_thread, u85_thread_stack,
                  K_THREAD_STACK_SIZEOF(u85_thread_stack), u85_thread_entry,
                  nullptr, nullptr, nullptr, 5, 0, K_NO_WAIT);
  /* ExecuTorch program/method construction uses shared runtime machinery.
   * Prepare each method serially, then execute only the immutable prepared
   * methods in parallel on their independent NPU backends.
   */
  k_sem_give(&u55_start_sem);
  k_sem_take(&execute_ready_sem, K_FOREVER);
#if defined(CONFIG_DUAL_ET_LIVE_VISION)
  printk("dual-et: U85 MobileNetV2 remains in MRAM %p (%zu bytes)\n",
         model_u85_pte, kU85PteSize);
#endif
  k_sem_give(&u85_start_sem);
  k_sem_take(&execute_ready_sem, K_FOREVER);
  printk("dual-et: prepare U55=%llu us U85=%llu us\n",
         static_cast<unsigned long long>(k_cyc_to_us_floor64(
             prepare_ready_cycles[0] - prepare_start_cycles[0])),
         static_cast<unsigned long long>(k_cyc_to_us_floor64(
             prepare_ready_cycles[1] - prepare_start_cycles[1])));

  /* Run each method once without concurrent ExecuTorch host activity. This
   * distinguishes a malformed prepared method from a host-runtime race before
   * the NPU delegates are submitted in parallel.
   */
  printk("dual-et: isolated preflight U55 irqs=%u/%u\n",
         dual_ethosu_u55_irqs(), dual_ethosu_u85_irqs());
  k_sem_give(&u55_execute_go_sem);
  k_sem_take(&done_sem, K_FOREVER);
  k_sem_take(&execute_ready_sem, K_FOREVER);
  printk("dual-et: isolated U55 done irqs=%u/%u submit=%llu irq=%llu\n",
         dual_ethosu_u55_irqs(), dual_ethosu_u85_irqs(),
         static_cast<unsigned long long>(dual_ethosu_u55_submit_cycle()),
         static_cast<unsigned long long>(dual_ethosu_u55_irq_cycle()));
  printk("dual-et: isolated preflight U85\n");
  k_sem_give(&u85_execute_go_sem);
  k_sem_take(&done_sem, K_FOREVER);
  k_sem_take(&execute_ready_sem, K_FOREVER);
  printk("dual-et: isolated U85 done irqs=%u/%u submit=%llu irq=%llu\n",
         dual_ethosu_u55_irqs(), dual_ethosu_u85_irqs(),
         static_cast<unsigned long long>(dual_ethosu_u85_submit_cycle()),
         static_cast<unsigned long long>(dual_ethosu_u85_irq_cycle()));
  if (execute_status[0] != 0 || execute_status[1] != 0) {
    printk("dual-et: preflight failed U55=%d U85=%d\n",
           execute_status[0], execute_status[1]);
    return 1;
  }
#if defined(CONFIG_DUAL_ET_LIVE_VISION)
  printk("dual-et: startup self-test passed; switching to camera in 5 seconds\n");
  k_msleep(5000);
  zephyr_dual_clear_display();
  use_test_image = false;
  /* Publish unambiguous defaults only after the old image and annotations
   * have been completely removed from the scanout buffer. */
  zephyr_dual_reset_results();
#endif
  printk("dual-et: live dual-NPU pipeline started\n");
  uint64_t u55_total_us = 0;
  uint64_t u85_total_us = 0;
  uint64_t span_total_us = 0;
  uint64_t overlap_total_us = 0;
  uint32_t inference_samples = 0;
  constexpr size_t kRollingWindow = 16U;
  uint32_t rolling_u55[kRollingWindow] = {};
  uint32_t rolling_u85[kRollingWindow] = {};
  uint32_t rolling_span[kRollingWindow] = {};
  uint32_t rolling_overlap[kRollingWindow] = {};
  uint64_t rolling_u55_sum = 0, rolling_u85_sum = 0;
  uint64_t rolling_span_sum = 0, rolling_overlap_sum = 0;
  size_t rolling_index = 0, rolling_count = 0;
  for (size_t frame = 1;; ++frame) {
    const int capture_rc = capture_live_frame();
    if (capture_rc != 0) {
      printk("dual-et: frame=%zu capture failed rc=%d\n", frame, capture_rc);
      k_msleep(100);
      continue;
    }
    /* Keep the most recently completed result visible while this frame is
     * processed. Each worker replaces its result only after inference. */
    k_sem_give(&u85_execute_go_sem);
    k_sem_give(&u55_execute_go_sem);
    k_sem_take(&input_copied_sem, K_FOREVER);
    k_sem_take(&input_copied_sem, K_FOREVER);
    const int release_rc = zephyr_dual_release_frame();
    if (release_rc != 0)
      printk("dual-et: frame=%zu release failed rc=%d\n", frame, release_rc);
    /* Both bound tensors are now stable. Start both delegates as one launch
     * phase; their NPU execution intervals are measured below. */
    k_sem_give(&u55_npu_launch_sem);
    k_sem_give(&u85_npu_launch_sem);
    k_sem_take(&done_sem, K_FOREVER);
    k_sem_take(&done_sem, K_FOREVER);
    if (execute_status[0] != 0 || execute_status[1] != 0) {
      printk("dual-et: frame=%zu inference failed U55=%d U85=%d\n", frame,
             execute_status[0], execute_status[1]);
      return 1;
    }
    k_sem_take(&execute_ready_sem, K_FOREVER);
    k_sem_take(&execute_ready_sem, K_FOREVER);
    {
      const uint64_t u55_us = k_cyc_to_us_floor64(
          execute_end_cycles[0] - execute_start_cycles[0]);
      const uint64_t u85_us = k_cyc_to_us_floor64(
          execute_end_cycles[1] - execute_start_cycles[1]);
      const uint64_t first_start =
          execute_start_cycles[0] < execute_start_cycles[1]
              ? execute_start_cycles[0] : execute_start_cycles[1];
      const uint64_t last_end =
          execute_end_cycles[0] > execute_end_cycles[1]
              ? execute_end_cycles[0] : execute_end_cycles[1];
      const uint64_t overlap_start =
          execute_start_cycles[0] > execute_start_cycles[1]
              ? execute_start_cycles[0] : execute_start_cycles[1];
      const uint64_t overlap_end =
          execute_end_cycles[0] < execute_end_cycles[1]
              ? execute_end_cycles[0] : execute_end_cycles[1];
      const uint64_t span_us = k_cyc_to_us_floor64(last_end - first_start);
      const uint64_t overlap_us = overlap_end > overlap_start
          ? k_cyc_to_us_floor64(overlap_end - overlap_start) : 0U;
      ++inference_samples;
      u55_total_us += u55_us;
      u85_total_us += u85_us;
      span_total_us += span_us;
      overlap_total_us += overlap_us;

      if (rolling_count == kRollingWindow) {
        rolling_u55_sum -= rolling_u55[rolling_index];
        rolling_u85_sum -= rolling_u85[rolling_index];
        rolling_span_sum -= rolling_span[rolling_index];
        rolling_overlap_sum -= rolling_overlap[rolling_index];
      } else {
        ++rolling_count;
      }
      rolling_u55[rolling_index] = static_cast<uint32_t>(u55_us);
      rolling_u85[rolling_index] = static_cast<uint32_t>(u85_us);
      rolling_span[rolling_index] = static_cast<uint32_t>(span_us);
      rolling_overlap[rolling_index] = static_cast<uint32_t>(overlap_us);
      rolling_u55_sum += u55_us;
      rolling_u85_sum += u85_us;
      rolling_span_sum += span_us;
      rolling_overlap_sum += overlap_us;
      rolling_index = (rolling_index + 1U) % kRollingWindow;
      zephyr_dual_show_parallel_summary(
          static_cast<uint32_t>(rolling_count),
          static_cast<uint32_t>(rolling_u55_sum / rolling_count),
          static_cast<uint32_t>(rolling_u85_sum / rolling_count),
          static_cast<uint32_t>(rolling_span_sum / rolling_count),
          static_cast<uint32_t>(rolling_overlap_sum / rolling_count));
      if (frame == 1U || (frame % 10U) == 0U) {
        printk("dual-et: PAR class=%d/%s frame=%zu sample=%u "
               "U55=%llu U85=%llu span=%llu overlap=%llu us "
               "avg U55/U85/span/overlap=%llu/%llu/%llu/%llu us\n",
               latest_class_id, latest_class_label, frame,
               inference_samples,
               (unsigned long long)u55_us, (unsigned long long)u85_us,
               (unsigned long long)span_us, (unsigned long long)overlap_us,
               (unsigned long long)(u55_total_us / inference_samples),
               (unsigned long long)(u85_total_us / inference_samples),
               (unsigned long long)(span_total_us / inference_samples),
               (unsigned long long)(overlap_total_us / inference_samples));
      }
    }
    if ((frame % 30U) == 0U)
      printk("dual-et: live frame=%zu IRQs=%u/%u\n", frame,
             dual_ethosu_u55_irqs(), dual_ethosu_u85_irqs());
  }
}
