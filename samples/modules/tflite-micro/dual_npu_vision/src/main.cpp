/* SPDX-License-Identifier: Apache-2.0 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <zephyr/cache.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include "model_layout.h"

extern "C" int dual_ethosu_init(void);
extern "C" unsigned dual_ethosu_u55_irqs(void);
extern "C" unsigned dual_ethosu_u85_irqs(void);
extern "C" int zephyr_dual_display_init(void);
extern "C" int zephyr_dual_capture_preview(void);
extern "C" int zephyr_dual_release_frame(void);
extern "C" uint16_t *zephyr_dual_preview_buffer(void);
extern "C" void zephyr_dual_get_rgb(uint32_t, uint32_t, uint8_t *, uint8_t *, uint8_t *);
extern "C" void zephyr_dual_show_vww_result(int);
extern "C" void zephyr_dual_show_yolo_result(int);
extern "C" void zephyr_dual_reset_results(void);
extern "C" void zephyr_dual_clear_display(void);
extern "C" void zephyr_dual_show_parallel_summary(uint32_t, uint32_t, uint32_t,
                                                    uint32_t, uint32_t);
extern "C" void zephyr_dual_show_bmp(const uint8_t *);

namespace {

enum class Workload { FaceDetection, VisualWakeWord };
constexpr uintptr_t kPayloadAddress = 0x80008000U;
constexpr uintptr_t kU85ModelAddress = kPayloadAddress;
constexpr uintptr_t kU55ModelAddress = kU85ModelAddress + U85_MODEL_SIZE;
constexpr uintptr_t kFaceBmpAddress = kU55ModelAddress + U55_MODEL_SIZE;
constexpr size_t kPayloadSize = U85_MODEL_SIZE + U55_MODEL_SIZE + FACE_BMP_SIZE;
constexpr size_t kLiveWidth = 480U;
constexpr size_t kLiveHeight = 352U;
const uint8_t *const u85_model = reinterpret_cast<const uint8_t *>(kU85ModelAddress);
const uint8_t *const u55_model = reinterpret_cast<const uint8_t *>(kU55ModelAddress);

/* Vela reports approximately 361 KiB for YOLO and 129 KiB for VWW. */
alignas(32) uint8_t u55_tensor_arena[640U * 1024U];
__attribute__((section(".alif_sram1.tensor_arena"), aligned(32)))
uint8_t u85_tensor_arena[384U * 1024U];

uint16_t *live_frame;
volatile bool use_test_image = true;
volatile bool latest_vww_person;
volatile int execute_status[2] = {-1, -1};
volatile uint64_t execute_start_cycles[2], execute_end_cycles[2];
volatile uint64_t prepare_start_cycles[2], prepare_ready_cycles[2];

K_THREAD_STACK_DEFINE(u55_thread_stack, 32768);
K_THREAD_STACK_DEFINE(u85_thread_stack, 32768);
struct k_thread u55_thread, u85_thread;
K_SEM_DEFINE(u55_start_sem, 0, 1);
K_SEM_DEFINE(u85_start_sem, 0, 1);
K_SEM_DEFINE(execute_ready_sem, 0, 2);
K_SEM_DEFINE(u55_execute_go_sem, 0, 1);
K_SEM_DEFINE(u85_execute_go_sem, 0, 1);
K_SEM_DEFINE(done_sem, 0, 2);
K_SEM_DEFINE(input_copied_sem, 0, 2);

void live_rgb(size_t x, size_t y, uint8_t &red, uint8_t &green, uint8_t &blue) {
  if (use_test_image) {
    const auto *bmp = reinterpret_cast<const uint8_t *>(kFaceBmpAddress);
    const size_t bx = x * 192U / kLiveWidth;
    const size_t by = 191U - y * 192U / kLiveHeight;
    const size_t offset = 54U + (by * 192U + bx) * 3U;
    blue = bmp[offset]; green = bmp[offset + 1U]; red = bmp[offset + 2U];
  } else {
    zephyr_dual_get_rgb(static_cast<uint32_t>(x), static_cast<uint32_t>(y),
                        &red, &green, &blue);
  }
}

void populate_input(TfLiteTensor *input, Workload workload) {
  auto *image = input->data.int8;
  const size_t side = workload == Workload::FaceDetection ? 192U : 128U;
  for (size_t y = 0; y < side; ++y) {
    const size_t sy = y * kLiveHeight / side;
    for (size_t x = 0; x < side; ++x) {
      uint8_t red, green, blue;
      live_rgb(x * kLiveWidth / side, sy, red, green, blue);
      const int gray = static_cast<int>((77U * red + 150U * green + 29U * blue) >> 8);
      const int q = static_cast<int>(roundf((gray / 255.0f) / input->params.scale)) +
                    input->params.zero_point;
      image[y * side + x] = static_cast<int8_t>(std::clamp(q, -128, 127));
    }
  }
  sys_cache_data_flush_range(image, input->bytes);
}

struct Detection { float x0, y0, x1, y1, score; };
float sigmoid(float value) { return 1.0f / (1.0f + expf(-value)); }

void render_yolo(TfLiteTensor *outputs[2], bool log_results) {
  Detection candidates[64];
  size_t count = 0;
  const int anchors[2][6] = {{38, 77, 47, 97, 61, 126},
                             {14, 26, 19, 37, 28, 55}};
  const int resolution[2] = {6, 12};
  for (size_t branch = 0; branch < 2; ++branch) {
    TfLiteTensor *tensor = outputs[branch];
    if (tensor == nullptr || tensor->type != kTfLiteInt8) continue;
    sys_cache_data_invd_range(tensor->data.raw, tensor->bytes);
    const int8_t *values = tensor->data.int8;
    const int grid = resolution[branch];
    for (int y = 0; y < grid; ++y) for (int x = 0; x < grid; ++x)
      for (int anchor = 0; anchor < 3; ++anchor) {
        const size_t base = (y * grid + x) * 18 + anchor * 6;
        auto dequant = [&](size_t field) {
          return (static_cast<int>(values[base + field]) - tensor->params.zero_point) *
                 tensor->params.scale;
        };
        const float score = sigmoid(dequant(4)) * sigmoid(dequant(5));
        if (score < 0.75f || count == ARRAY_SIZE(candidates)) continue;
        const float cx = (sigmoid(dequant(0)) + x) / grid * 192.0f;
        const float cy = (sigmoid(dequant(1)) + y) / grid * 192.0f;
        const float width = expf(dequant(2)) * anchors[branch][anchor * 2];
        const float height = expf(dequant(3)) * anchors[branch][anchor * 2 + 1];
        candidates[count++] = {cx - width / 2, cy - height / 2,
                               cx + width / 2, cy + height / 2, score};
      }
  }
  std::sort(candidates, candidates + count,
            [](const Detection &a, const Detection &b) { return a.score > b.score; });
  Detection selected[10];
  size_t selected_count = 0;
  for (size_t i = 0; i < count && selected_count < ARRAY_SIZE(selected); ++i) {
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
      if (intersection / (area_a + area_b - intersection) > 0.45f) {
        suppressed = true; break;
      }
    }
    if (!suppressed) selected[selected_count++] = candidates[i];
  }
  if (log_results) printk("dual-tflm: YOLO faces=%zu candidates=%zu\n", selected_count, count);
  zephyr_dual_show_yolo_result(static_cast<int>(selected_count));
  for (size_t i = 0; i < selected_count; ++i) {
    if (log_results)
      printk("dual-tflm: face %zu score=%d%% box=(%d,%d)-(%d,%d)\n", i,
             static_cast<int>(selected[i].score * 100),
             static_cast<int>(selected[i].x0), static_cast<int>(selected[i].y0),
             static_cast<int>(selected[i].x1), static_cast<int>(selected[i].y1));
    const int width = static_cast<int>(kLiveHeight);
    const int xoff = (static_cast<int>(kLiveWidth) - width) / 2;
    int x0 = std::clamp(xoff + static_cast<int>(selected[i].x0 * width / 192), 0, 479);
    int x1 = std::clamp(xoff + static_cast<int>(selected[i].x1 * width / 192), 0, 479);
    int y0 = std::clamp(static_cast<int>(selected[i].y0 * kLiveHeight / 192), 0, 351);
    int y1 = std::clamp(static_cast<int>(selected[i].y1 * kLiveHeight / 192), 0, 351);
    if (x1 > x0 && y1 > y0) for (int t = 0; t < 3; ++t) {
      for (int x = x0; x <= x1; ++x) {
        live_frame[(y0 + t) * kLiveWidth + x] = 0x07e0;
        live_frame[(y1 - t) * kLiveWidth + x] = 0x07e0;
      }
      for (int y = y0; y <= y1; ++y) {
        live_frame[y * kLiveWidth + x0 + t] = 0x07e0;
        live_frame[y * kLiveWidth + x1 - t] = 0x07e0;
      }
    }
  }
  sys_cache_data_flush_range(live_frame, kLiveWidth * kLiveHeight * 2U);
}

int run_model(const char *name, const uint8_t *model_data,
              uint8_t *arena, size_t arena_size,
              Workload workload, size_t worker_id) {
  struct k_sem *go = worker_id == 0 ? &u55_execute_go_sem : &u85_execute_go_sem;
  prepare_start_cycles[worker_id] = k_cycle_get_64();
  auto setup_failed = [&](int status) {
    execute_status[worker_id] = status;
    prepare_ready_cycles[worker_id] = k_cycle_get_64();
    printk("dual-tflm: %s setup failed rc=%d\n", name, status);
    k_sem_give(&execute_ready_sem);
    return status;
  };
  const tflite::Model *model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) return setup_failed(1);
  tflite::MicroMutableOpResolver<1> resolver;
  if (resolver.AddEthosU() != kTfLiteOk) return setup_failed(2);
  tflite::MicroInterpreter interpreter(model, resolver, arena, arena_size);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printk("dual-tflm: %s AllocateTensors failed arena=%p/%zu\n", name, arena, arena_size);
    return setup_failed(3);
  }
  TfLiteTensor *input = interpreter.input(0);
  if (input == nullptr || input->type != kTfLiteInt8) return setup_failed(4);
  printk("dual-tflm: %s ready arena=%p/%zu input=%p/%zu outputs=%zu\n",
         name, arena, arena_size, input->data.raw, input->bytes,
         interpreter.outputs_size());
  prepare_ready_cycles[worker_id] = k_cycle_get_64();
  execute_status[worker_id] = 0;

  for (size_t iteration = 0;; ++iteration) {
    k_sem_give(&execute_ready_sem);
    k_sem_take(go, K_FOREVER);
    populate_input(input, workload);
    if (!use_test_image) k_sem_give(&input_copied_sem);
    execute_start_cycles[worker_id] = k_cycle_get_64();
    const TfLiteStatus status = interpreter.Invoke();
    execute_end_cycles[worker_id] = k_cycle_get_64();
    execute_status[worker_id] = status == kTfLiteOk ? 0 : 5;
    if (status == kTfLiteOk && workload == Workload::VisualWakeWord) {
      TfLiteTensor *output = interpreter.output(0);
      sys_cache_data_invd_range(output->data.raw, output->bytes);
      const int8_t *scores = output->data.int8;
      const bool person = static_cast<int>(scores[1]) >=
                          static_cast<int>(scores[0]) + 48;
      latest_vww_person = person;
      zephyr_dual_show_vww_result(person ? 1 : 0);
      if (iteration < 12U || iteration % 30U == 0U)
        printk("dual-tflm: frame=%zu VWW=%s scores=%d/%d\n", iteration,
               person ? "PERSON" : "NO_PERSON", scores[0], scores[1]);
    } else if (status == kTfLiteOk) {
      TfLiteTensor *outputs[2] = {interpreter.output(0), interpreter.output(1)};
      render_yolo(outputs, iteration % 30U == 0U);
    }
    k_sem_give(&done_sem);
  }
}

void u55_thread_entry(void *, void *, void *) {
  k_sem_take(&u55_start_sem, K_FOREVER);
  (void)run_model("U55-YOLO", u55_model, u55_tensor_arena,
                  sizeof(u55_tensor_arena), Workload::FaceDetection, 0);
}
void u85_thread_entry(void *, void *, void *) {
  k_sem_take(&u85_start_sem, K_FOREVER);
  (void)run_model("U85-VWW", u85_model, u85_tensor_arena,
                  sizeof(u85_tensor_arena), Workload::VisualWakeWord, 1);
}

}  // namespace

int main(void) {
  printk("*** dual TFLite Micro parallel YOLO(U55) + VWW(U85) ***\n");
  if (zephyr_dual_display_init() != 0) {
    printk("dual-tflm: display/camera initialization failed\n");
    return 1;
  }
  live_frame = zephyr_dual_preview_buffer();
  sys_cache_data_invd_range(reinterpret_cast<void *>(kPayloadAddress), kPayloadSize);
  const auto *u85 = reinterpret_cast<const volatile uint8_t *>(kU85ModelAddress);
  const auto *u55 = reinterpret_cast<const volatile uint8_t *>(kU55ModelAddress);
  const auto *bmp = reinterpret_cast<const volatile uint16_t *>(kFaceBmpAddress);
  if (u85[4] != 'T' || u85[5] != 'F' || u85[6] != 'L' || u85[7] != '3' ||
      u55[4] != 'T' || u55[5] != 'F' || u55[6] != 'L' || u55[7] != '3' ||
      bmp[0] != 0x4d42U) {
    printk("dual-tflm: payload invalid\n");
    return 1;
  }
  printk("dual-tflm: models U55=%p/%u U85=%p/%u BMP=%p/%u\n",
         u55_model, U55_MODEL_SIZE, u85_model, U85_MODEL_SIZE,
         reinterpret_cast<const void *>(kFaceBmpAddress), FACE_BMP_SIZE);
  if (dual_ethosu_init() != 0) {
    printk("dual-tflm: NPU initialization failed\n");
    return 1;
  }
  zephyr_dual_show_bmp(reinterpret_cast<const uint8_t *>(kFaceBmpAddress));
  printk("dual-tflm: startup self-test image displayed\n");

  k_thread_create(&u55_thread, u55_thread_stack,
                  K_THREAD_STACK_SIZEOF(u55_thread_stack), u55_thread_entry,
                  nullptr, nullptr, nullptr, 5, 0, K_NO_WAIT);
  k_thread_create(&u85_thread, u85_thread_stack,
                  K_THREAD_STACK_SIZEOF(u85_thread_stack), u85_thread_entry,
                  nullptr, nullptr, nullptr, 4, 0, K_NO_WAIT);
  k_sem_give(&u55_start_sem);
  k_sem_take(&execute_ready_sem, K_FOREVER);
  k_sem_give(&u85_start_sem);
  k_sem_take(&execute_ready_sem, K_FOREVER);
  printk("dual-tflm: prepare U55=%llu us U85=%llu us\n",
         (unsigned long long)k_cyc_to_us_floor64(prepare_ready_cycles[0] - prepare_start_cycles[0]),
         (unsigned long long)k_cyc_to_us_floor64(prepare_ready_cycles[1] - prepare_start_cycles[1]));
  if (execute_status[0] != 0 || execute_status[1] != 0) {
    printk("dual-tflm: model setup failed U55=%d U85=%d\n",
           execute_status[0], execute_status[1]);
    return 1;
  }

  k_sem_give(&u55_execute_go_sem);
  k_sem_take(&done_sem, K_FOREVER);
  k_sem_take(&execute_ready_sem, K_FOREVER);
  k_sem_give(&u85_execute_go_sem);
  k_sem_take(&done_sem, K_FOREVER);
  k_sem_take(&execute_ready_sem, K_FOREVER);
  if (execute_status[0] != 0 || execute_status[1] != 0) {
    printk("dual-tflm: preflight failed U55=%d U85=%d\n",
           execute_status[0], execute_status[1]);
    return 1;
  }
  printk("dual-tflm: startup self-test passed; switching to camera in 5 seconds\n");
  k_msleep(5000);
  zephyr_dual_clear_display();
  use_test_image = false;
  zephyr_dual_reset_results();
  printk("dual-tflm: live dual-NPU pipeline started\n");

  constexpr size_t kWindow = 16U;
  uint32_t ru55[kWindow] = {}, ru85[kWindow] = {}, rspan[kWindow] = {}, roverlap[kWindow] = {};
  uint64_t su55 = 0, su85 = 0, sspan = 0, soverlap = 0;
  size_t ri = 0, count = 0;
  for (size_t frame = 1;; ++frame) {
    const int capture_rc = zephyr_dual_capture_preview();
    if (capture_rc != 0) {
      printk("dual-tflm: frame=%zu capture failed rc=%d\n", frame, capture_rc);
      k_msleep(100);
      continue;
    }
    k_sem_give(&u85_execute_go_sem);
    k_sem_give(&u55_execute_go_sem);
    k_sem_take(&input_copied_sem, K_FOREVER);
    k_sem_take(&input_copied_sem, K_FOREVER);
    const int release_rc = zephyr_dual_release_frame();
    if (release_rc != 0)
      printk("dual-tflm: frame=%zu release failed rc=%d\n", frame, release_rc);
    k_sem_take(&done_sem, K_FOREVER);
    k_sem_take(&done_sem, K_FOREVER);
    if (execute_status[0] != 0 || execute_status[1] != 0) {
      printk("dual-tflm: frame=%zu inference failed U55=%d U85=%d\n",
             frame, execute_status[0], execute_status[1]);
      return 1;
    }
    k_sem_take(&execute_ready_sem, K_FOREVER);
    k_sem_take(&execute_ready_sem, K_FOREVER);
    const uint64_t u55_us = k_cyc_to_us_floor64(execute_end_cycles[0] - execute_start_cycles[0]);
    const uint64_t u85_us = k_cyc_to_us_floor64(execute_end_cycles[1] - execute_start_cycles[1]);
    const uint64_t first = std::min(execute_start_cycles[0], execute_start_cycles[1]);
    const uint64_t last = std::max(execute_end_cycles[0], execute_end_cycles[1]);
    const uint64_t os = std::max(execute_start_cycles[0], execute_start_cycles[1]);
    const uint64_t oe = std::min(execute_end_cycles[0], execute_end_cycles[1]);
    const uint64_t span_us = k_cyc_to_us_floor64(last - first);
    const uint64_t overlap_us = oe > os ? k_cyc_to_us_floor64(oe - os) : 0U;
    if (count == kWindow) {
      su55 -= ru55[ri]; su85 -= ru85[ri]; sspan -= rspan[ri]; soverlap -= roverlap[ri];
    } else ++count;
    ru55[ri] = static_cast<uint32_t>(u55_us);
    ru85[ri] = static_cast<uint32_t>(u85_us);
    rspan[ri] = static_cast<uint32_t>(span_us);
    roverlap[ri] = static_cast<uint32_t>(overlap_us);
    su55 += u55_us; su85 += u85_us; sspan += span_us; soverlap += overlap_us;
    ri = (ri + 1U) % kWindow;
    zephyr_dual_show_parallel_summary(static_cast<uint32_t>(count),
        static_cast<uint32_t>(su55 / count), static_cast<uint32_t>(su85 / count),
        static_cast<uint32_t>(sspan / count), static_cast<uint32_t>(soverlap / count));
    if (frame == 1U || frame % 10U == 0U)
      printk("dual-tflm: PAR %s frame=%zu U55=%llu U85=%llu span=%llu overlap=%llu us\n",
             latest_vww_person ? "PERSON" : "NO_PERSON", frame,
             (unsigned long long)u55_us, (unsigned long long)u85_us,
             (unsigned long long)span_us, (unsigned long long)overlap_us);
    if (frame % 30U == 0U)
      printk("dual-tflm: live frame=%zu IRQs=%u/%u\n", frame,
             dual_ethosu_u55_irqs(), dual_ethosu_u85_irqs());
  }
}
