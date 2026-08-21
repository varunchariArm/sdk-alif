cmake_minimum_required(VERSION 3.20)

set(CMAKE_SKIP_INSTALL_RULES ON CACHE BOOL "" FORCE)
set(ETHOSU_MULTI_VARIANT ON CACHE BOOL "" FORCE)
set(CMSIS_PATH "$ENV{ZEPHYR_BASE}/../modules/hal/cmsis"
    CACHE PATH "" FORCE)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(tflm_dual_npu C CXX)

set(MODEL_DIR "${CMAKE_CURRENT_LIST_DIR}/models")
set(U85_MODEL "${MODEL_DIR}/vww_u85_256.tflite")
set(U55_MODEL "${MODEL_DIR}/yolo_fastest_face_u55_256.tflite")
set(FACE_BMP "${MODEL_DIR}/man_and_baby.bmp")
set(SRAM0_PAYLOAD "${CMAKE_CURRENT_BINARY_DIR}/u85_model.bin")

file(SIZE "${U85_MODEL}" U85_MODEL_SIZE)
file(SIZE "${U55_MODEL}" U55_MODEL_SIZE)
file(SIZE "${FACE_BMP}" FACE_BMP_SIZE)
configure_file(src/model_layout.h.in
               "${CMAKE_CURRENT_BINARY_DIR}/model_layout.h" @ONLY)

add_custom_command(
  OUTPUT "${SRAM0_PAYLOAD}"
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_LIST_DIR}/tools/pack_sram0_payload.py"
          --output "${SRAM0_PAYLOAD}" "${U85_MODEL}" "${U55_MODEL}" "${FACE_BMP}"
  DEPENDS "${U85_MODEL}" "${U55_MODEL}" "${FACE_BMP}"
          "${CMAKE_CURRENT_LIST_DIR}/tools/pack_sram0_payload.py"
  VERBATIM)
add_custom_target(gen_sram0_payload ALL DEPENDS "${SRAM0_PAYLOAD}")
add_dependencies(app gen_sram0_payload)

target_sources(app PRIVATE
  src/main.cpp
  src/board_init.c
  src/zephyr_vision.c
  src/ethosu_platform.c)
target_include_directories(app PRIVATE src "${CMAKE_CURRENT_BINARY_DIR}")

if(TARGET ethosu_core_driver)
  target_link_libraries(app PRIVATE ethosu_core_driver)
  target_compile_definitions(ethosu_core_driver PRIVATE
    ETHOSU_SEMAPHORE_WAIT_INFERENCE=10000
    NPU_MEM_ATTR_0=0x20
    NPU_MEM_ATTR_1=0x20
    NPU_MEM_ATTR_2=0x24
    NPU_MEM_ATTR_3=0x24
    AXI_LIMIT_SRAM_MAX_BEATS=2
    AXI_LIMIT_EXT_MAX_BEATS=2
    AXI_LIMIT_EXT_MAX_OUTSTANDING_READ=1
    AXI_LIMIT_EXT_MAX_OUTSTANDING_WRITE=1)
endif()

# The Alif callback implements the legacy single-device ABI. This application
# supplies the platform hooks and both NPU instances to the multi-variant
# upstream Ethos-U core driver.
set(alif_ethosu_target "..__modules__hal__alif__drivers__ethos_u")
if(TARGET "${alif_ethosu_target}")
  get_target_property(alif_sources "${alif_ethosu_target}" SOURCES)
  list(FILTER alif_sources EXCLUDE REGEX "/ethosu_callback\\.c$")
  if(NOT alif_sources)
    set(alif_sources "${CMAKE_CURRENT_LIST_DIR}/src/ethosu_alif_abi_stub.c")
  endif()
  set_property(TARGET "${alif_ethosu_target}" PROPERTY SOURCES "${alif_sources}")
endif()
