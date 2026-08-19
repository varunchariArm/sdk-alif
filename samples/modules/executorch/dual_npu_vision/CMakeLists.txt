cmake_minimum_required(VERSION 3.20)

set(CMAKE_SKIP_INSTALL_RULES ON CACHE BOOL "" FORCE)
set(ETHOSU_MULTI_VARIANT ON CACHE BOOL "" FORCE)
set(CMSIS_PATH "$ENV{ZEPHYR_BASE}/../modules/hal/cmsis"
    CACHE PATH "" FORCE)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(executorch_dual_npu C CXX)

set(EXECUTORCH_ROOT ${ZEPHYR_EXECUTORCH_MODULE_DIR})
set(ET_PYTHON "${Python3_EXECUTABLE}")
set(MODEL_DIR "${CMAKE_CURRENT_LIST_DIR}/models")
set(U85_PTE "${MODEL_DIR}/vww_u85_256.pte")
set(U55_PTE "${MODEL_DIR}/yolo_fastest_face_u55_256.pte")
set(FACE_BMP "${MODEL_DIR}/man_and_baby.bmp")
set(SRAM0_PAYLOAD "${CMAKE_CURRENT_BINARY_DIR}/u85_model.bin")
if(CONFIG_DUAL_ET_LIVE_VISION)
  set(SRAM0_PAYLOAD_INPUTS ${U85_PTE} ${U55_PTE} ${FACE_BMP})
else()
  set(SRAM0_PAYLOAD_INPUTS ${U85_PTE} ${U55_PTE})
endif()
add_custom_command(
  OUTPUT ${SRAM0_PAYLOAD}
  COMMAND ${ET_PYTHON}
          ${CMAKE_CURRENT_LIST_DIR}/tools/pack_sram0_payload.py
          --output ${SRAM0_PAYLOAD} ${SRAM0_PAYLOAD_INPUTS}
  DEPENDS ${SRAM0_PAYLOAD_INPUTS}
          ${CMAKE_CURRENT_LIST_DIR}/tools/pack_sram0_payload.py
  VERBATIM)
add_custom_target(gen_sram0_payload ALL DEPENDS ${SRAM0_PAYLOAD})
add_dependencies(app gen_sram0_payload)

target_sources(app PRIVATE
  src/main.cpp
  src/board_init.c
  src/zephyr_vision.c
  src/ethosu_platform.c
  ${EXECUTORCH_ROOT}/examples/arm/executor_runner/arm_memory_allocator.cpp)
target_include_directories(app PRIVATE
  src ${CMAKE_CURRENT_BINARY_DIR}
  ${EXECUTORCH_ROOT}/examples/arm/executor_runner)

target_link_libraries(app PRIVATE libexecutorch)
if(CONFIG_CPU_CORTEX_M)
  if(TARGET cortex_m_ops_lib)
    target_link_libraries(app PRIVATE cortex_m_ops_lib)
  endif()
  if(TARGET cortex_m_kernels)
    target_link_libraries(app PRIVATE cortex_m_kernels)
  endif()
endif()
if(TARGET executorch_delegate_ethos_u)
  target_link_libraries(app PRIVATE executorch_delegate_ethos_u)
endif()
if(TARGET ethosu_core_driver)
  target_link_libraries(app PRIVATE ethosu_core_driver)
  target_compile_definitions(ethosu_core_driver PRIVATE
    ETHOSU_SEMAPHORE_WAIT_INFERENCE=10000
    # Alif MLEK's E8/U85 settings: use Normal, non-cacheable,
    # non-bufferable transactions.  The core driver's default
    # Device attributes stall when U85 fetches this large model from MRAM.
    NPU_MEM_ATTR_0=0x20
    NPU_MEM_ATTR_1=0x20
    NPU_MEM_ATTR_2=0x24
    NPU_MEM_ATTR_3=0x24
    AXI_LIMIT_SRAM_MAX_BEATS=2
    AXI_LIMIT_EXT_MAX_BEATS=2
    # ensemble_vela.ini advertises the E8 MRAM channel as one outstanding
    # read and no writes.  The driver's generic 64/32 defaults can deadlock
    # the MRAM bridge as soon as the command stream requests constants.
    AXI_LIMIT_EXT_MAX_OUTSTANDING_READ=1
    AXI_LIMIT_EXT_MAX_OUTSTANDING_WRITE=1)
endif()

# The Alif callback implements the legacy single-device ABI. This application
# supplies platform hooks for the multi-variant core driver instead.
set(alif_ethosu_target "..__modules__hal__alif__drivers__ethos_u")
if(TARGET "${alif_ethosu_target}")
  get_target_property(alif_sources "${alif_ethosu_target}" SOURCES)
  list(FILTER alif_sources EXCLUDE REGEX "/ethosu_callback\\.c$")
  if(NOT alif_sources)
    set(alif_sources "${CMAKE_CURRENT_LIST_DIR}/src/ethosu_alif_abi_stub.c")
  endif()
  set_property(TARGET "${alif_ethosu_target}" PROPERTY SOURCES "${alif_sources}")
endif()
