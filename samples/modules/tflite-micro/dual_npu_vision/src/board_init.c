/* SPDX-License-Identifier: Apache-2.0 */

#include <zephyr/device.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/init.h>
#include <zephyr/sys/__assert.h>
#include <zephyr/sys/sys_io.h>

#include <se_service.h>
#include <soc_common.h>

/*
 * Native E8 camera/display setup from the Alif SDK main branch. This must run directly
 * after the SE service (priority 45) and before the camera/DSI devices.
 * Keep both NPU clocks enabled for this dual-NPU variant.
 */
static int dual_native_set_parameters(void)
{
	run_profile_t runp = {0};
	int rc;

#if DT_NODE_HAS_STATUS(DT_NODELABEL(camera_select), okay)
	const struct gpio_dt_spec sel =
		GPIO_DT_SPEC_GET(DT_NODELABEL(camera_select), select_gpios);
	(void)gpio_pin_configure_dt(&sel, GPIO_OUTPUT);
	(void)gpio_pin_set_dt(&sel, 1);
#endif

#if defined(CONFIG_ENSEMBLE_GEN2) && defined(CONFIG_MIPI_DSI)
	const struct gpio_dt_spec mux =
		GPIO_DT_SPEC_GET(DT_NODELABEL(mipi_dsi), cam_disp_mux_gpios);
	(void)gpio_pin_configure_dt(&mux, GPIO_OUTPUT_ACTIVE);
#endif

	runp.power_domains = PD_SYST_MASK | PD_SSE700_AON_MASK | PD_DBSS_MASK;
	runp.dcdc_voltage = 825;
	runp.dcdc_mode = DCDC_MODE_PWM;
	runp.aon_clk_src = CLK_SRC_LFXO;
	runp.run_clk_src = CLK_SRC_PLL;
	runp.vdd_ioflex_3V3 = IOFLEX_LEVEL_1V8;
	runp.cpu_clk_freq = CLOCK_FREQUENCY_400MHZ;
	runp.memory_blocks = MRAM_MASK;
#if DT_NODE_EXISTS(DT_NODELABEL(sram0))
	runp.memory_blocks |= SRAM0_MASK;
#endif
#if DT_NODE_EXISTS(DT_NODELABEL(sram1))
	runp.memory_blocks |= SRAM1_MASK;
#endif
	runp.phy_pwr_gating = MIPI_TX_DPHY_MASK | MIPI_RX_DPHY_MASK |
		MIPI_PLL_DPHY_MASK | LDO_PHY_MASK;
	runp.ip_clock_gating = CDC200_MASK | CAMERA_MASK | MIPI_CSI_MASK |
		MIPI_DSI_MASK | NPU_HP_MASK | NPU_HE_MASK;

	rc = se_service_set_run_cfg(&runp);
	__ASSERT(rc == 0, "SE: set_run_cfg failed = %d", rc);
	if (rc != 0) {
		return rc;
	}

	sys_write32(0x140001, CLKCTRL_PER_MST_CAMERA_PIXCLK_CTRL);
	return 0;
}

SYS_INIT(dual_native_set_parameters, PRE_KERNEL_1, 46);
