/* SPDX-License-Identifier: Apache-2.0 */

#include <stdlib.h>
#include <string.h>

#include <cmsis_core.h>
#include <ethosu_device.h>
#include <ethosu_driver.h>
#include <soc_memory_map.h>
#include <zephyr/devicetree.h>
#include <zephyr/cache.h>
#include <zephyr/irq.h>
#include <zephyr/kernel.h>

#define U55_NODE DT_NODELABEL(ethosu0)
#define U85_NODE DT_NODELABEL(ethosu1)

#define ETHOSU_COP1_FOURCC ('1' << 24 | 'P' << 16 | 'O' << 8 | 'C')
#define ETHOSU_COP_OPTIMIZER_CONFIG 1U
#define ETHOSU_COP_COMMAND_STREAM 2U
#define ETHOSU_COP_NOP 5U

static struct ethosu_driver u55_driver;
static struct ethosu_driver u85_driver;
static volatile unsigned u55_irqs;
static volatile unsigned u85_irqs;
static volatile uint64_t u55_submit_cycle;
static volatile uint64_t u85_submit_cycle;
static volatile uint64_t u55_irq_cycle;
static volatile uint64_t u85_irq_cycle;

#define U85_CMD_MIRROR_SIZE (16 * 1024)
#define U85_WEIGHT_MIRROR_SIZE (512 * 1024)
static uint8_t u85_cmd_mirror[U85_CMD_MIRROR_SIZE]
	__attribute__((section(".alif_sram1.u85_mirror"), aligned(32)));
static uint8_t u85_weight_mirror[U85_WEIGHT_MIRROR_SIZE]
	__attribute__((section(".alif_sram1.u85_mirror"), aligned(32)));
static uint8_t u55_fast_scratch[1536] __aligned(16);
static uint8_t u85_fast_scratch[1536] __aligned(16);

/* ExecuTorch's default Ethos-U I/O copy is a plain memcpy. The M55 cache is
 * not coherent with either NPU: invalidate NPU-written sources before CPU
 * reads, and clean CPU-written destinations before NPU reads. This function
 * serves both the input-to-scratch and scratch-to-output copy directions. */
void arm_ethos_io_memcpy(void *dst, const void *src, size_t size)
{
	sys_cache_data_invd_range((void *)src, size);
	memcpy(dst, src, size);
	sys_cache_data_flush_range(dst, size);
}

extern void dual_report_checkpoint(unsigned value);

unsigned char *arm_ethos_fast_scratch_for_product(uint32_t product)
{
	return product == ETHOSU_PRODUCT_U85 ? u85_fast_scratch : u55_fast_scratch;
}

int arm_ethos_prepare_invoke_buffers(uint32_t product,
				     const void **cmd_data,
				     size_t cmd_size,
				     const void **weight_data,
				     size_t weight_size)
{
	if (product != ETHOSU_PRODUCT_U85) {
		return 0;
	}
	/* Shared SRAM0/SRAM1 and MRAM/OSPI are directly visible to U85.
	 * Mirroring is only required for command and weight data linked into
	 * the M55's private ITCM.
	 */
	uintptr_t cmd = (uintptr_t)*cmd_data;
	uintptr_t weights = (uintptr_t)*weight_data;
	bool cmd_visible = (cmd >= DTCM_BASE && cmd < DTCM_BASE + DTCM_SIZE) ||
			   (cmd >= 0x02000000U && cmd < 0x02800000U);
	bool weights_visible =
		(weights >= DTCM_BASE && weights < DTCM_BASE + DTCM_SIZE) ||
		(weights >= 0x02000000U && weights < 0x02800000U);
	static unsigned invoke_count;
	if (invoke_count++ == 0U || (invoke_count % 30U) == 0U) {
		printk("dual-et: U85 invoke buffers cmd=%p/%zu weights=%p/%zu visible=%d/%d\n",
		       *cmd_data, cmd_size, *weight_data, weight_size,
		       cmd_visible, weights_visible);
	}
	if (cmd_visible && weights_visible) {
		return 0;
	}
	if (cmd_size > sizeof(u85_cmd_mirror) ||
	    weight_size > sizeof(u85_weight_mirror)) {
		return -1;
	}

	memcpy(u85_cmd_mirror, *cmd_data, cmd_size);
	memcpy(u85_weight_mirror, *weight_data, weight_size);
	*cmd_data = u85_cmd_mirror;
	*weight_data = u85_weight_mirror;
	return 0;
}

void *ethosu_mutex_create(void)
{
	struct k_mutex *m = malloc(sizeof(*m));
	if (m != NULL) {
		k_mutex_init(m);
	}
	return m;
}

void ethosu_mutex_destroy(void *mutex) { free(mutex); }
int ethosu_mutex_lock(void *mutex) { return k_mutex_lock(mutex, K_FOREVER); }
int ethosu_mutex_unlock(void *mutex) { return k_mutex_unlock(mutex); }

void *ethosu_semaphore_create(void)
{
	struct k_sem *s = malloc(sizeof(*s));
	if (s != NULL) {
		/* Core-driver main owns the semaphore policy: waiters start empty and
		 * may accumulate one token per registered NPU of the same variant.
		 */
		k_sem_init(s, 0, K_SEM_MAX_LIMIT);
	}
	return s;
}

void ethosu_semaphore_destroy(void *sem) { free(sem); }

int ethosu_semaphore_take(void *sem, uint64_t timeout)
{
	k_timeout_t t = timeout == ETHOSU_SEMAPHORE_WAIT_FOREVER ? K_FOREVER : K_TICKS(timeout);
	return k_sem_take(sem, t);
}

int ethosu_semaphore_give(void *sem)
{
	k_sem_give(sem);
	return 0;
}

static uint64_t alif_address_remap(uint64_t address, int index)
{
	ARG_UNUSED(index);
	return local_to_global((void *)(uintptr_t)address);
}

static unsigned int alif_u85_config_select(uint64_t address, int index)
{
	/* HP DTCM is exposed to the NPUs at its 0x50800000 global alias and
	 * shared SRAM0/SRAM1 occupy 0x02000000-0x027fffff. All must be reached
	 * through U85's SRAM AXI interface (MEM_ATTR 0).
	 */
	if ((address >= DTCM_GLOBAL_BASE &&
	     address < DTCM_GLOBAL_BASE + DTCM_SIZE) ||
	    (address >= 0x02000000ULL && address < 0x02800000ULL)) {
		return 0;
	}
	if (address >= 0x80000000ULL) {
		/* MRAM/OSPI is reached through U85's EXT AXI interface. */
		return 2;
	}

	/* Preserve the U85 backend defaults for non-DTCM regions. */
	if (index < 0) {
		return 2;
	}
	return index == 0 ? 3 : 1;
}

static struct ethosu_device_user_ops alif_u55_user_ops = {
	.address_remap = alif_address_remap,
	.config_select = NULL,
};

static struct ethosu_device_user_ops alif_u85_user_ops = {
	.address_remap = alif_address_remap,
	.config_select = alif_u85_config_select,
};

void ethosu_flush_dcache(const uint64_t *base, const size_t *sizes, int count)
{
	for (int i = 0; i < count; ++i) {
		SCB_CleanDCache_by_Addr((uint32_t *)(uintptr_t)base[i], sizes[i]);
	}
}

void ethosu_invalidate_dcache(const uint64_t *base, const size_t *sizes, int count)
{
	for (int i = 0; i < count; ++i) {
		SCB_InvalidateDCache_by_Addr((uint32_t *)(uintptr_t)base[i], sizes[i]);
	}
}

static void u55_isr(const void *unused)
{
	ARG_UNUSED(unused);
	u55_irq_cycle = k_cycle_get_64();
	u55_irqs++;
	ethosu_irq_handler(&u55_driver);
}

static void u85_isr(const void *unused)
{
	ARG_UNUSED(unused);
	u85_irq_cycle = k_cycle_get_64();
	u85_irqs++;
	ethosu_irq_handler(&u85_driver);
}

void ethosu_inference_begin(struct ethosu_driver *drv, void *user_arg)
{
	ARG_UNUSED(user_arg);
	if (drv == &u55_driver) {
		u55_submit_cycle = k_cycle_get_64();
	} else if (drv == &u85_driver) {
		u85_submit_cycle = k_cycle_get_64();
	}
}

int dual_ethosu_init(void)
{
	int ret;

	dual_report_checkpoint(0x201);
	ret = ethosu_init_ex(&u55_driver, &ethosu_device_desc_u55,
			     &ethosu_device_config_u55, &alif_u55_user_ops,
			     (void *)DT_REG_ADDR(U55_NODE), NULL, 0, 1, 1);
	if (ret != 0) {
		return ret;
	}

	dual_report_checkpoint(0x202);
	ret = ethosu_init_ex(&u85_driver, &ethosu_device_desc_u85,
			     &ethosu_device_config_u85, &alif_u85_user_ops,
			     (void *)DT_REG_ADDR(U85_NODE), NULL, 0, 1, 1);
	if (ret != 0) {
		return ret;
	}

	dual_report_checkpoint(0x203);
	IRQ_CONNECT(DT_IRQN(U55_NODE), 3, u55_isr, NULL, 0);
	IRQ_CONNECT(DT_IRQN(U85_NODE), 3, u85_isr, NULL, 0);
	dual_report_checkpoint(0x204);
	irq_enable(DT_IRQN(U55_NODE));
	dual_report_checkpoint(0x205);
	irq_enable(DT_IRQN(U85_NODE));
	dual_report_checkpoint(0x206);
	return 0;
}

unsigned dual_ethosu_u55_irqs(void) { return u55_irqs; }
unsigned dual_ethosu_u85_irqs(void) { return u85_irqs; }
uint64_t dual_ethosu_u55_submit_cycle(void) { return u55_submit_cycle; }
uint64_t dual_ethosu_u85_submit_cycle(void) { return u85_submit_cycle; }
uint64_t dual_ethosu_u55_irq_cycle(void) { return u55_irq_cycle; }
uint64_t dual_ethosu_u85_irq_cycle(void) { return u85_irq_cycle; }
