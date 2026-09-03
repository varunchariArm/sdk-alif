/* SPDX-License-Identifier: Apache-2.0 */

#include <stdlib.h>
#include <cmsis_core.h>
#include <ethosu_device.h>
#include <ethosu_driver.h>
#include <pmu_ethosu.h>
#include <soc_memory_map.h>
#include <zephyr/devicetree.h>
#include <zephyr/cache.h>
#include <zephyr/irq.h>
#include <zephyr/kernel.h>

#define U55_NODE DT_NODELABEL(ethosu0)
#define U85_NODE DT_NODELABEL(ethosu1)

static struct ethosu_driver u55_driver;
static struct ethosu_driver u85_driver;
static volatile unsigned u55_irqs;
static volatile unsigned u85_irqs;
static volatile uint64_t u55_pmu_cycles;
static volatile uint64_t u85_pmu_cycles;

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
	u55_irqs++;
	ethosu_irq_handler(&u55_driver);
}

static void u85_isr(const void *unused)
{
	ARG_UNUSED(unused);
	u85_irqs++;
	ethosu_irq_handler(&u85_driver);
}

void ethosu_inference_begin(struct ethosu_driver *drv, void *user_arg)
{
	ARG_UNUSED(user_arg);
	ETHOSU_PMU_Enable(drv);
	ETHOSU_PMU_PMCCNTR_CFG_Set_Start_Event(drv, ETHOSU_PMU_NPU_ACTIVE);
	ETHOSU_PMU_PMCCNTR_CFG_Set_Stop_Event(drv, ETHOSU_PMU_NPU_IDLE);
	ETHOSU_PMU_CNTR_Enable(drv, ETHOSU_PMU_CCNT_Msk);
	ETHOSU_PMU_CYCCNT_Reset(drv);
}

void ethosu_inference_end(struct ethosu_driver *drv, void *user_arg)
{
	ARG_UNUSED(user_arg);
	uint64_t cycles = ETHOSU_PMU_Get_CCNTR(drv);
	ETHOSU_PMU_CNTR_Disable(drv, ETHOSU_PMU_CCNT_Msk);
	ETHOSU_PMU_Disable(drv);
	if (drv == &u55_driver) {
		u55_pmu_cycles = cycles;
	} else if (drv == &u85_driver) {
		u85_pmu_cycles = cycles;
	}
}

int dual_ethosu_init(void)
{
	int ret;

	ret = ethosu_init_ex(&u55_driver, &ethosu_device_desc_u55,
			     &ethosu_device_config_u55, &alif_u55_user_ops,
			     (void *)DT_REG_ADDR(U55_NODE), NULL, 0, 1, 1);
	if (ret != 0) {
		return ret;
	}

	ret = ethosu_init_ex(&u85_driver, &ethosu_device_desc_u85,
			     &ethosu_device_config_u85, &alif_u85_user_ops,
			     (void *)DT_REG_ADDR(U85_NODE), NULL, 0, 1, 1);
	if (ret != 0) {
		return ret;
	}

	IRQ_CONNECT(DT_IRQN(U55_NODE), 3, u55_isr, NULL, 0);
	IRQ_CONNECT(DT_IRQN(U85_NODE), 3, u85_isr, NULL, 0);
	irq_enable(DT_IRQN(U55_NODE));
	irq_enable(DT_IRQN(U85_NODE));
	return 0;
}

unsigned dual_ethosu_u55_irqs(void) { return u55_irqs; }
unsigned dual_ethosu_u85_irqs(void) { return u85_irqs; }
uint64_t dual_ethosu_u55_pmu_cycles(void) { return u55_pmu_cycles; }
uint64_t dual_ethosu_u85_pmu_cycles(void) { return u85_pmu_cycles; }
