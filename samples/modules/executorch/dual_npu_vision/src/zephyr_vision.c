/* SPDX-License-Identifier: Apache-2.0 */

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <zephyr/cache.h>
#include <zephyr/device.h>
#include <zephyr/drivers/display.h>
#include <zephyr/drivers/display/cdc200.h>
#include <zephyr/drivers/mipi_dsi/dsi_dw.h>
#include <zephyr/drivers/video.h>
#include <zephyr/drivers/video/video_alif.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#define DISPLAY_WIDTH 480U
#define DISPLAY_HEIGHT 800U
#define PREVIEW_HEIGHT 352U
#define PREVIEW_Y (((DISPLAY_HEIGHT - PREVIEW_HEIGHT) / 2U) - 40U)
#define CAMERA_OUTPUT_WIDTH 192U
#define CAMERA_OUTPUT_HEIGHT 192U
#define PREVIEW_X ((DISPLAY_WIDTH - PREVIEW_HEIGHT) / 2U)
static const struct device *display_dev;
static const struct device *camera_dev;
static uint16_t *framebuffer;
static struct video_buffer *camera_buffer;
static struct video_buffer *camera_buffers[5];
static struct video_format camera_format;
static bool camera_started;
static bool camera_streaming;

/* Callback required by Alif's ISP wrapper. Keep the vendor library quiet;
 * Zephyr-side failures are still reported by the video driver. */
int log_level(void) { return 0; }

static int start_camera(void)
{
	struct video_caps caps;
	int rc;

	if (camera_started) return 0;
	/* Match the updated Alif reference pipeline: RAW10 from MT9M114 is
	 * demosaiced by the hardware ISP into planar RGB888. */
	camera_dev = DEVICE_DT_GET_ONE(vsi_isp_pico);
	if (!device_is_ready(camera_dev)) return -ENODEV;

	memset(&caps, 0, sizeof(caps));
	rc = video_get_caps(camera_dev, VIDEO_EP_IN, &caps);
	if (rc != 0) return rc;
	for (const struct video_format_cap *cap = caps.format_caps;
	     cap != NULL && cap->pixelformat != 0; ++cap) {
		if (cap->pixelformat == VIDEO_PIX_FMT_Y10P) {
			/* Match the PR reference: keep walking so the last Y10P
			 * capability (1288x728 for MT9M114) is selected.  The
			 * sensor-specific crop-x0=280 is invalid for 640x480. */
			camera_format.pixelformat = VIDEO_PIX_FMT_Y10P;
			camera_format.width = cap->width_min;
			camera_format.height = cap->height_min;
			camera_format.pitch = cap->width_min * 2U;
		}
	}
	if (camera_format.pixelformat == 0) return -ENOTSUP;
	printk("dual-et: ISP input selected %ux%u pitch=%u\n",
	       camera_format.width, camera_format.height, camera_format.pitch);
	rc = video_set_format(camera_dev, VIDEO_EP_IN, &camera_format);
	if (rc != 0) return rc;
	camera_format.pixelformat = VIDEO_PIX_FMT_RGB888_PLANAR_PRIVATE;
	/* The MT9M114-specific ISP overlay selects a centered 728x728 ROI.
	 * Keep the ISP output square as well; requesting 480x272 from that ROI is
	 * rejected by the updated ISP library and previously exposed invalid edge
	 * pixels as noise. */
	camera_format.width = CAMERA_OUTPUT_WIDTH;
	camera_format.height = CAMERA_OUTPUT_HEIGHT;
	camera_format.pitch = camera_format.width * 3U;
	rc = video_set_format(camera_dev, VIDEO_EP_OUT, &camera_format);
	if (rc != 0) return rc;
	const size_t capture_size = (size_t)camera_format.pitch * camera_format.height;

	for (size_t i = 0; i < ARRAY_SIZE(camera_buffers); ++i) {
		camera_buffers[i] = video_buffer_alloc(capture_size, K_NO_WAIT);
		if (camera_buffers[i] == NULL) {
			printk("dual-et: camera buffer %zu allocation failed size=%zu\n",
			       i, capture_size);
			return -ENOMEM;
		}
		rc = video_enqueue(camera_dev, VIDEO_EP_OUT, camera_buffers[i]);
		if (rc != 0) {
			printk("dual-et: camera buffer %zu enqueue failed rc=%d\n", i, rc);
			return rc;
		}
	}
	camera_started = true;
	printk("dual-et: Zephyr MT9M114 ready %ux%u pitch=%u buffers=%zu first=%p/%u\n",
	       camera_format.width, camera_format.height, camera_format.pitch,
	       ARRAY_SIZE(camera_buffers), camera_buffers[0]->buffer,
	       (unsigned)camera_buffers[0]->size);
	return 0;
}

int zephyr_dual_display_init(void)
{
	const struct device *panel = DEVICE_DT_GET(DT_ALIAS(panel));
	const struct device *dsi = DEVICE_DT_GET(DT_ALIAS(mipi_dsi));
	struct cdc200_fb_desc fb = {0};
	int rc;

	display_dev = DEVICE_DT_GET(DT_CHOSEN(zephyr_display));
	if (!device_is_ready(panel) || !device_is_ready(dsi) ||
	    !device_is_ready(display_dev)) return -ENODEV;
	rc = dsi_dw_set_mode(dsi, DSI_DW_VIDEO_MODE);
	if (rc == 0) rc = display_blanking_off(panel);
	if (rc != 0) return rc;
	cdc200_set_enable(display_dev, true);
	cdc200_get_framebuffer(display_dev, 0, &fb);
	if (fb.fb_addr == NULL || fb.fb_size < DISPLAY_WIDTH * DISPLAY_HEIGHT * 2U)
		return -ENOMEM;
	framebuffer = (uint16_t *)fb.fb_addr;
	memset(framebuffer, 0, DISPLAY_WIDTH * DISPLAY_HEIGHT * 2U);
	sys_cache_data_flush_range(framebuffer, DISPLAY_WIDTH * DISPLAY_HEIGHT * 2U);
	printk("dual-et: Zephyr MW405 ready fb=%p/%zu\n", fb.fb_addr, fb.fb_size);
	rc = start_camera();
	if (rc != 0) printk("dual-et: camera initialization failed rc=%d\n", rc);
	return rc;
}

uint16_t *zephyr_dual_preview_buffer(void)
{
	return framebuffer ? framebuffer + PREVIEW_Y * DISPLAY_WIDTH : NULL;
}

int zephyr_dual_capture_preview(void)
{
	struct video_buffer *captured = NULL;
	int rc = start_camera();
	if (rc != 0) return rc;

	/* Return the frame held stable during the previous inference, while four
	 * other buffers kept the continuously streaming CSI/CPI pipeline fed. */
	if (camera_buffer != NULL) {
		rc = video_enqueue(camera_dev, VIDEO_EP_OUT, camera_buffer);
		if (rc != 0) return rc;
		camera_buffer = NULL;
	}
	if (!camera_streaming) {
		rc = video_stream_start(camera_dev);
		if (rc != 0 && rc != -EBUSY) return rc;
		camera_streaming = true;
	}
	/* The Alif CPI driver can transiently return EAGAIN at a frame boundary
	 * even with a blocking timeout. Preserve the queued-buffer state and wait
	 * briefly for the next completed buffer instead of dropping the frame. */
	for (unsigned retry = 0; retry < 50U; ++retry) {
		rc = video_dequeue(camera_dev, VIDEO_EP_OUT, &captured, K_MSEC(100));
		if (rc != -EAGAIN) break;
	}
	if (rc != 0 || captured == NULL) return rc != 0 ? rc : -EIO;
	/* Inference is slower than the camera. Drain all already-completed frames,
	 * requeue stale ones, and retain only the newest frame. Without this step
	 * one completed buffer accumulates per iteration until the pool empties. */
	for (;;) {
		struct video_buffer *newer = NULL;
		rc = video_dequeue(camera_dev, VIDEO_EP_OUT, &newer, K_NO_WAIT);
		if (rc == -EAGAIN) break;
		if (rc != 0 || newer == NULL) return rc != 0 ? rc : -EIO;
		rc = video_enqueue(camera_dev, VIDEO_EP_OUT, captured);
		if (rc != 0) return rc;
		captured = newer;
	}
	camera_buffer = captured;
	sys_cache_data_invd_range(camera_buffer->buffer,
		(size_t)camera_format.pitch * camera_format.height);

	const size_t plane_size = (size_t)camera_format.width * camera_format.height;
	const uint8_t *red = camera_buffer->buffer;
	const uint8_t *green = red + plane_size;
	const uint8_t *blue = green + plane_size;
	for (uint32_t y = 0; y < PREVIEW_HEIGHT; ++y) {
		const uint32_t sy = y * camera_format.height / PREVIEW_HEIGHT;
		uint16_t *dst = framebuffer + (PREVIEW_Y + y) * DISPLAY_WIDTH;
		memset(dst, 0, DISPLAY_WIDTH * sizeof(*dst));
		for (uint32_t x = 0; x < PREVIEW_HEIGHT; ++x) {
			const uint32_t sx = x * camera_format.width / PREVIEW_HEIGHT;
			const size_t i = (size_t)sy * camera_format.width + sx;
			dst[PREVIEW_X + x] = ((uint16_t)(red[i] >> 3) << 11) |
				 ((uint16_t)(green[i] >> 2) << 5) | (blue[i] >> 3);
		}
	}
	sys_cache_data_flush_range(framebuffer + PREVIEW_Y * DISPLAY_WIDTH,
		PREVIEW_HEIGHT * DISPLAY_WIDTH * 2U);
	return 0;
}

int zephyr_dual_release_frame(void)
{
	if (camera_buffer == NULL) return 0;
	int rc = video_enqueue(camera_dev, VIDEO_EP_OUT, camera_buffer);
	if (rc == 0) camera_buffer = NULL;
	return rc;
}

void zephyr_dual_get_rgb(uint32_t x, uint32_t y,
			 uint8_t *red, uint8_t *green, uint8_t *blue)
{
	/* x is expressed in the logical 480x272 preview coordinate system used by
	 * both models. Scale it onto the centered square camera image. */
	const uint32_t sx = x * camera_format.width / DISPLAY_WIDTH;
	const uint32_t sy = y * camera_format.height / PREVIEW_HEIGHT;
	const size_t plane_size = (size_t)camera_format.width * camera_format.height;
	const size_t i = (size_t)sy * camera_format.width + sx;
	const uint8_t *planes = camera_buffer->buffer;
	*red = planes[i];
	*green = planes[plane_size + i];
	*blue = planes[2U * plane_size + i];
}

static void reload_display(void)
{
	sys_cache_data_flush_range(framebuffer, DISPLAY_WIDTH * DISPLAY_HEIGHT * 2U);
}

static void summary_text(uint32_t x, uint32_t y, const char *text, uint16_t color);
void zephyr_dual_clear_display(void)
{
	memset(framebuffer, 0, DISPLAY_WIDTH * DISPLAY_HEIGHT * 2U);
	reload_display();
}

static const uint8_t *summary_glyph(char c)
{
	static const uint8_t digits[10][7] = {
		{14,17,19,21,25,17,14},{4,12,4,4,4,4,14},{14,17,1,2,4,8,31},
		{30,1,1,14,1,1,30},{2,6,10,18,31,2,2},{31,16,16,30,1,1,30},
		{14,16,16,30,17,17,14},{31,1,2,4,8,8,8},{14,17,17,14,17,17,14},
		{14,17,17,15,1,1,14}};
	static const uint8_t letters[26][7] = {
		{14,17,17,31,17,17,17},{30,17,17,30,17,17,30},{14,17,16,16,16,17,14},
		{30,17,17,17,17,17,30},{31,16,16,30,16,16,31},{31,16,16,30,16,16,16},
		{14,17,16,23,17,17,15},{17,17,17,31,17,17,17},{14,4,4,4,4,4,14},
		{1,1,1,1,17,17,14},{17,18,20,24,20,18,17},{16,16,16,16,16,16,31},
		{17,27,21,21,17,17,17},{17,25,25,21,19,19,17},{14,17,17,17,17,17,14},
		{30,17,17,30,16,16,16},{14,17,17,17,21,18,13},{30,17,17,30,20,18,17},
		{15,16,16,14,1,1,30},{31,4,4,4,4,4,4},{17,17,17,17,17,17,14},
		{17,17,17,17,17,10,4},{17,17,17,21,21,21,10},{17,17,10,4,10,17,17},
		{17,17,10,4,4,4,4},{31,1,2,4,8,16,31}};
	static const uint8_t blank[7]={0};
	if (c >= '0' && c <= '9') return digits[c - '0'];
	if (c >= 'A' && c <= 'Z') return letters[c - 'A'];
	return blank;
}

void zephyr_dual_show_ssd_result(int faces)
{
	char line[24];
	for (uint32_t y = 12U; y < 38U; ++y)
		memset(framebuffer + y * DISPLAY_WIDTH, 0, DISPLAY_WIDTH * sizeof(*framebuffer));
	snprintf(line, sizeof(line), "U55 SSD F %d", faces);
	summary_text(18U, 14U, line, faces > 0 ? 0x07e0 : 0xffff);
	reload_display();
}

void zephyr_dual_show_classification(int class_id, int confidence, const char *label)
{
	char line[28];
	for (uint32_t y = 42U; y < 70U; ++y)
		memset(framebuffer + y * DISPLAY_WIDTH, 0, DISPLAY_WIDTH * sizeof(*framebuffer));
	snprintf(line, sizeof(line), "U85 %s %d", label && label[0] ? label : "CLASS", confidence);
	summary_text(18U, 44U, line, class_id >= 0 ? 0x07ff : 0xffff);
	reload_display();
}

void zephyr_dual_reset_results(void)
{
	zephyr_dual_show_ssd_result(0);
	zephyr_dual_show_classification(-1, 0, "CLASS");
}

static void summary_text(uint32_t x, uint32_t y, const char *text, uint16_t color)
{
	const uint32_t scale = 3U;
	for (; *text; ++text, x += 6U * scale) {
		const uint8_t *glyph = summary_glyph(*text);
		for (uint32_t row = 0; row < 7U; ++row)
			for (uint32_t col = 0; col < 5U; ++col)
				if (glyph[row] & BIT(4U-col))
					for (uint32_t dy=0; dy<scale; ++dy)
						for (uint32_t dx=0; dx<scale; ++dx)
							framebuffer[(y+row*scale+dy)*DISPLAY_WIDTH+x+col*scale+dx]=color;
	}
}

void zephyr_dual_show_parallel_summary(uint32_t samples, uint32_t u55_us,
	uint32_t u85_us, uint32_t span_us, uint32_t overlap_us)
{
	char line[24]; const uint32_t top=575U;
	for (uint32_t y=top; y<735U; ++y)
		for (uint32_t x=16U; x<464U; ++x) framebuffer[y*DISPLAY_WIDTH+x]=0;
	summary_text(24U,top,"PAR AVG US",0xffe0);
	snprintf(line,sizeof(line),"U55 %lu",(unsigned long)u55_us); summary_text(24U,top+28U,line,0xffff);
	snprintf(line,sizeof(line),"U85 %lu",(unsigned long)u85_us); summary_text(24U,top+56U,line,0xffff);
	snprintf(line,sizeof(line),"SPAN %lu",(unsigned long)span_us); summary_text(24U,top+84U,line,0x07ff);
	snprintf(line,sizeof(line),"OVLP %lu",(unsigned long)overlap_us); summary_text(24U,top+112U,line,overlap_us?0x07e0:0xf800);
	snprintf(line,sizeof(line),"N %lu",(unsigned long)samples); summary_text(300U,top,line,0xffff);
	reload_display();
}

void zephyr_dual_show_bmp(const uint8_t *bmp)
{
	const uint8_t *pixels=bmp+54U; const uint32_t size=PREVIEW_HEIGHT;
	const uint32_t xoff=(DISPLAY_WIDTH-size)/2U;
	for (uint32_t y=0; y<PREVIEW_HEIGHT; ++y) {
		memset(framebuffer+(PREVIEW_Y+y)*DISPLAY_WIDTH,0,DISPLAY_WIDTH*2U);
		for (uint32_t x=0; x<size; ++x) {
			uint32_t sx=x*192U/size, sy=191U-y*192U/size;
			const uint8_t *bgr=pixels+(sy*192U+sx)*3U;
			framebuffer[(PREVIEW_Y+y)*DISPLAY_WIDTH+xoff+x]=
				((uint16_t)(bgr[2]>>3)<<11)|((uint16_t)(bgr[1]>>2)<<5)|(bgr[0]>>3);
		}
	}
	reload_display();
}
