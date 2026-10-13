/*
 * nvscan-display: show a guest GPU's scanout, published into shared memory,
 * as a QEMU console.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * A guest framebuffer driver (the NullMoth NVRMFB "nvscan" bridge) copies
 * every frame it flips into an ivshmem BAR: a one-page header, then two
 * frame slots. This device reads the same memory backend from the host side
 * and presents the newest complete frame on its own graphic console, so the
 * passed-through GPU's output appears in QEMU's window.
 *
 * Until the guest publishes frames (firmware, boot picker, early boot) the
 * console mirrors a fallback console (the VGA device by default). A system
 * reset clears the header, so a reboot falls back again instead of freezing
 * on the last frame.
 *
 *   -object memory-backend-file,id=nvscan,size=128M,mem-path=/dev/shm/nvscanout,share=on
 *   -device nvscan-display,memdev=nvscan,fallback=vga     (before the VGA device: console 0)
 *   -device ivshmem-plain,memdev=nvscan
 */
#include "qemu/osdep.h"
#include "qapi/error.h"
#include "qemu/module.h"
#include "qemu/atomic.h"
#include "hw/core/qdev-properties.h"
#include "system/hostmem.h"
#include "system/reset.h"
#include "ui/console.h"
#include "qom/object.h"

/* The shared-memory layout (version 1). */
#define NVSCAN_MAGIC        0x4e565343414e3031ULL   /* "NVSCAN01" */
#define NVSCAN_VERSION      1u
#define NVSCAN_X8R8G8B8     1u
#define NVSCAN_MAX_SLOTS    2u

typedef struct NVScanHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t header_bytes;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t pixel_format;
    uint32_t slot_bytes;
    uint32_t slot_count;
    uint32_t write_slot;
    uint32_t flags;
    uint64_t frame_sequence;
    uint64_t frame_time_ns;
    uint64_t dropped_frames;
} NVScanHeader;

#define TYPE_NVSCAN_DISPLAY "nvscan-display"
OBJECT_DECLARE_SIMPLE_TYPE(NVScanDisplay, NVSCAN_DISPLAY)

struct NVScanDisplay {
    DeviceState parent_obj;

    HostMemoryBackend *hostmem;
    char *fallback;             /* device id of the console shown before frames arrive */

    QemuConsole *con;
    QemuConsole *fallback_con;
    uint8_t *base;
    uint64_t size;
    uint64_t shown_sequence;    /* last frame put on the console; 0 = none */
    bool showing_frames;
};

/* Make the console's surface width x height, x8r8g8b8. */
static DisplaySurface *nvscan_surface(NVScanDisplay *s, int width, int height)
{
    DisplaySurface *ds = qemu_console_surface(s->con);

    if (!ds || surface_width(ds) != width || surface_height(ds) != height ||
        surface_format(ds) != PIXMAN_x8r8g8b8) {
        ds = qemu_create_displaysurface(width, height);
        dpy_gfx_replace_surface(s->con, ds);
    }
    return ds;
}

/* Copy the newest frame if there is one. Returns whether frames are live. */
static bool nvscan_show_frame(NVScanDisplay *s)
{
    NVScanHeader *h = (NVScanHeader *)s->base;

    if (qatomic_read(&h->magic) != NVSCAN_MAGIC || h->version != NVSCAN_VERSION) {
        return false;
    }
    uint64_t seq = qatomic_load_acquire(&h->frame_sequence);
    if (seq == 0) {
        return false;
    }
    if (seq == s->shown_sequence) {
        return true;
    }

    uint32_t slot = h->write_slot, width = h->width, height = h->height;
    uint32_t stride = h->stride, slot_bytes = h->slot_bytes;
    uint64_t offset = (uint64_t)h->header_bytes + (uint64_t)slot * slot_bytes;
    if (h->pixel_format != NVSCAN_X8R8G8B8 || slot >= NVSCAN_MAX_SLOTS ||
        width == 0 || height == 0 || width > 16384 || height > 16384 ||
        stride < width * 4u || (uint64_t)stride * height > slot_bytes ||
        offset + (uint64_t)stride * height > s->size) {
        return false;
    }

    DisplaySurface *ds = nvscan_surface(s, width, height);
    const uint8_t *src = s->base + offset;
    uint8_t *dst = surface_data(ds);
    int dst_stride = surface_stride(ds);
    for (uint32_t y = 0; y < height; y++) {
        memcpy(dst + (size_t)y * dst_stride, src + (size_t)y * stride, width * 4u);
    }

    /*
     * The writer fills the other slot for frame seq+1 and only then returns
     * to this one, after publishing seq+1. If seq moved during the copy this
     * slot may be half overwritten: keep the copy on screen anyway (at worst
     * one torn frame) but take the next frame on the following refresh.
     */
    smp_rmb();
    if (qatomic_read(&h->frame_sequence) == seq) {
        s->shown_sequence = seq;
    }
    dpy_gfx_update_full(s->con);
    return true;
}

/* Before frames arrive: mirror the fallback console. */
static void nvscan_show_fallback(NVScanDisplay *s)
{
    if (!s->fallback_con && s->fallback && *s->fallback) {
        s->fallback_con = qemu_console_lookup_by_device_name(s->fallback, 0, NULL);
    }
    if (!s->fallback_con) {
        return;
    }
    graphic_hw_update(s->fallback_con);
    DisplaySurface *src = qemu_console_surface(s->fallback_con);
    if (!src) {
        return;
    }
    DisplaySurface *ds = nvscan_surface(s, surface_width(src), surface_height(src));
    pixman_image_composite(PIXMAN_OP_SRC, src->image, NULL, ds->image,
                           0, 0, 0, 0, 0, 0, surface_width(src), surface_height(src));
    dpy_gfx_update_full(s->con);
}

static void nvscan_gfx_update(void *opaque)
{
    NVScanDisplay *s = opaque;
    bool live = nvscan_show_frame(s);

    if (live != s->showing_frames) {
        s->showing_frames = live;
        s->shown_sequence = 0;
    }
    if (!live) {
        nvscan_show_fallback(s);
    }
}

static void nvscan_invalidate(void *opaque)
{
    NVScanDisplay *s = opaque;

    s->shown_sequence = 0;
}

static const GraphicHwOps nvscan_ops = {
    .invalidate = nvscan_invalidate,
    .gfx_update = nvscan_gfx_update,
};

/* The guest driver rewrites the header when it starts again. */
static void nvscan_reset(void *opaque)
{
    NVScanDisplay *s = opaque;
    NVScanHeader *h = (NVScanHeader *)s->base;

    qatomic_set(&h->magic, 0);
    qatomic_set(&h->frame_sequence, 0);
    s->shown_sequence = 0;
}

static void nvscan_realize(DeviceState *dev, Error **errp)
{
    NVScanDisplay *s = NVSCAN_DISPLAY(dev);

    if (!s->hostmem) {
        error_setg(errp, "nvscan-display: 'memdev' is required");
        return;
    }
    MemoryRegion *mr = host_memory_backend_get_memory(s->hostmem);
    s->base = memory_region_get_ram_ptr(mr);
    s->size = memory_region_size(mr);
    if (!s->base || s->size < sizeof(NVScanHeader)) {
        error_setg(errp, "nvscan-display: memdev is not host memory of at least one page");
        return;
    }
    s->con = graphic_console_init(dev, 0, &nvscan_ops, s);
    qemu_register_reset(nvscan_reset, s);
}

static void nvscan_unrealize(DeviceState *dev)
{
    NVScanDisplay *s = NVSCAN_DISPLAY(dev);

    qemu_unregister_reset(nvscan_reset, s);
    graphic_console_close(s->con);
}

static const Property nvscan_properties[] = {
    DEFINE_PROP_LINK("memdev", NVScanDisplay, hostmem, TYPE_MEMORY_BACKEND,
                     HostMemoryBackend *),
    DEFINE_PROP_STRING("fallback", NVScanDisplay, fallback),
};

static void nvscan_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);

    set_bit(DEVICE_CATEGORY_DISPLAY, dc->categories);
    dc->desc = "guest scanout published in shared memory, as a console";
    dc->realize = nvscan_realize;
    dc->unrealize = nvscan_unrealize;
    dc->hotpluggable = false;
    device_class_set_props(dc, nvscan_properties);
}

static const TypeInfo nvscan_info = {
    .name          = TYPE_NVSCAN_DISPLAY,
    .parent        = TYPE_DEVICE,
    .instance_size = sizeof(NVScanDisplay),
    .class_init    = nvscan_class_init,
};

static void nvscan_register_types(void)
{
    type_register_static(&nvscan_info);
}

type_init(nvscan_register_types)
