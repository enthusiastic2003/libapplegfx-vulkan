/*
 * libapplegfx-vulkan — host timer callbacks (Phase 1.A online event)
 * src/timer/timer.c
 *
 * Copyright © 2026 Matthew Jackson
 * SPDX-License-Identifier: MIT
 */

#include "timer.h"
#include "libapplegfx-vulkan.h"
#include "../common/log.h"
#ifdef LAGFX_HAVE_VULKAN
#include "../device.h"
#include "../display.h"
#include "../protocol/state.h"
#include "../vulkan/iosurface.h"
#endif
#include <stdatomic.h>
#include <stdlib.h>

#define LAGFX_MAX_DISPLAYS 8u

/* Called by QEMU at ~60Hz from a host timer.
 *
 * Implements the critical deadlock fix (2026-05-03): fire online IRQ
 * IMMEDIATELY when guest enables display (ss[+0x104] == 0xC). Previous
 * attempts to defer or threshold-count led to WindowServer timeouts.
 */
bool lagfx_timer_tick_vblank(
    lagfx_device_t *dev,
    void *shell_opaque,
    bool (*write_memory)(void *, uint64_t, uint64_t, const void *),
    bool (*read_memory)(void *, uint64_t, uint64_t, void *)) {

    if (!dev || !write_memory || !read_memory) {
        return false;
    }

#ifdef LAGFX_HAVE_VULKAN
    /* LAGFX_DUMP_PASSES: dump every per-pass IOSurface once, ~12s in — well after
     * the composite draw burst has created + filled the per-pass views. */
    if (getenv("LAGFX_DUMP_PASSES")) {
        extern void lagfx_dump_all_passes(lagfx_protocol_t *, lagfx_display_t *);
        static int done = 0; static uint32_t settle = 0;
        lagfx_protocol_t *p = (lagfx_protocol_t *)dev->protocol_state;
        if (!done && p) {
            /* Content-driven: the composite burst creates a 1280x1024 per-pass
             * view. Wait until one exists, then let it settle a few frames so
             * the whole burst finishes, then dump every surface once. */
            int have_view = 0;
            for (uint32_t r = 0; r < p->resources.count; r++) {
                lagfx_vk_iosurface_t *io =
                    (lagfx_vk_iosurface_t *)p->resources.entries[r].host_handle;
                if (io && io->image != VK_NULL_HANDLE
                    && io->width == 1280u && io->height == 1024u) { have_view = 1; break; }
            }
            if (have_view && ++settle >= 120u) {
                done = 1;
                lagfx_dump_all_passes(p, dev->displays[0]);
            }
        }
    }
#endif

    bool irq_raised = false;

    /* Check each display for enable() completion (ss[+0x104] == 0xC).
     *
     * DEADLOCK FIX (2026-05-03): Previous attempts deferred or threshold-counted
     * online events, all hitting chicken-and-egg problems:
     *   - defer 5s → WindowServer times out waiting for display
     *   - wait 5 submits → same timeout issue
     *   - threshold counting → still blocks on lock ordering
     *
     * Correct solution: fire IRQ IMMEDIATELY when ss[+0x104]==0xC. This breaks
     * the ABBA cycle because WindowServer's doSetDisplayMode() can proceed
     * without waiting for DisplayPipe to complete its lock acquisition.
     */
    for (unsigned i = 0u; i < LAGFX_MAX_DISPLAYS; ++i) {
        uint64_t ss_gpa = dev->display_ss_gpa[i];
        if (ss_gpa == 0u || !(dev->display_ss_installed & (1u << i))) {
            continue;
        }

        /* Read enable flag at ss[+0x104]. Guest writes 0xC when enable() completes. */
        uint32_t enabled_mask = 0u;
        if (!read_memory(shell_opaque, ss_gpa + 0x104u, sizeof(enabled_mask), &enabled_mask)) {
            LAGFX_LOG("timer_tick_vblank: display[%u] skip (no read)", i);
            continue;
        }

        /* DEBUG: Log every tick to confirm timer fires */
        LAGFX_LOG("timer_tick_vblank: display[%u] ss[+0x104]=%08X", i, enabled_mask);

        /* Check if display just enabled (ss[+0x104] == 0xC). */
        bool enabled = enabled_mask != 0u;
        if (enabled && !(dev->display_ss_enabled & (1u << i))) {
            /* CRITICAL: Raise online IRQ IMMEDIATELY. Do NOT defer or wait. */

            /* Post the connect event: bit 2 in the display's pending events
             * (ss[+0x100]); the guest answers by reading the page's mode and
             * setting up the display pipe. LAGFX_SS_EVENT overrides the bits
             * (bit 3 is the other event the guest enables). */
            const char *ev = getenv("LAGFX_SS_EVENT");
            uint32_t pending = ev ? (uint32_t)strtoul(ev, NULL, 0) : 0x4u;
            if (write_memory(shell_opaque, ss_gpa + 0x100u, sizeof(pending), &pending)) {
                irq_raised = true;
                /* The guest only looks at a display's events when its bit is
                 * set in the mask it reads at BAR0+0x1014. */
                lagfx_protocol_t *pp = (lagfx_protocol_t *)dev->protocol_state;
                atomic_fetch_or_explicit(&pp->pending_displays_bitmask, 1u << i,
                                         memory_order_release);
                LAGFX_LOG("display[%u]: guest enabled events 0x%x, posted 0x%x",
                          i, enabled_mask, pending);

                LAGFX_LOG("timer_tick_vblank: display[%u] enabled (ss[+0x104]=0xC), "
                          "raised IRQ vec=0 IMMEDIATELY", i);
            } else {
                LAGFX_WARN("timer_tick_vblank: display[%u] enable() but write_memory failed", i);
            }

            /* Mark as enabled to avoid repeated IRQ raises. */
            dev->display_ss_enabled |= (1u << i);
        }

        /* The cursor: shared page +0xe00 {u16 x, u16 y}, +0xe04 u32 visible
         * (written by the guest; found by moving the pointer). */
        uint64_t cw = 0u;
        if (read_memory(shell_opaque, ss_gpa + 0xe00u, sizeof cw, &cw)
            && cw != dev->cursor_ss_word[i] && dev->displays[i]) {
            lagfx_display_t *d = dev->displays[i];
            bool vis = (cw >> 32) != 0u, was = (dev->cursor_ss_word[i] >> 32) != 0u;
            bool first = dev->cursor_ss_word[i] == 0u;
            dev->cursor_ss_word[i] = cw;
            d->cursor_pos = (lagfx_coord_t){ (int16_t)(cw & 0xffffu), (int16_t)((cw >> 16) & 0xffffu) };
            if ((first || vis != was) && d->desc.callbacks.cursor_show)
                d->desc.callbacks.cursor_show(d->desc.callbacks.opaque, vis);
            if (d->desc.callbacks.cursor_moved)
                d->desc.callbacks.cursor_moved(d->desc.callbacks.opaque);
        }
    }

    if (irq_raised && dev->desc.shell.raise_interrupt) {
        dev->desc.shell.raise_interrupt(dev->desc.shell.opaque, 0u);
    }

    return irq_raised;
}
