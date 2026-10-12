/*
 * libapplegfx-vulkan — render pipelines and draws through metal2vulkan
 * src/m2v/m2v_draw.h
 *
 * SPDX-License-Identifier: MIT
 *
 * A pipeline compiled by metal2vulkan (lagfx_m2v.h) with the Vulkan objects
 * made from it, cached by its stage libraries and fixed state; and a draw
 * planned by metal2vulkan from the task's bindings and recorded into the
 * pass's render target. Used when LAGFX_M2V is set; the older translator stays
 * the fallback for anything this path reports it can't do.
 */
#ifndef LAGFX_M2V_DRAW_H
#define LAGFX_M2V_DRAW_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "protocol/state.h"

#if defined(LAGFX_HAVE_M2V) && defined(LAGFX_HAVE_VULKAN)
#include "vulkan/draw_record.h"
#include "vulkan/iosurface.h"

struct lagfx_vk_state;

/* The cached pipeline for these stage libraries and the task's decoded
 * pipeline state (blend, vertex descriptor of `reference`); compiles on a
 * miss. NULL if metal2vulkan refuses (logged once per key). */
void *lagfx_m2v_pipeline_get(lagfx_protocol_t *p, lagfx_task_entry_t *task,
                             struct lagfx_vk_state *vk, uint32_t reference,
                             const uint8_t *vlib, size_t vlen,
                             const uint8_t *flib, size_t flen);

/* Draw `vertex_count` vertices (Metal primitive type `prim`) with the task's
 * bindings into `target` (in `target_layout`, left in COLOR_ATTACHMENT_OPTIMAL
 * and written back to *target_layout). False when this path can't draw it;
 * the caller falls back. */
bool lagfx_m2v_draw(lagfx_protocol_t *p, lagfx_task_entry_t *task, struct lagfx_vk_state *vk,
                    void *pipeline, lagfx_vk_iosurface_t *target, uint32_t *target_layout,
                    const lagfx_draw_region_t *region, uint32_t prim, uint32_t vertex_start,
                    uint32_t vertex_count, uint32_t instance_count);
#endif

#endif /* LAGFX_M2V_DRAW_H */
