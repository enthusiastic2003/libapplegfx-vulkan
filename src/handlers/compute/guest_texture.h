/*
 * libapplegfx-vulkan — guest textures as render targets
 * src/handlers/compute/guest_texture.h
 *
 * SPDX-License-Identifier: MIT
 *
 * Ties a guest texture object (decoded with protocol/texture_object.h) to a
 * host VkImage registered under (ref, task) in the resource registry, uploads
 * sampled textures from guest memory, runs a
 * render pass's load-action clear on it, and copies CPU-visible textures
 * (shared/managed storage) back into their guest memory once the command
 * buffer that wrote them has been processed.
 */
#ifndef LIBAPPLEGFX_HANDLERS_COMPUTE_GUEST_TEXTURE_H
#define LIBAPPLEGFX_HANDLERS_COMPUTE_GUEST_TEXTURE_H

#include <stdbool.h>
#include <stdint.h>

#include "protocol/state.h"
#include "protocol/texture_object.h"

#ifdef LAGFX_HAVE_VULKAN
#include "vulkan/iosurface.h"

struct lagfx_vk_state;

/* Find or create the host image for texture object `ref` of `task`, sized and
 * formatted from the guest's descriptor. `load`: the pass keeps the previous
 * contents, which then come from guest memory when that is newer. Returns NULL
 * if `ref` isn't a texture object we can decode. *out_obj receives the decoded
 * object. IOSurface textures (object type 0x05) use their surface's memory. */
lagfx_vk_iosurface_t *lagfx_guest_texture_target(lagfx_protocol_t *p,
                                                 lagfx_task_entry_t *task,
                                                 struct lagfx_vk_state *vk,
                                                 uint32_t ref, bool load,
                                                 lagfx_texture_object_t *out_obj);

/* Find or create the host image for texture object `ref` of `task` to sample
 * from. A new image (or one whose guest memory the guest announced it wrote)
 * gets its contents from the texture's guest memory, unless the storage is
 * private or the GPU rendered into it. IOSurface contents are re-read at the
 * first use in each command buffer. Returns NULL if `ref` isn't a texture
 * object we can decode. */
lagfx_vk_iosurface_t *lagfx_guest_texture_source(lagfx_protocol_t *p,
                                                 lagfx_task_entry_t *task,
                                                 struct lagfx_vk_state *vk,
                                                 uint32_t ref);

/* Clear the whole image to rgba (one-shot submit, waits). */
bool lagfx_guest_texture_clear(struct lagfx_vk_state *vk, lagfx_vk_iosurface_t *ios,
                               const double rgba[4]);

/* Copy every texture marked written in this task back to guest memory
 * (CPU-visible storage modes only) and clear the marks. */
void lagfx_guest_texture_writeback(lagfx_protocol_t *p, lagfx_task_entry_t *task,
                                   struct lagfx_vk_state *vk);
#endif

/* Remember that a pass stored into texture `ref` of `task`. */
void lagfx_guest_texture_mark_written(lagfx_task_entry_t *task, uint32_t ref);

#endif /* LIBAPPLEGFX_HANDLERS_COMPUTE_GUEST_TEXTURE_H */
