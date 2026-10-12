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

/* Copy `ios` (currently in `layout`, which it is left in) into its reused
 * snapshot image and return the snapshot, ready to sample: for a draw that
 * samples the image it renders into, which Vulkan can't do in place. NULL on
 * failure. */
lagfx_vk_iosurface_t *lagfx_guest_texture_snapshot(struct lagfx_vk_state *vk,
                                                   lagfx_vk_iosurface_t *ios,
                                                   VkImageLayout layout);

/* Clear the whole image to rgba (one-shot submit, waits). */
bool lagfx_guest_texture_clear(struct lagfx_vk_state *vk, lagfx_vk_iosurface_t *ios,
                               const double rgba[4]);

/* A blit encoder's copyFromBuffer:...toTexture: (blit op 0x12c). */
typedef struct {
    uint32_t buffer_ref, texture_ref;
    uint64_t offset, bytes_per_row, bytes_per_image;
    uint32_t width, height, depth;
    uint32_t origin_x, origin_y, origin_z;
    uint32_t slice, level;
} lagfx_blit_buffer_to_texture_t;

/* Do the copy in guest memory (the texture's memory, from the buffer's) and
 * mark the texture's host image stale. Level 0 only for now. */
bool lagfx_guest_texture_copy_from_buffer(lagfx_protocol_t *p, lagfx_task_entry_t *task,
                                          const lagfx_blit_buffer_to_texture_t *c);

/* Copy every texture marked written in this task back to guest memory
 * (CPU-visible storage modes only) and clear the marks. */
void lagfx_guest_texture_writeback(lagfx_protocol_t *p, lagfx_task_entry_t *task,
                                   struct lagfx_vk_state *vk);
#endif

/* Read IOSurface `surface_id` (4 bytes per element) from its memory in task 0
 * into a new buffer of width * 4-byte rows (caller frees). NULL on failure. */
uint8_t *lagfx_iosurface_read_pixels(lagfx_protocol_t *p, uint32_t surface_id,
                                     uint32_t *out_w, uint32_t *out_h);

#ifdef LAGFX_HAVE_VULKAN
/* Decode texture object `ref` of `task` (IOSurface textures included). */
bool lagfx_guest_texture_info(lagfx_protocol_t *p, const lagfx_task_entry_t *task, uint32_t ref,
                              lagfx_texture_object_t *out);
#endif

/* Remember that a pass stored into texture `ref` of `task`. */
void lagfx_guest_texture_mark_written(lagfx_task_entry_t *task, uint32_t ref);

#endif /* LIBAPPLEGFX_HANDLERS_COMPUTE_GUEST_TEXTURE_H */
