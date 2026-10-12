/*
 * libapplegfx-vulkan — guest textures as render targets
 * src/handlers/compute/guest_texture.c
 *
 * SPDX-License-Identifier: MIT
 *
 * See guest_texture.h.
 */

#include "guest_texture.h"

#include "common/le.h"
#include "common/log.h"
#include "device.h"
#include "protocol/object_resolver.h"
#include "protocol/resource_registry.h"
#include "task_translate.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Periodic warn-level summary, so cost is visible without info logging. */
static struct {
    unsigned long targets, created, clears, writebacks, skipped_storage, uploads;
    unsigned long long wb_bytes, up_bytes;
    double clear_ms, wb_ms;
    unsigned long events;
} stats;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void stats_tick(void) {
    if (++stats.events % 500u) return;
    LAGFX_WARN("guest_texture stats: targets=%lu created=%lu clears=%lu (%.0f ms) "
               "writebacks=%lu (%llu KiB, %.0f ms) skipped(storage)=%lu uploads=%lu (%llu KiB)",
               stats.targets, stats.created, stats.clears, stats.clear_ms, stats.writebacks,
               stats.wb_bytes / 1024u, stats.wb_ms, stats.skipped_storage, stats.uploads,
               stats.up_bytes / 1024u);
}

void lagfx_guest_texture_mark_written(lagfx_task_entry_t *task, uint32_t ref) {
    if (!task || ref == 0u) return;
    for (uint8_t i = 0; i < task->n_written_tex; i++)
        if (task->written_tex[i] == ref) return;
    if (task->n_written_tex < sizeof task->written_tex / sizeof task->written_tex[0])
        task->written_tex[task->n_written_tex++] = ref;
    else
        LAGFX_WARN("guest_texture: written list full (task %u), dropping ref 0x%x",
                   task->id, ref);
}

#ifdef LAGFX_HAVE_VULKAN
#include "vulkan/command.h"
#include "vulkan/instance.h"
#include "vulkan/render_target.h"

/* An IOSurface texture's memory belongs to its surface: object <surface id>
 * of task 0 (see protocol/texture_object.h). */
static bool attach_surface(lagfx_protocol_t *p, lagfx_texture_object_t *obj,
                           const lagfx_task_entry_t **mem_task) {
    lagfx_task_entry_t *t0 = lagfx_protocol_find_task(p, 0u);
    uint8_t type = 0;
    uint64_t va = 0, gpa = 0;
    uint8_t b[LAGFX_IOSURFACE_RECORD_LEN];
    lagfx_iosurface_record_t rec;
    if (!t0 || !lagfx_resolve_object_data(p, t0, obj->surface_id, &type, &va, &gpa)
        || va == 0u || type != 0x04u || !lagfx_task_read_virtual(p, t0, va, sizeof b, b)
        || !lagfx_parse_iosurface_record(b, sizeof b, &rec)
        || !lagfx_texture_object_attach_surface(obj, &rec)) {
        LAGFX_WARN("guest_texture: IOSurface %u not found in task 0", obj->surface_id);
        return false;
    }
    *mem_task = t0;
    if (getenv("LAGFX_DUMP_WB")) {
        /* once per surface id: where its memory is (guest physical) */
        static uint64_t seen[64];
        uint64_t key = ((uint64_t)obj->surface_id << 32) | rec.page;
        bool dup = false;
        for (unsigned k = 0; k < 64u; k++) dup |= seen[k] == key;
        static unsigned n;
        if (!dup) {
            seen[n++ % 64u] = key;
            uint64_t gpa = 0;
            lagfx_task_translate(p, t0, rec.page << 12, &gpa);
            LAGFX_WARN("SURFACE %u: %ux%u bpr %u fourcc 0x%08x size 0x%llx task0 va 0x%llx gpa 0x%llx",
                       obj->surface_id, rec.width, rec.height, rec.bytes_per_row, rec.fourcc,
                       (unsigned long long)rec.size, (unsigned long long)(rec.page << 12),
                       (unsigned long long)gpa);
        }
    }
    return true;
}

uint8_t *lagfx_iosurface_read_pixels(lagfx_protocol_t *p, uint32_t surface_id,
                                     uint32_t *out_w, uint32_t *out_h) {
    lagfx_task_entry_t *t0 = p ? lagfx_protocol_find_task(p, 0u) : NULL;
    uint8_t type = 0;
    uint64_t va = 0, gpa = 0;
    uint8_t b[LAGFX_IOSURFACE_RECORD_LEN];
    lagfx_iosurface_record_t rec;
    memset(&rec, 0, sizeof rec);
    memset(b, 0, sizeof b);
    bool found = t0 && lagfx_resolve_object_data(p, t0, surface_id, &type, &va, &gpa) && va != 0u;
    bool read = found && lagfx_task_read_virtual(p, t0, va, sizeof b, b);
    bool parsed = read && lagfx_parse_iosurface_record(b, sizeof b, &rec);
    if (!found || type != 0x04u || !parsed || rec.bytes_per_element != 4u) {
        LAGFX_WARN("iosurface: surface %u not readable (found=%d type=0x%02x read=%d parsed=%d "
                   "%ux%u bpr %u bpe %u size 0x%llx fourcc 0x%08x)",
                   surface_id, found, type, read, parsed, lagfx_le32(b + 0x18),
                   lagfx_le32(b + 0x1c), (unsigned)(b[0x20] | b[0x21] << 8),
                   (unsigned)(b[0x22] | b[0x23] << 8), (unsigned long long)lagfx_le64(b),
                   lagfx_le32(b + 0x0c));
        return NULL;
    }
    size_t row = (size_t)rec.width * 4u;
    uint8_t *px = malloc(row * rec.height);
    if (!px) return NULL;
    for (uint32_t y = 0; y < rec.height; y++) {
        if (!lagfx_task_read_virtual(p, t0, (rec.page << 12) + (uint64_t)y * rec.bytes_per_row,
                                     (uint32_t)row, px + (size_t)y * row)) {
            LAGFX_WARN("iosurface: surface %u row %u unreadable", surface_id, y);
            free(px);
            return NULL;
        }
    }
    *out_w = rec.width;
    *out_h = rec.height;
    return px;
}

/* Decode texture object `ref` of `task`. *mem_task receives the task whose
 * address space holds the texels (task 0 for IOSurfaces). */
static bool read_texture_object(lagfx_protocol_t *p, const lagfx_task_entry_t *task,
                                uint32_t ref, lagfx_texture_object_t *obj,
                                const lagfx_task_entry_t **mem_task) {
    uint8_t type = 0;
    uint64_t va = 0, gpa = 0;
    if (!lagfx_resolve_object_data(p, task, ref, &type, &va, &gpa) || va == 0u) return false;
    *mem_task = task;
    if (type == 0x05u) {
        uint8_t b[LAGFX_IOSURFACE_TEXTURE_MIN_LEN];
        return lagfx_task_read_virtual(p, task, va, sizeof b, b)
               && lagfx_parse_iosurface_texture_object(b, sizeof b, obj)
               && attach_surface(p, obj, mem_task);
    }
    if (type != 0x02u && type != 0x03u && type != 0x0cu) return false;
    uint8_t b[LAGFX_TEXTURE_OBJECT_MIN_LEN];
    if (!lagfx_task_read_virtual(p, task, va, sizeof b, b)) return false;
    return lagfx_parse_texture_object(b, sizeof b, obj);
}

bool lagfx_guest_texture_info(lagfx_protocol_t *p, const lagfx_task_entry_t *task, uint32_t ref,
                              lagfx_texture_object_t *out) {
    const lagfx_task_entry_t *mem_task;
    return p && task && out && read_texture_object(p, task, ref, out, &mem_task);
}

/* Find or create the host image for (task, ref) matching the decoded object.
 * *created is set when a new image was made (its contents are undefined). */
static lagfx_vk_iosurface_t *ensure_image(lagfx_protocol_t *p, lagfx_task_entry_t *task,
                                          struct lagfx_vk_state *vk, uint32_t ref,
                                          const lagfx_texture_object_t *obj, bool *created) {
    *created = false;
    VkFormat fmt = lagfx_metal_pixel_format_to_vk(obj->pixel_format);
    /* Object ids are reused for new objects; the texture's memory location
     * tells an entry for a new texture from a stale one. */
    uint64_t where = obj->n_place ? obj->place[0].page << 12 : 0u;
    lagfx_resource_entry_t *e = lagfx_resource_lookup(&p->resources, ref, task->id);
    if (e && e->type == LAGFX_RESOURCE_TYPE_TEXTURE && e->host_handle) {
        lagfx_vk_iosurface_t *cur = (lagfx_vk_iosurface_t *)e->host_handle;
        if (cur->width == obj->width && cur->height == obj->height && cur->format == fmt) {
            if (e->gpu_addr == where) return cur;
            /* same shape, different texture: reuse the image, new contents */
            e->gpu_addr = where;
            e->backing_dirty = 0u;
            cur->gpu_drawn = 0u;
            *created = true;
            return cur;
        }
    }

    lagfx_vk_iosurface_t *ios = NULL;
    if (lagfx_vk_iosurface_create(vk, obj->width, obj->height, obj->pixel_format, &ios) != LAGFX_OK
        || !ios)
        return NULL;
    if (!e || e->type != LAGFX_RESOURCE_TYPE_TEXTURE) {
        lagfx_resource_register(&p->resources, ref, LAGFX_RESOURCE_TYPE_TEXTURE, task->id, 0u, 0u);
        e = lagfx_resource_lookup(&p->resources, ref, task->id);
        if (!e || e->type != LAGFX_RESOURCE_TYPE_TEXTURE) {
            LAGFX_WARN("guest_texture: can't register ref 0x%x (task %u)", ref, task->id);
            lagfx_vk_iosurface_destroy(vk, ios);
            return NULL;
        }
    } else if (e->host_handle) {
        /* Same ref, different size/format (the guest reused the object id, or
         * an older path created it with guessed dimensions). The old image is
         * deliberately not freed: cached descriptor sets may still reference
         * it. TODO: release once descriptor/pipeline caches track images. */
        LAGFX_LOG("guest_texture: replacing image for ref 0x%x (task %u)", ref, task->id);
    }
    stats.created++;
    e->host_handle = ios;
    e->image = ios->image;
    e->view = ios->view;
    e->size = (uint64_t)obj->bytes_per_row * obj->height;
    e->gpu_addr = where;
    e->backing_dirty = 0u;
    *created = true;
    LAGFX_LOG("guest_texture: task %u ref 0x%x -> %ux%u fmt %u storage %u bpr %u",
              task->id, ref, obj->width, obj->height, obj->pixel_format, obj->storage,
              obj->bytes_per_row);
    return ios;
}

/* LAGFX_DUMP_WB=<dir>: save each written-back (wb-) and uploaded (up-) image
 * of a 4-byte format, read as BGRA, as <dir>/<kind>-t<task>-r<ref>-s<surface>.ppm
 * (surface 0: not an IOSurface); the latest wins. */
static void dump_image(const char *kind, const lagfx_task_entry_t *task, uint32_t ref,
                       const lagfx_texture_object_t *obj, const uint8_t *px, size_t stride) {
    const char *dir = getenv("LAGFX_DUMP_WB");
    if (!dir || lagfx_metal_pixel_format_bytes(obj->pixel_format) != 4u) return;
    char path[512];
    snprintf(path, sizeof path, "%s/%s-t%u-r%x-s%u.ppm", dir, kind, task->id, ref,
             obj->surface_id);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P6\n%u %u\n255\n", obj->width, obj->height);
    for (uint32_t y = 0; y < obj->height; y++)
        for (uint32_t x = 0; x < obj->width; x++) {
            const uint8_t *q = px + (size_t)y * stride + (size_t)x * 4u;
            uint8_t rgb[3] = { q[2], q[1], q[0] };
            fwrite(rgb, 1, 3, f);
        }
    fclose(f);
}

static bool upload_from_guest(lagfx_protocol_t *p, const lagfx_task_entry_t *task,
                              const lagfx_task_entry_t *mem_task,
                              struct lagfx_vk_state *vk, lagfx_vk_iosurface_t *ios,
                              const lagfx_texture_object_t *obj, uint32_t ref);

/* IOSurfaces are written by the CPU without notice, so their contents are
 * re-read once per command buffer, unless the GPU copy is newer. */
static bool needs_refresh(const lagfx_task_entry_t *task, const lagfx_vk_iosurface_t *ios,
                          const lagfx_texture_object_t *obj) {
    return obj->surface_id != 0u && !ios->gpu_drawn && ios->upload_gen != task->cmdbuf_gen + 1u;
}

lagfx_vk_iosurface_t *lagfx_guest_texture_target(lagfx_protocol_t *p,
                                                 lagfx_task_entry_t *task,
                                                 struct lagfx_vk_state *vk,
                                                 uint32_t ref, bool load,
                                                 lagfx_texture_object_t *out_obj) {
    lagfx_texture_object_t obj;
    const lagfx_task_entry_t *mem_task;
    if (!p || !task || !vk || !read_texture_object(p, task, ref, &obj, &mem_task)) return NULL;
    if (out_obj) *out_obj = obj;
    stats.targets++;
    stats_tick();
    bool created;
    lagfx_vk_iosurface_t *ios = ensure_image(p, task, vk, ref, &obj, &created);
    if (!ios) return NULL;
    /* A pass that loads the previous contents of an IOSurface (or of a new
     * image for CPU-visible memory) starts from guest memory. */
    if (load && obj.storage != LAGFX_STORAGE_PRIVATE
        && (needs_refresh(task, ios, &obj) || (created && !ios->gpu_drawn))
        && upload_from_guest(p, task, mem_task, vk, ios, &obj, ref))
        ios->upload_gen = task->cmdbuf_gen + 1u;
    /* Rendered into: the GPU copy is now the authoritative one. */
    ios->gpu_drawn = 1u;
    return ios;
}

/* Read `len` bytes at texture-memory offset `off` through the placement
 * entries (task virtual addresses). */
static bool read_texture_memory(lagfx_protocol_t *p, const lagfx_task_entry_t *task,
                                const lagfx_texture_object_t *obj, uint64_t off,
                                uint8_t *dst, uint64_t len) {
    uint64_t base = 0;
    for (uint32_t e = 0; e < obj->n_place && len > 0u; e++) {
        uint64_t esize = obj->place[e].size;
        if (off >= base + esize) { base += esize; continue; }
        uint64_t va = (obj->place[e].page << 12) + (off - base);
        uint64_t n = base + esize - off;
        if (n > len) n = len;
        if (!lagfx_task_read_virtual(p, task, va, (uint32_t)n, dst)) return false;
        dst += n; off += n; len -= n;
        base += esize;
    }
    return len == 0u;
}

/* Copy the texture's guest memory (rows of bytes_per_row) into the image. */
static bool upload_from_guest(lagfx_protocol_t *p, const lagfx_task_entry_t *task,
                              const lagfx_task_entry_t *mem_task,
                              struct lagfx_vk_state *vk, lagfx_vk_iosurface_t *ios,
                              const lagfx_texture_object_t *obj, uint32_t ref) {
    uint32_t bpp = lagfx_metal_pixel_format_bytes(obj->pixel_format);
    if (bpp == 0u) {
        LAGFX_WARN("guest_texture: upload of format %u not supported (ref 0x%x)",
                   obj->pixel_format, ref);
        return false;
    }
    size_t tight = (size_t)obj->width * bpp;
    if (obj->bytes_per_row < tight) return false;
    uint8_t *px = malloc(tight * obj->height);
    if (!px) return false;
    bool ok = true;
    for (uint32_t y = 0; y < obj->height && ok; y++)
        ok = read_texture_memory(p, mem_task, obj, (uint64_t)y * obj->bytes_per_row,
                                 px + (size_t)y * tight, tight);
    if (getenv("LAGFX_DUMP_OBJ")) {
        uint32_t nz = 0;
        for (size_t k = 0; ok && k < tight * obj->height; k++) nz += px[k] != 0u;
        LAGFX_WARN("guest_texture: upload task %u ref 0x%x surface %u %ux%u fmt %u storage %u "
                   "bpr %u place %u x {0x%llx @ page 0x%llx} read=%d nonzero=%u/%zu "
                   "first=%02x%02x%02x%02x",
                   task->id, ref, obj->surface_id, obj->width, obj->height, obj->pixel_format, obj->storage,
                   obj->bytes_per_row, obj->n_place,
                   (unsigned long long)(obj->n_place ? obj->place[0].size : 0u),
                   (unsigned long long)(obj->n_place ? obj->place[0].page : 0u), ok, nz,
                   tight * obj->height, px[0], px[1], px[2], px[3]);
    }
    if (ok) dump_image("up", task, ref, obj, px, tight);
    if (ok) ok = lagfx_vk_iosurface_upload_pixels(vk, ios, px, tight * obj->height) == LAGFX_OK;
    free(px);
    stats.uploads++;
    stats.up_bytes += (unsigned long long)tight * obj->height;
    stats_tick();
    if (!ok)
        LAGFX_WARN("guest_texture: upload of ref 0x%x (task %u) failed", ref, task->id);
    return ok;
}

lagfx_vk_iosurface_t *lagfx_guest_texture_source(lagfx_protocol_t *p,
                                                 lagfx_task_entry_t *task,
                                                 struct lagfx_vk_state *vk,
                                                 uint32_t ref) {
    lagfx_texture_object_t obj;
    const lagfx_task_entry_t *mem_task;
    if (!p || !task || !vk || !read_texture_object(p, task, ref, &obj, &mem_task)) return NULL;
    bool created;
    lagfx_vk_iosurface_t *ios = ensure_image(p, task, vk, ref, &obj, &created);
    if (!ios) return NULL;
    lagfx_resource_entry_t *e = lagfx_resource_lookup(&p->resources, ref, task->id);
    bool dirty = e && e->backing_dirty;
    /* Contents come from guest memory when the image is new, when the guest
     * announced a CPU write since, or (IOSurfaces) at the first use in a
     * command buffer. Private textures have no CPU-written memory, and an
     * image the GPU rendered into is newer than guest memory. */
    if (obj.storage != LAGFX_STORAGE_PRIVATE
        && (((created || dirty) && (!ios->gpu_drawn || dirty)) || needs_refresh(task, ios, &obj))) {
        if (upload_from_guest(p, task, mem_task, vk, ios, &obj, ref)) {
            ios->gpu_drawn = 0u;
            ios->upload_gen = task->cmdbuf_gen + 1u;
            if (e) e->backing_dirty = 0u;
        }
    }
    return ios;
}

/* Record + submit a one-shot command buffer and wait for it. */
static bool submit_wait(struct lagfx_vk_state *vk, VkCommandBuffer cb) {
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence = VK_NULL_HANDLE;
    if (vkCreateFence(vk->device, &fci, NULL, &fence) != VK_SUCCESS) return false;
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
    };
    bool ok = vkQueueSubmit(vk->graphics_queue, 1, &si, fence) == VK_SUCCESS
              && vkWaitForFences(vk->device, 1, &fence, VK_TRUE, 2000000000ull) == VK_SUCCESS;
    vkDestroyFence(vk->device, fence, NULL);
    return ok;
}

lagfx_vk_iosurface_t *lagfx_guest_texture_snapshot(struct lagfx_vk_state *vk,
                                                   lagfx_vk_iosurface_t *ios,
                                                   VkImageLayout layout) {
    if (!vk || !ios || ios->image == VK_NULL_HANDLE || layout == VK_IMAGE_LAYOUT_UNDEFINED)
        return NULL;
    lagfx_vk_iosurface_t *snap = ios->snapshot;
    if (snap && (snap->width != ios->width || snap->height != ios->height
                 || snap->format != ios->format)) {
        lagfx_vk_iosurface_destroy(vk, snap);
        snap = ios->snapshot = NULL;
    }
    if (!snap) {
        if (lagfx_vk_iosurface_create_vk(vk, ios->width, ios->height, ios->format, &snap)
                != LAGFX_OK || !snap)
            return NULL;
        ios->snapshot = snap;
    }
    VkCommandBuffer cb = VK_NULL_HANDLE;
    if (lagfx_vk_cmdbuf_alloc(vk, &cb) != LAGFX_OK) return NULL;
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cb, &bi);
    VkImageSubresourceRange range = {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1,
    };
    VkImageMemoryBarrier pre[2] = {
        {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout = layout,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = ios->image,
            .subresourceRange = range,
        },
        {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            /* fully overwritten */
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = snap->image,
            .subresourceRange = range,
        },
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, NULL, 0, NULL, 2, pre);
    VkImageCopy region = {
        .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .extent = { ios->width, ios->height, 1 },
    };
    vkCmdCopyImage(cb, ios->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, snap->image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    VkImageMemoryBarrier post[2] = { pre[0], pre[1] };
    post[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    post[0].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    post[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    post[0].newLayout = layout;
    post[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    post[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    post[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    post[1].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, 0, NULL, 0, NULL, 2, post);
    vkEndCommandBuffer(cb);
    bool ok = submit_wait(vk, cb);
    lagfx_vk_cmdbuf_free(vk, cb);
    if (!ok) return NULL;
    snap->layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return snap;
}

bool lagfx_guest_texture_clear(struct lagfx_vk_state *vk, lagfx_vk_iosurface_t *ios,
                               const double rgba[4]) {
    if (!vk || !ios || ios->image == VK_NULL_HANDLE) return false;
    double t0 = now_ms();
    VkCommandBuffer cb = VK_NULL_HANDLE;
    if (lagfx_vk_cmdbuf_alloc(vk, &cb) != LAGFX_OK) return false;
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cb, &bi);
    VkImageSubresourceRange range = {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1,
    };
    VkImageMemoryBarrier to_dst = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        /* A clear overwrites everything, so prior contents may be discarded. */
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = ios->image,
        .subresourceRange = range,
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, NULL, 0, NULL, 1, &to_dst);
    VkClearColorValue cv = { .float32 = { (float)rgba[0], (float)rgba[1],
                                          (float)rgba[2], (float)rgba[3] } };
    vkCmdClearColorImage(cb, ios->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &cv, 1, &range);
    VkImageMemoryBarrier to_read = to_dst;
    to_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    to_read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, 0, NULL, 0, NULL, 1, &to_read);
    vkEndCommandBuffer(cb);
    bool ok = submit_wait(vk, cb);
    lagfx_vk_cmdbuf_free(vk, cb);
    if (ok) ios->layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    stats.clears++;
    stats.clear_ms += now_ms() - t0;
    return ok;
}

static uint32_t bytes_per_pixel(uint32_t metal_format) {
    /* the readback helper is 4-byte only */
    return lagfx_metal_pixel_format_bytes(metal_format) == 4u ? 4u : 0u;
}

/* Write `len` bytes at texture-memory offset `off` through the placement
 * entries, page by page via the task's VA translation. Never writes to an
 * address that doesn't translate. */
static bool write_texture_memory(lagfx_protocol_t *p, const lagfx_task_entry_t *mem_task,
                                 const lagfx_texture_object_t *obj, uint64_t off,
                                 const uint8_t *src, uint64_t len) {
    lagfx_device_t *dev = (lagfx_device_t *)p->dev;
    if (!dev->desc.shell.write_memory) return false;
    uint64_t base = 0;
    for (uint32_t e = 0; e < obj->n_place && len > 0u; e++) {
        uint64_t esize = obj->place[e].size;
        if (off >= base + esize) { base += esize; continue; }
        uint64_t va = (obj->place[e].page << 12) + (off - base);
        uint64_t n = base + esize - off;
        if (n > len) n = len;
        while (n > 0u) {
            uint64_t page_left = 0x1000u - (va & 0xfffu);
            uint64_t chunk = n < page_left ? n : page_left;
            uint64_t gpa = 0;
            if (!lagfx_task_translate(p, mem_task, va, &gpa)) return false;
            if (!dev->desc.shell.write_memory(dev->desc.shell.opaque, gpa, chunk, src))
                return false;
            va += chunk; src += chunk; off += chunk; len -= chunk; n -= chunk;
        }
        base += esize;
    }
    return len == 0u;
}

static void writeback_one(lagfx_protocol_t *p, lagfx_task_entry_t *task,
                          struct lagfx_vk_state *vk, uint32_t ref) {
    lagfx_texture_object_t obj;
    const lagfx_task_entry_t *mem_task;
    if (!read_texture_object(p, task, ref, &obj, &mem_task)) return;
    /* Managed textures and IOSurfaces only. Metal makes managed textures
     * CPU-visible at an explicit synchronize, which is when the guest reads
     * them; IOSurfaces are read by the CPU (and other processes) directly, so
     * they're copied after every command buffer that wrote them. Other shared
     * textures would need the same copy — too slow this way; they need
     * host-mapped guest memory instead. */
    if (obj.storage != LAGFX_STORAGE_MANAGED && obj.surface_id == 0u) {
        stats.skipped_storage++;
        return;
    }
    double t0 = now_ms();
    uint32_t bpp = bytes_per_pixel(obj.pixel_format);
    if (bpp == 0u) {
        LAGFX_WARN("guest_texture: write-back of format %u not supported (ref 0x%x)",
                   obj.pixel_format, ref);
        return;
    }
    lagfx_resource_entry_t *e = lagfx_resource_lookup(&p->resources, ref, task->id);
    lagfx_vk_iosurface_t *ios = e ? (lagfx_vk_iosurface_t *)e->host_handle : NULL;
    if (!ios || ios->width != obj.width || ios->height != obj.height) return;

    lagfx_vk_render_target_t rt;
    if (lagfx_vk_render_target_wrap(ios->image, ios->view, ios->memory, ios->width,
                                    ios->height, ios->format, &rt) != LAGFX_OK)
        return;
    rt.layout = ios->layout;
    size_t tight = (size_t)obj.width * bpp, stride = 0;
    uint8_t *px = malloc(tight * obj.height);
    if (!px) return;
    if (lagfx_vk_render_target_readback(vk, &rt, px, tight * obj.height, &stride) != LAGFX_OK) {
        free(px);
        return;
    }
    ios->layout = rt.layout;
    if (stride == 0u) stride = tight;
    dump_image("wb", task, ref, &obj, px, stride);
    bool ok = true;
    for (uint32_t y = 0; y < obj.height && ok; y++)
        ok = write_texture_memory(p, mem_task, &obj, (uint64_t)y * obj.bytes_per_row,
                                  px + (size_t)y * stride, tight);
    free(px);
    stats.writebacks++;
    stats.wb_bytes += (unsigned long long)tight * obj.height;
    stats.wb_ms += now_ms() - t0;
    stats_tick();
    /* Guest memory now matches the image, so later CPU writes to a surface
     * may be picked up again. */
    if (ok && obj.surface_id != 0u) ios->gpu_drawn = 0u;
    if (ok)
        LAGFX_LOG("guest_texture: wrote back task %u ref 0x%x (%ux%u)", task->id, ref,
                  obj.width, obj.height);
    else
        LAGFX_WARN("guest_texture: write-back of ref 0x%x (task %u) failed to reach guest memory",
                   ref, task->id);
}

bool lagfx_guest_texture_copy_from_buffer(lagfx_protocol_t *p, lagfx_task_entry_t *task,
                                          const lagfx_blit_buffer_to_texture_t *c) {
    lagfx_texture_object_t obj;
    const lagfx_task_entry_t *mem_task;
    if (!p || !task || !c || !read_texture_object(p, task, c->texture_ref, &obj, &mem_task))
        return false;
    if (c->level != 0u) {
        LAGFX_WARN("guest_texture: copy into level %u of texture 0x%x not supported", c->level,
                   c->texture_ref);
        return false;
    }
    uint32_t bpp = lagfx_metal_pixel_format_bytes(obj.pixel_format);
    if (!bpp) return false;
    /* the source buffer: one {size, page} entry (m2v/m2v_draw.c) */
    uint8_t type = 0, d[16];
    uint64_t va = 0, gpa = 0;
    if (!lagfx_resolve_object_data(p, task, c->buffer_ref, &type, &va, &gpa) || va == 0u
        || !lagfx_task_read_virtual(p, task, va, sizeof d, d))
        return false;
    uint64_t bsize = lagfx_le64(d), bva = (uint64_t)lagfx_le32(d + 8) << 12;
    uint64_t row = (uint64_t)c->width * bpp;
    uint64_t slice_stride = (uint64_t)obj.bytes_per_row * obj.height;   /* level 0 image */
    uint8_t *tmp = malloc((size_t)row);
    if (!tmp) return false;
    bool ok = true;
    for (uint32_t z = 0; z < c->depth && ok; z++)
        for (uint32_t y = 0; y < c->height && ok; y++) {
            uint64_t so = c->offset + (uint64_t)z * c->bytes_per_image + (uint64_t)y * c->bytes_per_row;
            if (so + row > bsize) { ok = false; break; }
            uint64_t layer = obj.texture_type == 7u ? c->origin_z + z : c->slice + z;
            uint64_t to = layer * slice_stride + (uint64_t)(c->origin_y + y) * obj.bytes_per_row
                        + (uint64_t)c->origin_x * bpp;
            ok = lagfx_task_read_virtual(p, task, bva + so, (uint32_t)row, tmp)
                 && write_texture_memory(p, mem_task, &obj, to, tmp, row);
        }
    free(tmp);
    /* the host image is stale now */
    lagfx_resource_entry_t *e = lagfx_resource_lookup(&p->resources, c->texture_ref, task->id);
    if (e) {
        e->backing_dirty = 1u;
        if (e->host_handle) ((lagfx_vk_iosurface_t *)e->host_handle)->gpu_drawn = 0u;
    }
    if (!ok)
        LAGFX_WARN("guest_texture: copy from buffer 0x%x into texture 0x%x failed", c->buffer_ref,
                   c->texture_ref);
    return ok;
}

void lagfx_guest_texture_writeback(lagfx_protocol_t *p, lagfx_task_entry_t *task,
                                   struct lagfx_vk_state *vk) {
    if (!p || !task || !vk) return;
    for (uint8_t i = 0; i < task->n_written_tex; i++)
        writeback_one(p, task, vk, task->written_tex[i]);
    task->n_written_tex = 0;
}
#endif /* LAGFX_HAVE_VULKAN */
