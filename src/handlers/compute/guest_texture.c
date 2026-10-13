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
    unsigned long mapped_uploads, mapped_writebacks;   /* GPU copies (guest_map_copy) */
    unsigned long skipped_uploads;                     /* guest memory unchanged */
    unsigned long long wb_bytes, up_bytes;
    double clear_ms, wb_ms, up_ms;
    unsigned long events;
} stats;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void stats_tick(void) {
    /* every 500 events, and at least once a second while busy */
    static double last;
    double now = now_ms();
    if (++stats.events % 500u && now - last < 1000.0) return;
    last = now;
    LAGFX_WARN("guest_texture stats: targets=%lu created=%lu clears=%lu (%.0f ms) "
               "writebacks=%lu (%llu KiB, %.0f ms) skipped(storage)=%lu uploads=%lu (%llu KiB) "
               "upload_ms=%.0f mapped: uploads=%lu writebacks=%lu unchanged=%lu",
               stats.targets, stats.created, stats.clears, stats.clear_ms, stats.writebacks,
               stats.wb_bytes / 1024u, stats.wb_ms, stats.skipped_storage, stats.uploads,
               stats.up_bytes / 1024u, stats.up_ms, stats.mapped_uploads, stats.mapped_writebacks,
               stats.skipped_uploads);
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
#include "vulkan/guest_map.h"
#include "vulkan/render_target.h"

/* Read an IOSurface record: one plane's worth, then the rest of the plane
 * entries when it has more (a full-length read could run off the page). */
static bool read_surface_record(lagfx_protocol_t *p, const lagfx_task_entry_t *t0, uint64_t va,
                                uint8_t *b, size_t cap, lagfx_iosurface_record_t *rec) {
    size_t len = LAGFX_IOSURFACE_RECORD_MIN_LEN;
    if (cap < len || !lagfx_task_read_virtual(p, t0, va, (uint32_t)len, b)) return false;
    uint32_t count = lagfx_le32(b + 0x10);
    if (count > 1u) {
        len = 0x14u + 0x10u * (size_t)count;
        if (len > cap || !lagfx_task_read_virtual(p, t0, va, (uint32_t)len, b)) return false;
    }
    return lagfx_parse_iosurface_record(b, len, rec);
}

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
        || va == 0u || type != 0x04u || !read_surface_record(p, t0, va, b, sizeof b, &rec)
        || !lagfx_texture_object_attach_surface(obj, &rec)) {
        LAGFX_WARN("guest_texture: IOSurface %u (plane %u) not found in task 0", obj->surface_id,
                   obj->plane);
        return false;
    }
    *mem_task = t0;
    if (getenv("LAGFX_SURFLOG")) {
        /* once per (surface, page): the raw record */
        static uint64_t seen_r[256];
        static unsigned nr;
        uint64_t key = ((uint64_t)obj->surface_id << 32) | rec.page;
        bool dup = false;
        for (unsigned k = 0; k < 256u; k++) dup |= seen_r[k] == key;
        if (!dup) {
            seen_r[nr++ % 256u] = key;
            /* past the fields we decode: multi-plane surfaces carry more */
            uint8_t wide[0xa0];
            if (!lagfx_task_read_virtual(p, t0, va, sizeof wide, wide)) memset(wide, 0, sizeof wide);
            for (unsigned row = 0; row < sizeof wide; row += 32u) {
                char hx[3 * 32 + 1];
                size_t hn = 0;
                for (unsigned k = row; k < row + 32u; k++)
                    hn += (size_t)snprintf(hx + hn, sizeof hx - hn, "%02x ", wide[k]);
                LAGFX_WARN("SURFREC id=%u tex %ux%u fmt %u +%02x: %s", obj->surface_id, obj->width,
                           obj->height, obj->pixel_format, row, hx);
            }
        }
    }
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
    bool read = found && lagfx_task_read_virtual(p, t0, va, LAGFX_IOSURFACE_RECORD_MIN_LEN, b);
    bool parsed = read && read_surface_record(p, t0, va, b, sizeof b, &rec);
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
/* A buffer texture's memory is its buffer's: up to four {u64 size, u32 page,
 * u32 flags} entries, flag bit 16 on the last. */
static bool attach_buffer(lagfx_protocol_t *p, const lagfx_task_entry_t *task,
                          lagfx_texture_object_t *obj) {
    uint8_t type = 0, d[64];
    uint64_t va = 0, gpa = 0;
    if (!lagfx_resolve_object_data(p, task, obj->buffer_ref, &type, &va, &gpa) || va == 0u
        || !lagfx_task_read_virtual(p, task, va, sizeof d, d))
        return false;
    uint64_t total = 0;
    for (uint32_t e = 0; e < 4u; e++) {
        uint64_t size = lagfx_le64(d + e * 16u);
        uint32_t page = lagfx_le32(d + e * 16u + 8u), flags = lagfx_le32(d + e * 16u + 12u);
        if (size == 0u || page == 0u) break;
        obj->place[obj->n_place].size = size;
        obj->place[obj->n_place].page = page;
        obj->n_place++;
        total += size;
        if (flags & 0x10000u) break;
    }
    uint64_t need = obj->base_offset + (uint64_t)obj->bytes_per_row * (obj->height - 1u)
                  + (uint64_t)obj->width * lagfx_metal_pixel_format_bytes(obj->pixel_format);
    if (obj->n_place == 0u || need > total) {
        LAGFX_WARN("guest_texture: buffer 0x%x (%llu bytes in %u entries) can't hold the texture "
                   "(%llu bytes)", obj->buffer_ref, (unsigned long long)total, obj->n_place,
                   (unsigned long long)need);
        return false;
    }
    return true;
}

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
    if (type == 0x08u) {
        uint8_t b[LAGFX_BUFFER_TEXTURE_MIN_LEN];
        return lagfx_task_read_virtual(p, task, va, sizeof b, b)
               && lagfx_parse_buffer_texture_object(b, sizeof b, obj)
               && attach_buffer(p, task, obj);
    }
    if (type != 0x02u && type != 0x03u && type != 0x0cu) return false;
    /* a mipmapped texture's descriptor follows its level records */
    uint8_t b[LAGFX_TEXTURE_OBJECT_MAX_LEN];
    if (lagfx_task_read_virtual(p, task, va, sizeof b, b))
        return lagfx_parse_texture_object(b, sizeof b, obj);
    return lagfx_task_read_virtual(p, task, va, LAGFX_TEXTURE_OBJECT_MIN_LEN, b)
           && lagfx_parse_texture_object(b, LAGFX_TEXTURE_OBJECT_MIN_LEN, obj);
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
    uint64_t where = obj->n_place ? (obj->place[0].page << 12) + obj->base_offset : 0u;
    lagfx_resource_entry_t *e = lagfx_resource_lookup(&p->resources, ref, task->id);
    if (e && e->type == LAGFX_RESOURCE_TYPE_TEXTURE && e->host_handle) {
        lagfx_vk_iosurface_t *cur = (lagfx_vk_iosurface_t *)e->host_handle;
        if (cur->width == obj->width && cur->height == obj->height && cur->format == fmt) {
            if (e->gpu_addr == where) return cur;
            /* same shape, different texture: reuse the image, new contents */
            e->gpu_addr = where;
            e->backing_dirty = 0u;
            cur->gpu_drawn = 0u;
            cur->sum_valid = 0u;
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
    e->surface_id = obj->surface_id;
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
static uint8_t half_to_u8(uint16_t h) {
    if (h & 0x8000u) return 0u;
    uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    if (e >= 15u) return 255u;   /* >= 1.0, inf, nan */
    float f = e ? (1.0f + (float)m / 1024.0f) / (float)(1u << (15u - e))
                : (float)m / (1024.0f * 16384.0f);
    return (uint8_t)(f * 255.0f + 0.5f);
}

/* LAGFX_DUMP_WB=<dir> dumps uploads/write-backs as <kind>-t<task>-r<ref>-s<surface>.ppm
 * (RGB) and .pgm (alpha). LAGFX_DUMP_WB_TASK=<id> keeps one task. 16-byte
 * RGBA32Uint/Sint (123/124) is shown as the BGRA8 bytes it packs (4x width);
 * RGBA16Float (115) is clamped to 0..1. LAGFX_DUMP_WB_MAXPX=<n> skips images
 * of more than n pixels. */
static void dump_image(const char *kind, const lagfx_task_entry_t *task, uint32_t ref,
                       const lagfx_texture_object_t *obj, const uint8_t *px, size_t stride) {
    const char *dir = getenv("LAGFX_DUMP_WB");
    const char *only = getenv("LAGFX_DUMP_WB_TASK");
    const char *maxpx = getenv("LAGFX_DUMP_WB_MAXPX");
    if (!dir || (only && strtoul(only, NULL, 0) != task->id)) return;
    if (maxpx && (uint64_t)obj->width * obj->height > strtoull(maxpx, NULL, 0)) return;
    uint32_t fmt = obj->pixel_format, bpp = lagfx_metal_pixel_format_bytes(fmt), w = obj->width;
    bool packed = fmt == 123u || fmt == 124u, half = fmt == 115u;
    if (packed) { bpp = 4u; w *= 4u; }
    if (bpp != 4u && bpp != 1u && !half) return;
    char path[512];
    snprintf(path, sizeof path, "%s/%s-t%u-r%x-s%u-f%u.ppm", dir, kind, task->id, ref,
             obj->surface_id, fmt);
    FILE *f = fopen(path, "wb");
    path[strlen(path) - 3] = 'p'; path[strlen(path) - 2] = 'g'; path[strlen(path) - 1] = 'm';
    FILE *fa = fopen(path, "wb");
    if (!f || !fa) { if (f) fclose(f); if (fa) fclose(fa); return; }
    fprintf(f, "P6\n%u %u\n255\n", w, obj->height);
    fprintf(fa, "P5\n%u %u\n255\n", w, obj->height);
    for (uint32_t y = 0; y < obj->height; y++)
        for (uint32_t x = 0; x < w; x++) {
            uint8_t rgb[3], a = 255u;
            if (half) {
                const uint8_t *q = px + (size_t)y * stride + (size_t)x * 8u;
                rgb[0] = half_to_u8((uint16_t)(q[0] | q[1] << 8));
                rgb[1] = half_to_u8((uint16_t)(q[2] | q[3] << 8));
                rgb[2] = half_to_u8((uint16_t)(q[4] | q[5] << 8));
                a = half_to_u8((uint16_t)(q[6] | q[7] << 8));
            } else {
                const uint8_t *q = px + (size_t)y * stride + (size_t)x * bpp;
                rgb[0] = q[bpp == 4u ? 2 : 0]; rgb[1] = q[bpp == 4u ? 1 : 0]; rgb[2] = q[0];
                if (bpp == 4u) a = q[3];
                if (fmt == 94u || fmt == 90u) {
                    /* 10-bit channels, low bits first: B,G,R (94) or R,G,B (90) */
                    uint32_t v = lagfx_le32(q);
                    uint8_t c0 = (uint8_t)((v >> 2) & 0xffu), c1 = (uint8_t)((v >> 12) & 0xffu),
                            c2 = (uint8_t)((v >> 22) & 0xffu);
                    rgb[0] = fmt == 94u ? c2 : c0;
                    rgb[1] = c1;
                    rgb[2] = fmt == 94u ? c0 : c2;
                    a = 255u;
                }
            }
            fwrite(rgb, 1, 3, f);
            fputc(a, fa);
        }
    fclose(f);
    fclose(fa);
}

static bool upload_from_guest(lagfx_protocol_t *p, const lagfx_task_entry_t *task,
                              const lagfx_task_entry_t *mem_task,
                              struct lagfx_vk_state *vk, lagfx_vk_iosurface_t *ios,
                              const lagfx_texture_object_t *obj, uint32_t ref);

/* The guest's memory is the texture: the CPU writes IOSurfaces, managed and
 * shared textures (replaceRegion, Core Animation's text masks and password dots)
 * without any command, so their contents are re-read once per command
 * buffer unless the GPU copy is newer (kat tex-update). Shared textures
 * likewise. Render targets of all three kinds are written back after each
 * command buffer, so re-reading them is safe. */
static bool needs_refresh(const lagfx_task_entry_t *task, const lagfx_vk_iosurface_t *ios,
                          const lagfx_texture_object_t *obj) {
    return obj->storage != LAGFX_STORAGE_PRIVATE
           && !ios->gpu_drawn && ios->upload_gen != task->cmdbuf_gen + 1u;
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
    ios->sum_valid = 0u;
    return ios;
}

/* Read `len` bytes at texture-memory offset `off` through the placement
 * entries (task virtual addresses). */
static bool read_texture_memory(lagfx_protocol_t *p, const lagfx_task_entry_t *task,
                                const lagfx_texture_object_t *obj, uint64_t off,
                                uint8_t *dst, uint64_t len) {
    uint64_t base = 0;
    off += obj->base_offset;
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

static bool guest_map_copy(lagfx_protocol_t *p, const lagfx_task_entry_t *mem_task,
                           struct lagfx_vk_state *vk, lagfx_vk_iosurface_t *ios,
                           const lagfx_texture_object_t *obj, bool to_image);
static bool texture_mapping(lagfx_protocol_t *p, const lagfx_task_entry_t *mem_task,
                            struct lagfx_vk_state *vk, lagfx_vk_iosurface_t *ios,
                            const lagfx_texture_object_t *obj, VkDeviceSize *off_out,
                            VkDeviceSize *len_out);

/* A fast 64-bit checksum of `n` bytes (four independent lanes, so it runs
 * near memory speed); used to tell whether guest memory changed. */
static uint64_t mem_checksum(const uint8_t *b, size_t n) {
    const uint64_t k = 0x9e3779b97f4a7c15ull;
    uint64_t h[4] = { k, k ^ 1u, k ^ 2u, k ^ 3u };
    size_t i = 0;
    for (; i + 32u <= n; i += 32u) {
        uint64_t w[4];
        memcpy(w, b + i, 32);
        for (int l = 0; l < 4; l++) {
            uint64_t x = h[l] ^ w[l];
            h[l] = ((x << 29) | (x >> 35)) * k;
        }
    }
    uint64_t t = 0;
    for (; i < n; i++) t = (t << 8) ^ b[i] ^ (t >> 56);
    uint64_t r = h[0] ^ (h[1] * 3u) ^ (h[2] * 5u) ^ (h[3] * 7u) ^ t ^ (uint64_t)n;
    return (r ^ (r >> 31)) * k;
}

/* The checksum of the texture's mapped guest memory; false when unmapped. */
static bool texture_checksum(lagfx_protocol_t *p, const lagfx_task_entry_t *mem_task,
                             struct lagfx_vk_state *vk, lagfx_vk_iosurface_t *ios,
                             const lagfx_texture_object_t *obj, uint64_t *sum) {
    VkDeviceSize off = 0, len = 0;
    if (getenv("LAGFX_DISABLE_UPLOAD_SKIP")
        || !texture_mapping(p, mem_task, vk, ios, obj, &off, &len))
        return false;
    *sum = mem_checksum(ios->gmap->base + off, (size_t)len);
    return true;
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
    /* the GPU copies straight from guest memory where it can; nothing to
     * copy when the memory still matches the image (written back or
     * uploaded since and unchanged) */
    double t0 = now_ms();
    uint64_t sum = 0;
    bool dump = getenv("LAGFX_DUMP_WB") != NULL;
    /* the checksum only when it can save the copy: an image unchanged since
     * its memory was last checksummed */
    bool summed = !dump && ios->sum_valid && !ios->gpu_drawn
                  && texture_checksum(p, mem_task, vk, ios, obj, &sum);
    if (summed && sum == ios->mem_sum) {
        stats.skipped_uploads++;
        stats.up_ms += now_ms() - t0;
        stats_tick();
        return true;
    }
    if (!dump && guest_map_copy(p, mem_task, vk, ios, obj, true)) {
        /* the image now equals the memory: checksum it unless that's done */
        ios->sum_valid = summed || texture_checksum(p, mem_task, vk, ios, obj, &sum);
        ios->mem_sum = sum;
        stats.up_ms += now_ms() - t0;
        stats.uploads++;
        stats.up_bytes += (unsigned long long)tight * obj->height;
        stats.mapped_uploads++;
        stats_tick();
        return true;
    }
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
    ios->sum_valid = 0u;   /* the image changed; its checksum isn't known */
    free(px);
    stats.uploads++;
    stats.up_bytes += (unsigned long long)tight * obj->height;
    stats.up_ms += now_ms() - t0;
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
    bool ok = lagfx_vk_queue_submit(vk, 1, &si, fence) == VK_SUCCESS
              && vkWaitForFences(vk->device, 1, &fence, VK_TRUE, 2000000000ull) == VK_SUCCESS;
    vkDestroyFence(vk->device, fence, NULL);
    return ok;
}

/* Copy between the image and the texture's guest memory on the GPU: the
 * guest pages are mapped into this process and imported as a buffer
 * (guest_map.c), so neither the CPU nor read/write_memory touch the texels.
 * False when the texture can't be mapped (the caller copies instead). The
 * mapping is kept with the image and checked against the task's page table
 * on each use. */
/* The texture's guest memory mapped into this process (ios->gmap, made or
 * checked against the task's page table here): *off is texel (0,0) in it,
 * *len the bytes from there to the end of the last row. False when the
 * texture can't be mapped. */
static bool texture_mapping(lagfx_protocol_t *p, const lagfx_task_entry_t *mem_task,
                            struct lagfx_vk_state *vk, lagfx_vk_iosurface_t *ios,
                            const lagfx_texture_object_t *obj, VkDeviceSize *off_out,
                            VkDeviceSize *len_out) {
    lagfx_device_t *dev = (lagfx_device_t *)p->dev;
    uint32_t bpp = lagfx_metal_pixel_format_bytes(obj->pixel_format);
    if (!vk->have_host_import || !dev->desc.shell.ram_fd || ios->map_failed
        || getenv("LAGFX_DISABLE_GUEST_MAP"))
        return false;
    if (bpp == 0u || obj->n_place != 1u || obj->bytes_per_row % bpp != 0u
        || obj->pixel_format == 250u || obj->pixel_format == 252u
        || ios->width != obj->width || ios->height != obj->height)
        return false;
    uint64_t start = (obj->place[0].page << 12) + obj->base_offset;
    uint64_t end = start + (uint64_t)obj->bytes_per_row * (obj->height - 1u)
                 + (uint64_t)obj->width * bpp;
    if (end - (obj->place[0].page << 12) > obj->place[0].size) return false;
    uint64_t first = start & ~0xfffull;
    uint32_t npages = (uint32_t)((end - first + 0xfffu) >> 12);
    VkDeviceSize off = start - first;
    if (npages == 0u || npages > 65536u || off % bpp != 0u || off % 4u != 0u) return false;

    uint64_t *gpa = malloc(sizeof(uint64_t) * npages);
    if (!gpa) return false;
    if (!lagfx_task_translate_pages(p, mem_task, first, npages, gpa)) {
        free(gpa);
        return false;
    }
    if (!lagfx_guest_map_same(ios->gmap, npages, gpa)) {
        lagfx_guest_map_destroy(vk, ios->gmap);
        ios->gmap = NULL;
        int *fd = malloc(sizeof(int) * npages);
        uint64_t *fo = malloc(sizeof(uint64_t) * npages);
        bool ok = fd && fo;
        for (uint32_t i = 0; ok && i < npages; i++)
            ok = dev->desc.shell.ram_fd(dev->desc.shell.opaque, gpa[i], &fd[i], &fo[i]);
        if (ok) ios->gmap = lagfx_guest_map_create(vk, npages, gpa, fd, fo);
        free(fd);
        free(fo);
        if (!ios->gmap) ios->map_failed = 1u;
    }
    free(gpa);
    if (!ios->gmap) return false;
    *off_out = off;
    *len_out = end - start;
    return true;
}

static bool guest_map_copy(lagfx_protocol_t *p, const lagfx_task_entry_t *mem_task,
                           struct lagfx_vk_state *vk, lagfx_vk_iosurface_t *ios,
                           const lagfx_texture_object_t *obj, bool to_image) {
    VkDeviceSize off = 0, len = 0;
    uint32_t bpp = lagfx_metal_pixel_format_bytes(obj->pixel_format);
    if (!texture_mapping(p, mem_task, vk, ios, obj, &off, &len)) return false;
    (void)len;

    VkCommandBuffer cb = VK_NULL_HANDLE;
    if (lagfx_vk_cmdbuf_alloc(vk, &cb) != LAGFX_OK) return false;
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cb, &bi);
    VkImageLayout keep = ios->layout;
    VkImageLayout xfer = to_image ? VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                                  : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkImageMemoryBarrier b = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = to_image ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = to_image ? VK_IMAGE_LAYOUT_UNDEFINED : keep, .newLayout = xfer,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = ios->image, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, NULL, 0, NULL, 1, &b);
    VkBufferImageCopy c = {
        .bufferOffset = off, .bufferRowLength = obj->bytes_per_row / bpp,
        .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .imageExtent = { obj->width, obj->height, 1 },
    };
    if (to_image)
        vkCmdCopyBufferToImage(cb, ios->gmap->buffer, ios->image, xfer, 1, &c);
    else
        vkCmdCopyImageToBuffer(cb, ios->image, xfer, ios->gmap->buffer, 1, &c);
    /* an upload leaves the image ready to sample; a write-back puts it back */
    VkImageLayout after = to_image || keep == VK_IMAGE_LAYOUT_UNDEFINED
        ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : keep;
    b.srcAccessMask = to_image ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_TRANSFER_READ_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.oldLayout = xfer;
    b.newLayout = after;
    VkBufferMemoryBarrier hb = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = ios->gmap->buffer, .size = VK_WHOLE_SIZE,
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL,
                         to_image ? 0u : 1u, &hb, 1, &b);
    vkEndCommandBuffer(cb);
    bool ok = submit_wait(vk, cb);
    lagfx_vk_cmdbuf_free(vk, cb);
    if (ok) ios->layout = after;
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
    ios->sum_valid = 0u;
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
    /* the readback helper sizes every format in the table */
    return lagfx_metal_pixel_format_bytes(metal_format);
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
    off += obj->base_offset;
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
                          struct lagfx_vk_state *vk, uint32_t ref, bool any_storage) {
    lagfx_texture_object_t obj;
    const lagfx_task_entry_t *mem_task;
    if (!read_texture_object(p, task, ref, &obj, &mem_task)) return;
    /* Every CPU-visible texture: guest memory is the texture, and the guest
     * reads it without telling us. Shared storage is coherent, so the CPU may
     * read it as soon as the command buffer completes (no synchronize is
     * sent: kat rt-shared-buffer); managed textures and IOSurfaces are read
     * after a synchronize or directly. Private textures stay on the host.
     * The copies cost time until guest memory is mapped into the host. */
    if (!any_storage && obj.storage == LAGFX_STORAGE_PRIVATE) {
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

    size_t tight = (size_t)obj.width * bpp, stride = 0;
    bool ok = true;
    if (!getenv("LAGFX_DUMP_WB") && guest_map_copy(p, mem_task, vk, ios, &obj, false)) {
        stats.mapped_writebacks++;
        /* the memory now equals the image: remember its checksum */
        uint64_t sum = 0;
        ios->sum_valid = 0u;
        if (obj.surface_id != 0u && texture_checksum(p, mem_task, vk, ios, &obj, &sum)) {
            ios->mem_sum = sum;
            ios->sum_valid = 1u;
        }
        goto written;
    }
    lagfx_vk_render_target_t rt;
    if (lagfx_vk_render_target_wrap(ios->image, ios->view, ios->memory, ios->width,
                                    ios->height, ios->format, &rt) != LAGFX_OK)
        return;
    rt.layout = ios->layout;
    uint8_t *px = malloc(tight * obj.height);
    if (!px) return;
    if (lagfx_vk_render_target_readback(vk, &rt, px, tight * obj.height, &stride) != LAGFX_OK) {
        free(px);
        return;
    }
    ios->layout = rt.layout;
    if (stride == 0u) stride = tight;
    dump_image("wb", task, ref, &obj, px, stride);
    for (uint32_t y = 0; y < obj.height && ok; y++)
        ok = write_texture_memory(p, mem_task, &obj, (uint64_t)y * obj.bytes_per_row,
                                  px + (size_t)y * stride, tight);
    free(px);
written:
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
        writeback_one(p, task, vk, task->written_tex[i], false);
    task->n_written_tex = 0;
}

/* CmdSynchronizeResources (0x35) and SynchronizeAndDiscard (0x3e): the guest
 * is about to read `ref` with the CPU (or unwire its pages), so a newer GPU
 * copy must reach guest memory now, whatever the storage mode: shared
 * render targets are otherwise never written back. */
void lagfx_guest_texture_synchronize(lagfx_protocol_t *p, lagfx_task_entry_t *task,
                                     struct lagfx_vk_state *vk, uint32_t ref) {
    if (!p || !task || !vk) return;
    lagfx_resource_entry_t *e = lagfx_resource_lookup(&p->resources, ref, task->id);
    if (!e || e->type != LAGFX_RESOURCE_TYPE_TEXTURE || !e->host_handle) return;
    if (!((lagfx_vk_iosurface_t *)e->host_handle)->gpu_drawn) return;
    writeback_one(p, task, vk, ref, true);
}
#endif /* LAGFX_HAVE_VULKAN */

/* CmdInvalidateResources (0x34): the backing pages of `ref` changed, so guest
 * memory is the truth and the host copy must be read again before its next
 * use. */
void lagfx_guest_texture_invalidate(lagfx_protocol_t *p, uint32_t task_id, uint32_t ref) {
    if (!p) return;
    lagfx_resource_entry_t *e = lagfx_resource_lookup(&p->resources, ref, task_id);
    if (!e || e->type != LAGFX_RESOURCE_TYPE_TEXTURE) return;
    e->backing_dirty = 1u;
#ifdef LAGFX_HAVE_VULKAN
    if (e->host_handle) ((lagfx_vk_iosurface_t *)e->host_handle)->gpu_drawn = 0u;
#endif
}
