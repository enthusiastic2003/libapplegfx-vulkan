/*
 * libapplegfx-vulkan — texture object decoder
 * src/protocol/texture_object.c
 *
 * SPDX-License-Identifier: MIT
 *
 * See texture_object.h for the layout.
 */

#include "texture_object.h"

#include <string.h>

static uint32_t rd16(const uint8_t *b) { return (uint32_t)b[0] | ((uint32_t)b[1] << 8); }
static uint32_t rd32(const uint8_t *b) { return rd16(b) | (rd16(b + 2) << 16); }
static uint64_t rd64(const uint8_t *b) { return (uint64_t)rd32(b) | ((uint64_t)rd32(b + 4) << 32); }

bool lagfx_parse_texture_object(const uint8_t *b, size_t len, lagfx_texture_object_t *out) {
    if (!b || !out || len < LAGFX_TEXTURE_OBJECT_MIN_LEN) return false;
    memset(out, 0, sizeof(*out));

    /* Level 0's record; the descriptor follows the last level's record, so
     * it is found by its repeated dimensions and (mipmapped) level count. */
    uint32_t width = rd32(b + 0x3c), height = rd32(b + 0x40);
    if (width == 0u || height == 0u || width > 16384u || height > 16384u) return false;
    const uint8_t *d = NULL;
    for (uint32_t n = 1; n <= LAGFX_TEXTURE_MAX_LEVELS; n++) {
        size_t at = LAGFX_TEXTURE_LEVEL0 + (size_t)LAGFX_TEXTURE_LEVEL_LEN * n;
        if (at + LAGFX_TEXTURE_LEVEL_LEN > len) break;
        const uint8_t *c = b + at;
        if (rd32(c + 0x10) != width || rd32(c + 0x14) != height || c[0x0c] < 0x40u) continue;
        if (n > 1u && rd16(c + 0x1c) != n) continue;
        d = c;
        break;
    }
    if (!d) return false;

    out->width         = width;
    out->height        = height;
    out->depth         = rd32(b + 0x44);
    out->bytes_per_row = rd32(b + 0x34);
    out->usage         = d[0x0d];
    out->pixel_format  = rd16(d + 0x0e);
    out->storage       = (rd16(d + 0x22) >> 4) & 0xfu;
    out->texture_type  = d[0x0c] - 0x40u;
    out->mip_levels    = rd16(d + 0x1c);
    out->sample_count  = rd16(d + 0x1e);
    out->array_length  = rd16(d + 0x20);

    if (out->pixel_format == 0u || out->storage > LAGFX_STORAGE_PRIVATE) return false;
    if (out->bytes_per_row < out->width) return false;

    uint64_t size = rd64(b + 0x00), page = rd64(b + 0x08) & 0xffffffffull;
    if (size != 0u && page != 0u) {
        out->place[0].size = size;
        out->place[0].page = page;
        out->n_place = 1;
    }
    /* Textures sub-allocated from a larger allocation (Core Animation's text
     * masks) start part-way into it. */
    out->base_offset = rd64(b + 0x10);
    return true;
}

bool lagfx_parse_iosurface_texture_object(const uint8_t *b, size_t len,
                                          lagfx_texture_object_t *out) {
    if (!b || !out || len < LAGFX_IOSURFACE_TEXTURE_MIN_LEN) return false;
    memset(out, 0, sizeof(*out));
    out->surface_id   = rd32(b + 0x00);
    out->usage        = b[0x15];
    out->pixel_format = rd16(b + 0x16);
    out->width        = rd32(b + 0x18);
    out->height       = rd32(b + 0x1c);
    out->depth        = rd32(b + 0x20);
    out->storage      = (rd16(b + 0x2a) >> 4) & 0xfu;
    out->texture_type = b[0x14] >= 0x40u ? b[0x14] - 0x40u : 2u;
    out->mip_levels   = rd16(b + 0x24);
    out->sample_count = rd16(b + 0x26);
    out->array_length = rd16(b + 0x28);
    /* a u16: the serializer leaves the slot's upper half unwritten, so it
     * holds whatever was in the ring before (public PVG implementation,
     * ops/backed_texture.rs; the wide descriptor form writes all four) */
    out->plane        = rd16(b + 0x34);
    if (out->surface_id == 0u || (b[0x14] & 0xf0u) != 0x40u) return false;
    if (out->width == 0u || out->height == 0u || out->width > 16384u || out->height > 16384u)
        return false;
    if (out->pixel_format == 0u || out->storage > LAGFX_STORAGE_PRIVATE) return false;
    return true;
}

bool lagfx_parse_buffer_texture_object(const uint8_t *b, size_t len, lagfx_texture_object_t *out) {
    if (!b || !out || len < LAGFX_BUFFER_TEXTURE_MIN_LEN) return false;
    memset(out, 0, sizeof(*out));
    out->buffer_ref    = rd32(b + 0x0c);
    out->base_offset   = rd64(b + 0x10);
    uint64_t bpr       = rd64(b + 0x18);
    out->usage         = b[0x21];
    out->pixel_format  = rd16(b + 0x22);
    out->width         = rd32(b + 0x24);
    out->height        = rd32(b + 0x28);
    out->depth         = rd32(b + 0x2c);
    out->texture_type  = b[0x20] >= 0x40u ? b[0x20] - 0x40u : 2u;
    out->mip_levels    = rd16(b + 0x30);
    out->sample_count  = rd16(b + 0x32);
    out->array_length  = rd16(b + 0x34);
    out->storage       = (rd16(b + 0x36) >> 4) & 0xfu;
    if (out->buffer_ref == 0u || (b[0x20] & 0xf0u) != 0x40u) return false;
    if (out->width == 0u || out->height == 0u || out->width > 16384u || out->height > 16384u)
        return false;
    if (out->pixel_format == 0u || out->storage > LAGFX_STORAGE_PRIVATE) return false;
    if (bpr < out->width || bpr > 0xffffffffull) return false;
    out->bytes_per_row = (uint32_t)bpr;
    return true;
}

bool lagfx_parse_iosurface_record(const uint8_t *b, size_t len, lagfx_iosurface_record_t *out) {
    if (!b || !out || len < LAGFX_IOSURFACE_RECORD_MIN_LEN) return false;
    memset(out, 0, sizeof(*out));
    out->size   = rd64(b + 0x00);
    out->page   = rd64(b + 0x08) & 0xffffffffull;
    out->fourcc = rd32(b + 0x0c);
    uint32_t count = rd32(b + 0x10);
    out->plane_count = count == 0u ? 1u : count;
    if (out->plane_count > LAGFX_IOSURFACE_MAX_PLANES) return false;
    for (uint32_t i = 0; i < out->plane_count; i++) {
        const uint8_t *e = b + 0x14u + 0x10u * i;
        if ((size_t)(e + 0x10 - b) > len) return false;
        lagfx_iosurface_plane_t *pl = &out->planes[i];
        pl->offset            = rd32(e + 0x0);
        pl->width             = rd32(e + 0x4);
        pl->height            = rd32(e + 0x8);
        pl->bytes_per_row     = rd16(e + 0xc);
        pl->bytes_per_element = e[0xf];
        if (pl->width == 0u || pl->height == 0u || pl->bytes_per_row < pl->width) return false;
        if ((uint64_t)pl->offset + (uint64_t)pl->bytes_per_row * pl->height > out->size)
            return false;
    }
    out->width             = out->planes[0].width;
    out->height            = out->planes[0].height;
    out->bytes_per_row     = out->planes[0].bytes_per_row;
    out->bytes_per_element = out->planes[0].bytes_per_element;
    return out->size != 0u && out->page != 0u;
}

bool lagfx_texture_object_attach_surface(lagfx_texture_object_t *t,
                                         const lagfx_iosurface_record_t *s) {
    if (!t || !s || t->plane >= s->plane_count) return false;
    const lagfx_iosurface_plane_t *pl = &s->planes[t->plane];
    /* A texture may cover part of its plane, never more. */
    if (t->width > pl->width || t->height > pl->height) return false;
    t->bytes_per_row = pl->bytes_per_row;
    t->base_offset = pl->offset;
    t->n_place = 1;
    t->place[0].size = s->size;
    t->place[0].page = s->page;
    return true;
}
