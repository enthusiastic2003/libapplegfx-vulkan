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

    out->width         = rd32(b + 0x3c);
    out->height        = rd32(b + 0x40);
    out->depth         = rd32(b + 0x44);
    out->bytes_per_row = rd32(b + 0x34);
    out->usage         = b[0x55];
    out->pixel_format  = rd16(b + 0x56);
    out->storage       = (rd16(b + 0x6a) >> 4) & 0xfu;
    out->texture_type  = b[0x54] >= 0x40u ? b[0x54] - 0x40u : 2u;
    out->mip_levels    = rd16(b + 0x64);
    out->sample_count  = rd16(b + 0x66);
    out->array_length  = rd16(b + 0x68);

    /* The dimensions are stored twice; disagreement means this isn't the
     * layout we know. */
    if (out->width == 0u || out->height == 0u || out->width > 16384u || out->height > 16384u)
        return false;
    if (rd32(b + 0x58) != out->width || rd32(b + 0x5c) != out->height) return false;
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
    if (!b || !out || len < LAGFX_IOSURFACE_RECORD_LEN) return false;
    memset(out, 0, sizeof(*out));
    out->size              = rd64(b + 0x00);
    out->page              = rd64(b + 0x08) & 0xffffffffull;
    out->fourcc            = rd32(b + 0x0c);
    out->width             = rd32(b + 0x18);
    out->height            = rd32(b + 0x1c);
    out->bytes_per_row     = rd16(b + 0x20);
    out->bytes_per_element = b[0x23];
    if (out->size == 0u || out->page == 0u || out->width == 0u || out->height == 0u)
        return false;
    if (out->bytes_per_row < out->width) return false;
    return (uint64_t)out->bytes_per_row * out->height <= out->size;
}

bool lagfx_texture_object_attach_surface(lagfx_texture_object_t *t,
                                         const lagfx_iosurface_record_t *s) {
    if (!t || !s) return false;
    /* A texture may cover part of the surface, never more. */
    if (t->width > s->width || t->height > s->height) return false;
    t->bytes_per_row = s->bytes_per_row;
    t->n_place = 1;
    t->place[0].size = s->size;
    t->place[0].page = s->page;
    return true;
}
