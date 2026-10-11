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

    /* The dimensions are stored twice; disagreement means this isn't the
     * layout we know. */
    if (out->width == 0u || out->height == 0u || out->width > 16384u || out->height > 16384u)
        return false;
    if (rd32(b + 0x58) != out->width || rd32(b + 0x5c) != out->height) return false;
    if (out->pixel_format == 0u || out->storage > LAGFX_STORAGE_PRIVATE) return false;
    if (out->bytes_per_row < out->width) return false;

    for (uint32_t e = 0; e < 2u; e++) {
        uint64_t size = rd64(b + e * 16u);
        uint64_t page = rd64(b + e * 16u + 8u) & 0xffffffffull;
        if (size == 0u || page == 0u) continue;
        out->place[out->n_place].size = size;
        out->place[out->n_place].page = page;
        out->n_place++;
    }
    return true;
}
