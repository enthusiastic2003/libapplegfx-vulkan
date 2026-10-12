/*
 * libapplegfx-vulkan — depth-stencil state object decoder
 * src/protocol/depth_stencil.c
 *
 * SPDX-License-Identifier: MIT
 */
#include "depth_stencil.h"

#include <string.h>

static uint32_t rd32(const uint8_t *b) {
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

void lagfx_depth_stencil_default(lagfx_depth_stencil_t *out) {
    memset(out, 0, sizeof *out);
    out->depth_compare = 7u;   /* Always */
    out->front = out->back = (lagfx_stencil_face_t){ 7u, 0u, 0u, 0u, 0xffu, 0xffu };
}

static lagfx_stencil_face_t face(uint32_t word, uint32_t read_mask, uint32_t write_mask) {
    return (lagfx_stencil_face_t){ word & 7u, (word >> 3) & 7u, (word >> 6) & 7u, (word >> 9) & 7u,
                                   read_mask & 0xffu, write_mask & 0xffu };
}

bool lagfx_parse_depth_stencil_object(const uint8_t *b, size_t len, lagfx_depth_stencil_t *out) {
    if (!b || !out || len < LAGFX_DEPTH_STENCIL_OBJECT_LEN) return false;
    if (rd32(b + 0x04) != LAGFX_DEPTH_STENCIL_OBJECT_LEN) return false;
    lagfx_depth_stencil_default(out);
    uint32_t flags = rd32(b + 0x0c);
    out->depth_compare = flags & 7u;
    out->depth_write = (flags >> 3) & 1u;
    if (flags & 0x10u) out->front = face(rd32(b + 0x10), rd32(b + 0x14), rd32(b + 0x18));
    if (flags & 0x20u) out->back = face(rd32(b + 0x1c), rd32(b + 0x20), rd32(b + 0x24));
    return true;
}
