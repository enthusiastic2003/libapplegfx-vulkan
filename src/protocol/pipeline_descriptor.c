/*
 * libapplegfx-vulkan — serialized render pipeline descriptor: colour attachments
 * src/protocol/pipeline_descriptor.c
 *
 * SPDX-License-Identifier: MIT
 *
 * See pipeline_descriptor.h for the encoding.
 */

#include "pipeline_descriptor.h"

#include <stdbool.h>
#include <string.h>

#define MAX_FIELDS 24u

static uint32_t rd32(const uint8_t *b) {
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16)
         | ((uint32_t)b[3] << 24);
}

typedef struct {
    uint32_t value[32];   /* by key, little-endian, up to 4 bytes */
    uint32_t present;     /* bit per key */
} record_t;

/* Parse one record at b[pos]; returns its end offset, or 0 if malformed. */
static size_t parse_record(const uint8_t *b, size_t len, size_t pos, record_t *r) {
    memset(r, 0, sizeof *r);
    if (pos >= len) return 0u;
    uint32_t nf = b[pos++];
    if (nf == 0u || nf > MAX_FIELDS) return 0u;
    /* keys are unique but not necessarily ascending (top level: 8, 1, 2) */
    for (uint32_t f = 0; f < nf; f++) {
        if (pos + 2u > len) return 0u;
        uint32_t key = b[pos], size = b[pos + 1u];
        if (key >= 32u || ((r->present >> key) & 1u) || pos + 2u + size > len) return 0u;
        uint32_t v = 0;
        for (uint32_t k = 0; k < size && k < 4u; k++) v |= (uint32_t)b[pos + 2u + k] << (8u * k);
        r->value[key] = v;
        r->present |= 1u << key;
        pos += 2u + size;
    }
    return pos;
}

static uint32_t field(const record_t *r, uint32_t key, uint32_t dflt) {
    return (r->present >> key) & 1u ? r->value[key] : dflt;
}

uint32_t lagfx_parse_pso_color_attachments(const uint8_t *blob, size_t len,
                                           lagfx_pso_color_attachment_t *out, uint32_t cap) {
    if (!blob || !out || cap == 0u || len < LAGFX_PSO_HEADER_LEN + 1u) return 0u;
    if (rd32(blob) != 0x0eu) return 0u;
    record_t top;
    if (!parse_record(blob, len, LAGFX_PSO_HEADER_LEN, &top) || !((top.present >> 8) & 1u))
        return 0u;
    size_t arr = LAGFX_PSO_HEADER_LEN + top.value[8];
    if (arr + 8u > len) return 0u;
    uint32_t n = rd32(blob + arr);
    if (n == 0u || n > 8u || arr + 4u + 4u * n > len || rd32(blob + arr + 4u) != 4u + 4u * n)
        return 0u;
    uint32_t got = 0;
    for (uint32_t k = 0; k < n && got < cap; k++) {
        record_t r;
        if (!parse_record(blob, len, arr + rd32(blob + arr + 4u + 4u * k), &r)) return 0u;
        lagfx_pso_color_attachment_t *a = &out[got++];
        a->index        = (uint8_t)field(&r, 0, 0);
        a->pixel_format = field(&r, 1, 0);
        a->blending     = (uint8_t)field(&r, 2, 0);
        a->src_rgb      = (uint8_t)field(&r, 3, 1);
        a->dst_rgb      = (uint8_t)field(&r, 4, 0);
        a->rgb_op       = (uint8_t)field(&r, 5, 0);
        a->src_alpha    = (uint8_t)field(&r, 6, 1);
        a->dst_alpha    = (uint8_t)field(&r, 7, 0);
        a->alpha_op     = (uint8_t)field(&r, 8, 0);
        a->write_mask   = (uint8_t)field(&r, 9, 15);
    }
    return got;
}

bool lagfx_parse_pso_functions(const uint8_t *blob, size_t len, uint32_t *vertex_fn,
                               uint32_t *fragment_fn) {
    if (!blob || len < LAGFX_PSO_HEADER_LEN + 1u || rd32(blob) != 0x0eu) return false;
    record_t top;
    if (!parse_record(blob, len, LAGFX_PSO_HEADER_LEN, &top) || !((top.present >> 1) & 1u))
        return false;
    if (vertex_fn) *vertex_fn = top.value[1];
    if (fragment_fn) *fragment_fn = field(&top, 2, 0);
    return true;
}
