/*
 * libapplegfx-vulkan — serialized MTLVertexDescriptor decoder
 * src/protocol/vertex_descriptor.c
 *
 * SPDX-License-Identifier: MIT
 *
 * See vertex_descriptor.h for the encoding.
 */

#include "vertex_descriptor.h"

#include <stdbool.h>
#include <string.h>

#define MAX_FIELDS 8u
#define MAX_RECORDS 31u

static uint32_t rd32(const uint8_t *b) {
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16)
         | ((uint32_t)b[3] << 24);
}

typedef struct {
    uint32_t value[16];   /* by key; 0 when omitted */
    uint16_t present;     /* bit per key */
} record_t;

/* Parse one record at b[pos]; returns its end offset, or 0 if malformed. */
static uint32_t parse_record(const uint8_t *b, uint32_t len, uint32_t pos, record_t *r) {
    memset(r, 0, sizeof(*r));
    if (pos >= len) return 0u;
    uint32_t nf = b[pos++];
    if (nf == 0u || nf > MAX_FIELDS) return 0u;
    int last_key = -1;
    for (uint32_t f = 0; f < nf; f++) {
        if (pos + 6u > len) return 0u;
        uint32_t key = b[pos], size = b[pos + 1u];
        if (size != 4u || key >= 16u || (int)key <= last_key) return 0u;
        r->value[key] = rd32(b + pos + 2u);
        r->present |= (uint16_t)(1u << key);
        last_key = (int)key;
        pos += 6u;
    }
    if (!(r->present & 1u)) return 0u;   /* every record leads with its index */
    return pos;
}

/* Parse the record array starting at b[pos]. Returns the record count, or 0 if
 * the bytes there aren't a well-formed array. */
static uint32_t parse_array(const uint8_t *b, uint32_t len, uint32_t pos,
                            record_t *recs, uint32_t cap) {
    if (pos + 8u > len) return 0u;
    uint32_t n = rd32(b + pos);
    if (n == 0u || n > MAX_RECORDS || n > cap) return 0u;
    if (pos + 4u + 4u * n > len) return 0u;
    if (rd32(b + pos + 4u) != 4u + 4u * n) return 0u;
    for (uint32_t k = 0; k < n; k++) {
        uint32_t start = pos + rd32(b + pos + 4u + 4u * k);
        uint32_t end = parse_record(b, len, start, &recs[k]);
        if (end == 0u) return 0u;
        if (k + 1u < n && pos + rd32(b + pos + 8u + 4u * k) != end) return 0u;
    }
    return n;
}

static bool is_attribute_array(const record_t *recs, uint32_t n) {
    for (uint32_t k = 0; k < n; k++) {
        if (recs[k].present & ~0xfu) return false;          /* keys 0..3 only */
        if (!(recs[k].present & (1u << 1))) return false;   /* format required */
        uint32_t fmt = recs[k].value[1];
        if (fmt == 0u || fmt > 60u) return false;           /* MTLVertexFormat range */
    }
    return true;
}

static bool is_layout_array(const record_t *recs, uint32_t n) {
    for (uint32_t k = 0; k < n; k++) {
        if (recs[k].present & ~0xfu) return false;
        if (!(recs[k].present & (1u << 3))) return false;   /* stride required */
        uint32_t stride = recs[k].value[3];
        if (stride == 0u || stride > 2048u || (stride & 3u)) return false;
    }
    return true;
}

uint32_t lagfx_parse_vertex_descriptor(const uint8_t *blob, uint32_t len,
                                       lagfx_vtxdesc_attr_t *attrs, uint32_t attr_cap,
                                       lagfx_vtxdesc_layout_t *layouts, uint32_t layout_cap,
                                       uint32_t *n_layouts) {
    if (n_layouts) *n_layouts = 0u;
    if (!blob || !attrs || attr_cap == 0u) return 0u;

    record_t recs[MAX_RECORDS];
    uint32_t na = 0u, nl = 0u;
    uint32_t pos = 16u;   /* skip the blob header */
    for (; pos + 8u <= len && na == 0u; pos++) {
        uint32_t n = parse_array(blob, len, pos, recs, MAX_RECORDS);
        if (n == 0u || !is_attribute_array(recs, n)) continue;
        for (uint32_t k = 0; k < n && na < attr_cap; k++) {
            attrs[na].index  = recs[k].value[0];
            attrs[na].fmt    = recs[k].value[1];
            attrs[na].off    = recs[k].value[2];
            attrs[na].bufidx = recs[k].value[3];
            na++;
        }
    }
    if (na == 0u) return 0u;

    /* Layouts follow the attribute array. */
    for (; pos + 8u <= len && layouts && n_layouts; pos++) {
        uint32_t n = parse_array(blob, len, pos, recs, MAX_RECORDS);
        if (n == 0u || !is_layout_array(recs, n)) continue;
        for (uint32_t k = 0; k < n; k++) {
            uint32_t j = 0;
            for (; j < nl; j++)
                if (layouts[j].index == recs[k].value[0]) break;
            if (j < nl || nl >= layout_cap) continue;
            layouts[nl].index  = recs[k].value[0];
            layouts[nl].stride = recs[k].value[3];
            layouts[nl].present = recs[k].present;
            for (uint32_t q = 0; q < 8u; q++) layouts[nl].value[q] = recs[k].value[q];
            nl++;
        }
        break;
    }
    if (n_layouts) *n_layouts = nl;

    /* Sort attributes by index (insertion; na <= 31). */
    for (uint32_t a = 1; a < na; a++) {
        lagfx_vtxdesc_attr_t t = attrs[a];
        uint32_t j = a;
        while (j > 0u && attrs[j - 1u].index > t.index) { attrs[j] = attrs[j - 1u]; j--; }
        attrs[j] = t;
    }
    return na;
}
