/*
 * libapplegfx-vulkan — serialized MTLVertexDescriptor decoder (unit)
 * tests/vertex-descriptor-unit.c
 *
 * SPDX-License-Identifier: MIT
 *
 * Synthetic blobs in the record-array encoding (see
 * src/protocol/vertex_descriptor.h). Built by hand, not captured.
 */

#include "protocol/vertex_descriptor.h"

#include <stdio.h>
#include <string.h>

static int failures;

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } \
} while (0)

typedef struct { uint8_t b[512]; uint32_t n; } blob_t;

static void put8(blob_t *bl, uint8_t v) { bl->b[bl->n++] = v; }
static void put32(blob_t *bl, uint32_t v) {
    for (int i = 0; i < 4; i++) put8(bl, (uint8_t)(v >> (8 * i)));
}
static void set32(blob_t *bl, uint32_t at, uint32_t v) {
    for (int i = 0; i < 4; i++) bl->b[at + i] = (uint8_t)(v >> (8 * i));
}

/* Append an array of records; fields[r] lists {key, value} pairs, zeros omitted. */
typedef struct { uint32_t key, value; } field_t;
static void put_array(blob_t *bl, const field_t fields[][4], const int *nfields, int nrec) {
    uint32_t start = bl->n;
    put32(bl, (uint32_t)nrec);
    for (int r = 0; r < nrec; r++) put32(bl, 0);
    for (int r = 0; r < nrec; r++) {
        set32(bl, start + 4u + 4u * (uint32_t)r, bl->n - start);
        int present = 0;
        for (int f = 0; f < nfields[r]; f++) if (f == 0 || fields[r][f].value) present++;
        put8(bl, (uint8_t)present);
        for (int f = 0; f < nfields[r]; f++) {
            if (f > 0 && fields[r][f].value == 0) continue;   /* omitted zero */
            put8(bl, (uint8_t)fields[r][f].key);
            put8(bl, 4);
            put32(bl, fields[r][f].value);
        }
    }
}

static void put_header(blob_t *bl) {
    put32(bl, 0x0e); put32(bl, 0); put32(bl, 0x40); put32(bl, 0);
    /* a top-level record with function refs, then padding text */
    put8(bl, 2); put8(bl, 1); put8(bl, 4); put32(bl, 0x12); put8(bl, 2); put8(bl, 4); put32(bl, 0x11);
    const char *label = "test.pipeline";
    for (const char *c = label; *c; c++) put8(bl, (uint8_t)*c);
    put8(bl, 0); put8(bl, 0);
}

static void test_interleaved_buffer0(void) {
    blob_t bl = {0};
    put_header(&bl);
    /* attrs: float2 @0, float2 @8, float @16 — all buffer 0 (bufidx omitted) */
    const field_t attrs[][4] = {
        {{0, 0}, {1, 29}, {2, 0}, {3, 0}},
        {{0, 1}, {1, 29}, {2, 8}, {3, 0}},
        {{0, 2}, {1, 28}, {2, 16}, {3, 0}},
    };
    const int na[] = {4, 4, 4};
    put_array(&bl, attrs, na, 3);
    const field_t layouts[][4] = {{{0, 0}, {3, 24}}, {{0, 0}, {3, 24}}};
    const int nl[] = {2, 2};
    put_array(&bl, layouts, nl, 2);
    /* color attachments: pixel format 80 must not be taken as a vertex format */
    const field_t color[][4] = {{{0, 0}, {1, 80}}};
    const int nc[] = {2};
    put_array(&bl, color, nc, 1);
    set32(&bl, 4, bl.n);

    lagfx_vtxdesc_attr_t a[8];
    lagfx_vtxdesc_layout_t l[8];
    uint32_t nlay = 99;
    uint32_t n = lagfx_parse_vertex_descriptor(bl.b, bl.n, a, 8, l, 8, &nlay);
    CHECK(n == 3);
    CHECK(a[0].index == 0 && a[0].fmt == 29 && a[0].off == 0 && a[0].bufidx == 0);
    CHECK(a[1].index == 1 && a[1].fmt == 29 && a[1].off == 8 && a[1].bufidx == 0);
    CHECK(a[2].index == 2 && a[2].fmt == 28 && a[2].off == 16 && a[2].bufidx == 0);
    CHECK(nlay == 1);   /* duplicate layout records collapse */
    CHECK(l[0].index == 0 && l[0].stride == 24);
}

static void test_buffer1_unsorted(void) {
    blob_t bl = {0};
    put_header(&bl);
    /* serialized out of index order; attribute 0 at offset 0 */
    const field_t attrs[][4] = {
        {{0, 2}, {1, 29}, {2, 24}, {3, 1}},
        {{0, 0}, {1, 31}, {2, 0}, {3, 1}},
        {{0, 3}, {1, 9}, {2, 32}, {3, 1}},
        {{0, 1}, {1, 29}, {2, 16}, {3, 1}},
    };
    const int na[] = {4, 4, 4, 4};
    put_array(&bl, attrs, na, 4);
    const field_t layouts[][4] = {{{0, 1}, {3, 48}}};
    const int nl[] = {2};
    put_array(&bl, layouts, nl, 1);
    set32(&bl, 4, bl.n);

    lagfx_vtxdesc_attr_t a[8];
    lagfx_vtxdesc_layout_t l[8];
    uint32_t nlay = 0;
    uint32_t n = lagfx_parse_vertex_descriptor(bl.b, bl.n, a, 8, l, 8, &nlay);
    CHECK(n == 4);
    for (uint32_t i = 0; i < n; i++) CHECK(a[i].index == i && a[i].bufidx == 1);
    CHECK(a[0].fmt == 31 && a[0].off == 0);
    CHECK(a[3].fmt == 9 && a[3].off == 32);
    CHECK(nlay == 1 && l[0].index == 1 && l[0].stride == 48);
}

static void test_no_vertex_descriptor(void) {
    blob_t bl = {0};
    put_header(&bl);
    const field_t color[][4] = {{{0, 0}, {1, 80}}};
    const int nc[] = {2};
    put_array(&bl, color, nc, 1);
    set32(&bl, 4, bl.n);

    lagfx_vtxdesc_attr_t a[8];
    uint32_t nlay = 99;
    CHECK(lagfx_parse_vertex_descriptor(bl.b, bl.n, a, 8, NULL, 0, &nlay) == 0);
    CHECK(nlay == 0);
}

static void test_truncated(void) {
    blob_t bl = {0};
    put_header(&bl);
    const field_t attrs[][4] = {{{0, 0}, {1, 29}, {2, 0}, {3, 0}}, {{0, 1}, {1, 29}, {2, 8}, {3, 0}}};
    const int na[] = {4, 4};
    put_array(&bl, attrs, na, 2);
    lagfx_vtxdesc_attr_t a[8];
    /* cut into the second record: no partial decode */
    CHECK(lagfx_parse_vertex_descriptor(bl.b, bl.n - 3, a, 8, NULL, 0, NULL) == 0);
}

int main(void) {
    test_interleaved_buffer0();
    test_buffer1_unsorted();
    test_no_vertex_descriptor();
    test_truncated();
    if (failures) return 1;
    printf("vertex-descriptor-unit: OK\n");
    return 0;
}
