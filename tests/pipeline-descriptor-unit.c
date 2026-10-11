/*
 * libapplegfx-vulkan — pipeline colour attachment decoder (unit)
 * tests/pipeline-descriptor-unit.c
 *
 * SPDX-License-Identifier: MIT
 *
 * Blobs have the shape tests/guest/kat.m `blend-probe` pipelines produced
 * (see src/protocol/pipeline_descriptor.h).
 */

#include "protocol/pipeline_descriptor.h"

#include <stdio.h>
#include <string.h>

static int failures;

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } \
} while (0)

static size_t put32(uint8_t *b, size_t at, uint32_t v) {
    for (int i = 0; i < 4; i++) b[at + i] = (uint8_t)(v >> (8 * i));
    return at + 4;
}

static size_t field(uint8_t *b, size_t at, uint8_t key, uint32_t v) {
    b[at] = key;
    b[at + 1] = 4;
    return put32(b, at + 2, v);
}

/* header + top-level record {8: array offset, 1: 2, 2: 1} + 4 pad + array of
 * one attachment with the given extra fields (key, value pairs). */
static size_t make(uint8_t *b, const uint32_t *kv, int n) {
    memset(b, 0, 256);
    size_t at = 16;
    b[at++] = 3;
    size_t arr_off_at = at;
    at = field(b, at, 8, 0);
    at = field(b, at, 1, 2);
    at = field(b, at, 2, 1);
    at += 4;
    put32(b, arr_off_at + 2, (uint32_t)(at - 16));
    size_t arr = at;
    at = put32(b, at, 1);
    at = put32(b, at, 8);
    b[at++] = (uint8_t)(2 + n);
    at = field(b, at, 0, 0);
    at = field(b, at, 1, 80);
    for (int i = 0; i < n; i++) at = field(b, at, (uint8_t)kv[2 * i], kv[2 * i + 1]);
    (void)arr;
    put32(b, 0, 0x0e);
    put32(b, 4, (uint32_t)at);
    put32(b, 8, 3);
    put32(b, 12, (uint32_t)(at - 16));
    return at;
}

int main(void) {
    uint8_t b[256];
    lagfx_pso_color_attachment_t a[8];

    size_t len = make(b, NULL, 0);                               /* blending off */
    CHECK(lagfx_parse_pso_color_attachments(b, len, a, 8) == 1);
    CHECK(a[0].index == 0 && a[0].pixel_format == 80 && a[0].blending == 0);
    CHECK(a[0].src_rgb == 1 && a[0].dst_rgb == 0 && a[0].src_alpha == 1 && a[0].dst_alpha == 0);
    CHECK(a[0].rgb_op == 0 && a[0].alpha_op == 0 && a[0].write_mask == 15);

    const uint32_t over[] = {2, 1, 3, 4, 4, 5};                 /* SourceAlpha / OneMinusSourceAlpha */
    len = make(b, over, 3);
    CHECK(lagfx_parse_pso_color_attachments(b, len, a, 8) == 1);
    CHECK(a[0].blending == 1 && a[0].src_rgb == 4 && a[0].dst_rgb == 5 && a[0].src_alpha == 1);

    const uint32_t alpha[] = {2, 1, 6, 8, 7, 7, 8, 4};          /* alpha factors + Max */
    len = make(b, alpha, 4);
    CHECK(lagfx_parse_pso_color_attachments(b, len, a, 8) == 1);
    CHECK(a[0].src_alpha == 8 && a[0].dst_alpha == 7 && a[0].alpha_op == 4 && a[0].src_rgb == 1);

    const uint32_t mask[] = {9, 9};                              /* Red | Alpha */
    len = make(b, mask, 1);
    CHECK(lagfx_parse_pso_color_attachments(b, len, a, 8) == 1 && a[0].write_mask == 9);

    /* not a pipeline object / truncated */
    len = make(b, NULL, 0);
    b[0] = 0x0d;
    CHECK(lagfx_parse_pso_color_attachments(b, len, a, 8) == 0);
    len = make(b, NULL, 0);
    CHECK(lagfx_parse_pso_color_attachments(b, 30, a, 8) == 0);

    if (failures) return 1;
    printf("pipeline-descriptor-unit: OK\n");
    return 0;
}
