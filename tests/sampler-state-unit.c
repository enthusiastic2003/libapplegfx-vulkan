/*
 * libapplegfx-vulkan — sampler state decoders (unit)
 * tests/sampler-state-unit.c
 *
 * SPDX-License-Identifier: MIT
 *
 * Values are what tests/guest/kat.m `sampler-probe` produced for samplers
 * with the settings named in each case (see src/protocol/sampler_state.h).
 */

#include "protocol/sampler_state.h"

#include <stdio.h>
#include <string.h>

static int failures;

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } \
} while (0)

static void put32(uint8_t *b, size_t at, uint32_t v) {
    for (int i = 0; i < 4; i++) b[at + i] = (uint8_t)(v >> (8 * i));
}

static void make_object(uint8_t *b, uint32_t word, float lod_min, float lod_max) {
    memset(b, 0, LAGFX_SAMPLER_OBJECT_LEN);
    put32(b, 0x00, 3);
    put32(b, 0x04, LAGFX_SAMPLER_OBJECT_LEN);
    put32(b, 0x08, 0x27);
    put32(b, 0x0c, word);
    uint32_t u;
    memcpy(&u, &lod_min, 4); put32(b, 0x14, u);
    memcpy(&u, &lod_max, 4); put32(b, 0x18, u);
}

int main(void) {
    uint8_t b[LAGFX_SAMPLER_OBJECT_LEN];
    lagfx_sampler_desc_t d;

    /* MTLSamplerDescriptor defaults */
    make_object(b, 0x84000000u, 0.0f, 3.402823466e38f);
    CHECK(lagfx_parse_sampler_object(b, sizeof b, &d));
    CHECK(d.min_filter == 0 && d.mag_filter == 0 && d.mip_filter == 0);
    CHECK(d.address[0] == LAGFX_SAMPLER_CLAMP_TO_EDGE && d.address[1] == LAGFX_SAMPLER_CLAMP_TO_EDGE);
    CHECK(d.normalized && d.max_anisotropy == 1 && d.lod_min == 0.0f && d.lod_max > 1e38f);

    make_object(b, 0x84000005u, 0, 1);       /* min + mag linear */
    CHECK(lagfx_parse_sampler_object(b, sizeof b, &d) && d.min_filter == 1 && d.mag_filter == 1);
    make_object(b, 0x84000004u, 0, 1);       /* mag linear only */
    CHECK(lagfx_parse_sampler_object(b, sizeof b, &d) && d.min_filter == 0 && d.mag_filter == 1);
    make_object(b, 0x84000200u, 0, 1);       /* s repeat */
    CHECK(lagfx_parse_sampler_object(b, sizeof b, &d) && d.address[0] == LAGFX_SAMPLER_REPEAT
          && d.address[1] == LAGFX_SAMPLER_CLAMP_TO_EDGE);
    make_object(b, 0x84003000u, 0, 1);       /* t mirror repeat */
    CHECK(lagfx_parse_sampler_object(b, sizeof b, &d) && d.address[1] == LAGFX_SAMPLER_MIRROR_REPEAT);
    make_object(b, 0x04000000u, 0, 1);       /* unnormalized */
    CHECK(lagfx_parse_sampler_object(b, sizeof b, &d) && !d.normalized);
    make_object(b, 0x86000000u, 0, 1);       /* mip linear */
    CHECK(lagfx_parse_sampler_object(b, sizeof b, &d) && d.mip_filter == 2);
    make_object(b, 0x90000000u, 0, 1);       /* anisotropy 4 */
    CHECK(lagfx_parse_sampler_object(b, sizeof b, &d) && d.max_anisotropy == 4);
    make_object(b, 0x84000000u, 1.0f, 5.0f); /* lod clamp 1..5 */
    CHECK(lagfx_parse_sampler_object(b, sizeof b, &d) && d.lod_min == 1.0f && d.lod_max == 5.0f);
    make_object(b, 0x84200500u, 0, 1);       /* s clamp to border, opaque white */
    CHECK(lagfx_parse_sampler_object(b, sizeof b, &d)
          && d.address[0] == LAGFX_SAMPLER_CLAMP_TO_BORDER && d.border_color == 2);

    /* not a sampler record */
    make_object(b, 0x84000000u, 0, 1);
    put32(b, 0x00, 4);
    CHECK(!lagfx_parse_sampler_object(b, sizeof b, &d));
    CHECK(!lagfx_parse_sampler_object(b, 0x10, &d));

    /* constexpr samplers */
    lagfx_decode_constexpr_sampler(0x807bff0000080049ull, &d);   /* filter::nearest */
    CHECK(d.min_filter == 0 && d.mag_filter == 0 && d.mip_filter == 0 && d.normalized);
    CHECK(d.address[0] == LAGFX_SAMPLER_CLAMP_TO_EDGE && d.address[2] == LAGFX_SAMPLER_CLAMP_TO_EDGE);
    CHECK(d.max_anisotropy == 1 && d.lod_min == 0.0f && d.lod_max == 65504.0f);
    lagfx_decode_constexpr_sampler(0x807bff0000080a49ull, &d);   /* filter::linear */
    CHECK(d.min_filter == 1 && d.mag_filter == 1);
    lagfx_decode_constexpr_sampler(0x807bff0000080249ull, &d);   /* mag linear, min nearest */
    CHECK(d.min_filter == 0 && d.mag_filter == 1);
    lagfx_decode_constexpr_sampler(0x807bff0000080092ull, &d);   /* address::repeat */
    CHECK(d.address[0] == LAGFX_SAMPLER_REPEAT && d.address[1] == LAGFX_SAMPLER_REPEAT
          && d.address[2] == LAGFX_SAMPLER_REPEAT);
    lagfx_decode_constexpr_sampler(0x807bff00000800dbull, &d);   /* mirrored_repeat */
    CHECK(d.address[0] == LAGFX_SAMPLER_MIRROR_REPEAT && d.address[2] == LAGFX_SAMPLER_MIRROR_REPEAT);
    lagfx_decode_constexpr_sampler(0x807bff000008004aull, &d);   /* s repeat, t clamp_to_edge */
    CHECK(d.address[0] == LAGFX_SAMPLER_REPEAT && d.address[1] == LAGFX_SAMPLER_CLAMP_TO_EDGE);
    lagfx_decode_constexpr_sampler(0x807bff0000088049ull, &d);   /* coord::pixel */
    CHECK(!d.normalized);
    lagfx_decode_constexpr_sampler(0x807bff0000080000ull, &d);   /* clamp_to_zero */
    CHECK(d.address[0] == LAGFX_SAMPLER_CLAMP_TO_BORDER && d.border_color == 0);
    lagfx_decode_constexpr_sampler(0x807bff0000084049ull, &d);   /* mip_filter::linear */
    CHECK(d.mip_filter == 2);
    lagfx_decode_constexpr_sampler(0x807bff0000380049ull, &d);   /* max_anisotropy(4) */
    CHECK(d.max_anisotropy == 4);
    lagfx_decode_constexpr_sampler(0x827bff0000080000ull, &d);   /* clamp_to_border, opaque_white */
    CHECK(d.address[0] == LAGFX_SAMPLER_CLAMP_TO_BORDER && d.border_color == 2);

    if (failures) return 1;
    printf("sampler-state-unit: OK\n");
    return 0;
}
