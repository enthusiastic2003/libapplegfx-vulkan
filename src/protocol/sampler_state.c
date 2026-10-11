/*
 * libapplegfx-vulkan — sampler state decoders
 * src/protocol/sampler_state.c
 *
 * SPDX-License-Identifier: MIT
 *
 * See sampler_state.h for both layouts.
 */

#include "sampler_state.h"

#include <math.h>
#include <string.h>

static uint32_t rd32(const uint8_t *b) {
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

static float rdf32(const uint8_t *b) {
    uint32_t u = rd32(b);
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

static float half_to_float(uint16_t h) {
    uint32_t sign = h >> 15, exp = (h >> 10) & 0x1fu, mant = h & 0x3ffu;
    float v;
    if (exp == 0u) v = ldexpf((float)mant, -24);
    else if (exp == 31u) v = mant ? NAN : INFINITY;
    else v = ldexpf((float)(mant | 0x400u), (int)exp - 25);
    return sign ? -v : v;
}

bool lagfx_parse_sampler_object(const uint8_t *b, size_t len, lagfx_sampler_desc_t *out) {
    if (!b || !out || len < LAGFX_SAMPLER_OBJECT_LEN) return false;
    if (rd32(b) != 3u || rd32(b + 4) != LAGFX_SAMPLER_OBJECT_LEN) return false;
    uint32_t w = rd32(b + 0x0c);
    memset(out, 0, sizeof *out);
    out->min_filter     = (uint8_t)(w & 3u);
    out->mag_filter     = (uint8_t)((w >> 2) & 3u);
    out->address[0]     = (uint8_t)((w >> 8) & 0xfu);
    out->address[1]     = (uint8_t)((w >> 12) & 0xfu);
    out->address[2]     = (uint8_t)((w >> 16) & 0xfu);
    out->border_color   = (uint8_t)((w >> 20) & 3u);
    out->mip_filter     = (uint8_t)((w >> 24) & 3u);
    out->max_anisotropy = (uint8_t)((w >> 26) & 0x1fu);
    out->normalized     = (w >> 31) != 0u;
    out->lod_min        = rdf32(b + 0x14);
    out->lod_max        = rdf32(b + 0x18);
    if (out->max_anisotropy == 0u) out->max_anisotropy = 1u;
    for (int i = 0; i < 3; i++)
        if (out->address[i] > LAGFX_SAMPLER_CLAMP_TO_BORDER) return false;
    return out->min_filter <= 1u && out->mag_filter <= 1u && out->mip_filter <= 2u;
}

void lagfx_decode_constexpr_sampler(uint64_t bits, lagfx_sampler_desc_t *out) {
    static const uint8_t kAddress[8] = {
        LAGFX_SAMPLER_CLAMP_TO_BORDER,   /* clamp_to_zero / clamp_to_border */
        LAGFX_SAMPLER_CLAMP_TO_EDGE,
        LAGFX_SAMPLER_REPEAT,
        LAGFX_SAMPLER_MIRROR_REPEAT,
        LAGFX_SAMPLER_MIRROR_CLAMP_TO_EDGE,
        LAGFX_SAMPLER_CLAMP_TO_EDGE, LAGFX_SAMPLER_CLAMP_TO_EDGE, LAGFX_SAMPLER_CLAMP_TO_EDGE,
    };
    memset(out, 0, sizeof *out);
    for (int i = 0; i < 3; i++) out->address[i] = kAddress[(bits >> (3 * i)) & 7u];
    out->mag_filter     = (uint8_t)((bits >> 9) & 1u);
    out->min_filter     = (uint8_t)((bits >> 11) & 1u);
    out->mip_filter     = (uint8_t)((bits >> 13) & 3u);
    out->normalized     = ((bits >> 15) & 1u) == 0u;
    out->max_anisotropy = (uint8_t)((((bits >> 19) & 0xfu) + 1u) / 2u);
    if (out->max_anisotropy == 0u) out->max_anisotropy = 1u;
    out->lod_min        = half_to_float((uint16_t)(bits >> 24));
    out->lod_max        = half_to_float((uint16_t)(bits >> 40));
    out->border_color   = (uint8_t)((bits >> 56) & 3u);
}
