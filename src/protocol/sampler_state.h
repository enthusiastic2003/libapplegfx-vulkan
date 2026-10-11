/*
 * libapplegfx-vulkan — sampler state decoders
 * src/protocol/sampler_state.h
 *
 * SPDX-License-Identifier: MIT
 *
 * Samplers reach the host in two encodings, both decoded into one
 * description using the public MTLSampler* enum values. The layouts were
 * matched against samplers created with known settings
 * (tests/guest/kat.m, `kat sampler-probe`).
 *
 * Sampler state object (resource slot type 0x07, MTLSamplerState):
 *   +0x00  u32  record type (3)
 *   +0x04  u32  record length (0x24)
 *   +0x08  u32  object id
 *   +0x0c  u32  packed state:
 *                 bits 0-1   minFilter      (MTLSamplerMinMagFilter)
 *                 bits 2-3   magFilter
 *                 bits 8-11  sAddressMode   (MTLSamplerAddressMode)
 *                 bits 12-15 tAddressMode
 *                 bits 16-19 rAddressMode
 *                 bits 20-21 borderColor    (MTLSamplerBorderColor)
 *                 bits 24-25 mipFilter      (MTLSamplerMipFilter)
 *                 bits 26-30 maxAnisotropy
 *                 bit  31    normalizedCoordinates
 *   +0x14  f32  lodMinClamp
 *   +0x18  f32  lodMaxClamp
 *
 * `constexpr sampler` in a shader (a constant i64 global in the AIR):
 *   bits 0-2, 3-5, 6-8  s/t/r address: 0 clamp_to_zero / clamp_to_border,
 *                       1 clamp_to_edge, 2 repeat, 3 mirrored_repeat
 *   bit  9              mag_filter linear
 *   bit  11             min_filter linear
 *   bits 13-14          mip_filter (0 none, 1 nearest, 2 linear)
 *   bit  15             coord::pixel
 *   bits 19-22          max_anisotropy n, stored as 2n-1
 *   bits 24-39          lod_clamp min (half float)
 *   bits 40-55          lod_clamp max (half float)
 *   bits 56-57          border_color (0 transparent_black, 1 opaque_black,
 *                       2 opaque_white)
 */
#ifndef LIBAPPLEGFX_PROTOCOL_SAMPLER_STATE_H
#define LIBAPPLEGFX_PROTOCOL_SAMPLER_STATE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* MTLSamplerAddressMode */
#define LAGFX_SAMPLER_CLAMP_TO_EDGE        0u
#define LAGFX_SAMPLER_MIRROR_CLAMP_TO_EDGE 1u
#define LAGFX_SAMPLER_REPEAT               2u
#define LAGFX_SAMPLER_MIRROR_REPEAT        3u
#define LAGFX_SAMPLER_CLAMP_TO_ZERO        4u
#define LAGFX_SAMPLER_CLAMP_TO_BORDER      5u

#define LAGFX_SAMPLER_OBJECT_LEN 0x24u

typedef struct {
    uint8_t min_filter, mag_filter;   /* MTLSamplerMinMagFilter: 0 nearest, 1 linear */
    uint8_t mip_filter;               /* MTLSamplerMipFilter: 0 none, 1 nearest, 2 linear */
    uint8_t address[3];               /* s, t, r: MTLSamplerAddressMode */
    uint8_t border_color;             /* MTLSamplerBorderColor */
    uint8_t max_anisotropy;           /* 1..16 */
    bool    normalized;
    float   lod_min, lod_max;
} lagfx_sampler_desc_t;

/* Sampler state object bytes -> description; false if not a sampler record. */
bool lagfx_parse_sampler_object(const uint8_t *b, size_t len, lagfx_sampler_desc_t *out);

/* Packed constexpr-sampler bits -> description. */
void lagfx_decode_constexpr_sampler(uint64_t bits, lagfx_sampler_desc_t *out);

#endif /* LIBAPPLEGFX_PROTOCOL_SAMPLER_STATE_H */
