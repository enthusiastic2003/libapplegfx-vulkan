/*
 * libapplegfx-vulkan — serialized render pipeline descriptor: colour attachments
 * src/protocol/pipeline_descriptor.h
 *
 * SPDX-License-Identifier: MIT
 *
 * A pipeline object's data (slot type 0x0e) is a 16-byte header
 * {u32 type, u32 total length, u32 object id, u32 body length} followed by a
 * top-level record. Records use the same encoding as the vertex descriptor
 * (src/protocol/vertex_descriptor.h): u8 field count, then {u8 key, u8 size,
 * value} per field, fields left at their default omitted. Arrays are a u32
 * count followed by u32 offsets (relative to the array) of their records.
 *
 * Top-level key 0x08 is the offset, from the start of the top-level record, of
 * the colour attachment array. Each attachment record (matched against
 * pipelines with known blend settings, tests/guest/kat.m `blend-probe`):
 *
 *   key  field                         default
 *   0    attachment index              0
 *   1    MTLPixelFormat                0
 *   2    blendingEnabled               0
 *   3    sourceRGBBlendFactor          1 (One)
 *   4    destinationRGBBlendFactor     0 (Zero)
 *   5    rgbBlendOperation             0 (Add)
 *   6    sourceAlphaBlendFactor        1 (One)
 *   7    destinationAlphaBlendFactor   0 (Zero)
 *   8    alphaBlendOperation           0 (Add)
 *   9    writeMask (MTLColorWriteMask) 15 (All)
 */
#ifndef LIBAPPLEGFX_PROTOCOL_PIPELINE_DESCRIPTOR_H
#define LIBAPPLEGFX_PROTOCOL_PIPELINE_DESCRIPTOR_H

#include <stddef.h>
#include <stdint.h>

#define LAGFX_PSO_HEADER_LEN 16u

typedef struct {
    uint8_t  index;
    uint32_t pixel_format;      /* MTLPixelFormat */
    uint8_t  blending;
    uint8_t  src_rgb, dst_rgb, rgb_op;        /* MTLBlendFactor / MTLBlendOperation */
    uint8_t  src_alpha, dst_alpha, alpha_op;
    uint8_t  write_mask;        /* MTLColorWriteMask: R 8, G 4, B 2, A 1 */
} lagfx_pso_color_attachment_t;

/* Decode the colour attachments of a pipeline object's data. Returns the
 * number written (0 if the layout isn't recognised). */
uint32_t lagfx_parse_pso_color_attachments(const uint8_t *blob, size_t len,
                                           lagfx_pso_color_attachment_t *out, uint32_t cap);

#endif /* LIBAPPLEGFX_PROTOCOL_PIPELINE_DESCRIPTOR_H */
