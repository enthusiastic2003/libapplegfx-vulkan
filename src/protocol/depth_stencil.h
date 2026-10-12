/*
 * libapplegfx-vulkan — depth-stencil state object (slot type 0x07) decoder
 * src/protocol/depth_stencil.h
 *
 * SPDX-License-Identifier: MIT
 *
 * An MTLDepthStencilState's object data, matched against a state made with
 * distinct values for every field (tests/guest/kat.m, `kat stencil-probe`):
 *
 *   +0x00  u32  4
 *   +0x04  u32  0x28 (length)
 *   +0x08  u32  the object's own id
 *   +0x0c  u32  bits 0-2 depth MTLCompareFunction, bit 3 depth write,
 *               bits 4 / 5 front / back stencil descriptor present
 *   +0x10  u32  front face: bits 0-2 MTLCompareFunction, 3-5 stencil failure,
 *               6-8 depth failure, 9-11 depth-stencil pass (MTLStencilOperation)
 *   +0x14  u32  front read mask
 *   +0x18  u32  front write mask
 *   +0x1c  u32  back face, as +0x10
 *   +0x20  u32  back read mask
 *   +0x24  u32  back write mask
 *
 * Metal's compare functions and stencil operations are numbered as Vulkan's
 * VkCompareOp and VkStencilOp.
 *
 * The render pass (inner op 0x1a) names the stencil attachment: +0x28 u32
 * texture, +0x3c u16 load action, +0x3e u16 store action, +0x44 u32 clear
 * value; setDepthStencilState is op 0x68 {u32 object}, and
 * setStencilFrontReferenceValue:backReferenceValue: op 0x77 {u32, u32}.
 */
#ifndef LIBAPPLEGFX_PROTOCOL_DEPTH_STENCIL_H
#define LIBAPPLEGFX_PROTOCOL_DEPTH_STENCIL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define LAGFX_DEPTH_STENCIL_OBJECT_LEN 0x28u

typedef struct {
    uint32_t compare, stencil_fail, depth_fail, pass;   /* Metal = Vulkan numbering */
    uint32_t read_mask, write_mask;
} lagfx_stencil_face_t;

typedef struct {
    uint32_t depth_compare;
    bool depth_write;
    lagfx_stencil_face_t front, back;
} lagfx_depth_stencil_t;

/* Metal's default state: depth Always without writes, stencil Always / Keep. */
void lagfx_depth_stencil_default(lagfx_depth_stencil_t *out);

/* Decode; false when the bytes don't look like a depth-stencil object. */
bool lagfx_parse_depth_stencil_object(const uint8_t *b, size_t len, lagfx_depth_stencil_t *out);

#endif /* LIBAPPLEGFX_PROTOCOL_DEPTH_STENCIL_H */
