/*
 * libapplegfx-vulkan — texture object (resource slot type 0x02/0x03/0x0c) decoder
 * src/protocol/texture_object.h
 *
 * SPDX-License-Identifier: MIT
 *
 * A texture's object data (the slot's bytes_va) starts with its placement:
 * up to two {u64 size, u64 page} entries locating the texel memory in the
 * task's address space (page << 12), followed by the texture descriptor.
 * Offsets below were matched against textures created with known
 * properties (tests/guest/kat.m, `kat probe`):
 *
 *   +0x00  {u64 size, u64 page} x 2   placement (page: low 32 bits)
 *   +0x34  u32  bytes per row
 *   +0x3c  u32  width            (repeated at +0x58)
 *   +0x40  u32  height           (repeated at +0x5c)
 *   +0x44  u32  depth
 *   +0x55  u8   MTLTextureUsage
 *   +0x56  u16  MTLPixelFormat
 *   +0x6a  u16  MTLResourceOptions (storage mode in bits 4..7)
 */
#ifndef LIBAPPLEGFX_PROTOCOL_TEXTURE_OBJECT_H
#define LIBAPPLEGFX_PROTOCOL_TEXTURE_OBJECT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define LAGFX_TEXTURE_OBJECT_MIN_LEN 0x70u

/* MTLStorageMode */
#define LAGFX_STORAGE_SHARED   0u
#define LAGFX_STORAGE_MANAGED  1u
#define LAGFX_STORAGE_PRIVATE  2u

typedef struct {
    uint32_t width, height, depth;
    uint32_t bytes_per_row;
    uint32_t pixel_format;    /* MTLPixelFormat */
    uint32_t usage;           /* MTLTextureUsage */
    uint32_t storage;         /* MTLStorageMode */
    uint32_t n_place;
    struct { uint64_t size, page; } place[2];
} lagfx_texture_object_t;

/* Decode; false when the bytes don't look like a texture object. */
bool lagfx_parse_texture_object(const uint8_t *b, size_t len, lagfx_texture_object_t *out);

#endif /* LIBAPPLEGFX_PROTOCOL_TEXTURE_OBJECT_H */
