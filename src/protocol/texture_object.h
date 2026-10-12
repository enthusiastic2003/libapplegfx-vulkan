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
 *
 * A texture made from an IOSurface (slot type 0x05) has no placement of its
 * own: it names the surface, and the descriptor sits 0x40 bytes earlier.
 *
 *   +0x00  u32  IOSurface id (IOSurfaceGetID)
 *   +0x10  u32  the object's own id
 *   +0x14  descriptor, as at +0x54 above (usage +0x15, format +0x16,
 *          width +0x18, height +0x1c, depth +0x20, options +0x2a)
 *
 * The surface itself is object <IOSurface id> of task 0, which holds every
 * surface's memory; its object data (slot type 0x04, 0x24 bytes) is
 *
 *   +0x00  u64  size (page-rounded allocation)
 *   +0x08  u32  page in task 0's address space (low 32 bits)
 *   +0x0c  u32  IOSurface pixel format (four-character code, e.g. 'BGRA')
 *   +0x18  u32  width
 *   +0x1c  u32  height
 *   +0x20  u16  bytes per row
 *   +0x23  u8   bytes per element (+0x22 has been 0)
 *
 * Matched against surfaces of known size and row stride
 * (tests/guest/kat.m, `kat iosurface-probe`).
 */
#ifndef LIBAPPLEGFX_PROTOCOL_TEXTURE_OBJECT_H
#define LIBAPPLEGFX_PROTOCOL_TEXTURE_OBJECT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define LAGFX_TEXTURE_OBJECT_MIN_LEN 0x70u
#define LAGFX_IOSURFACE_TEXTURE_MIN_LEN 0x30u
#define LAGFX_IOSURFACE_RECORD_LEN 0x24u

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
    uint32_t surface_id;      /* IOSurface-backed: the surface's id, else 0 */
} lagfx_texture_object_t;

/* An IOSurface's record in task 0. */
typedef struct {
    uint64_t size, page;
    uint32_t fourcc;
    uint32_t width, height;
    uint32_t bytes_per_row, bytes_per_element;
} lagfx_iosurface_record_t;

/* Decode; false when the bytes don't look like a texture object. */
bool lagfx_parse_texture_object(const uint8_t *b, size_t len, lagfx_texture_object_t *out);

/* Decode an IOSurface texture object. Leaves bytes_per_row and the placement
 * empty; lagfx_texture_object_attach_surface fills them. */
bool lagfx_parse_iosurface_texture_object(const uint8_t *b, size_t len,
                                          lagfx_texture_object_t *out);

bool lagfx_parse_iosurface_record(const uint8_t *b, size_t len, lagfx_iosurface_record_t *out);

/* Give an IOSurface texture its surface's memory (placement in task 0) and
 * row stride. False when the surface can't hold the texture. */
bool lagfx_texture_object_attach_surface(lagfx_texture_object_t *t,
                                         const lagfx_iosurface_record_t *s);

#endif /* LIBAPPLEGFX_PROTOCOL_TEXTURE_OBJECT_H */
