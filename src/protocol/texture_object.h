/*
 * libapplegfx-vulkan — texture object (resource slot type 0x02/0x03/0x0c) decoder
 * src/protocol/texture_object.h
 *
 * SPDX-License-Identifier: MIT
 *
 * A texture's object data (the slot's bytes_va) starts with its placement:
 * the allocation holding the texel memory in the task's address space
 * (page << 12) and where in it the texture lies, followed by the texture
 * descriptor. Offsets below were matched against textures created with
 * known properties (tests/guest/kat.m, `kat probe`):
 *
 *   +0x00  u64  allocation size
 *   +0x08  u64  allocation page (low 32 bits)
 *   +0x10  u64  byte offset of texel (0,0) in the allocation (non-zero for
 *               Core Animation's sub-allocated text masks, e.g. 0x100)
 *   +0x18  u64  texture length (bytes per row x height)
 *   +0x34  u32  bytes per row
 *   +0x3c  u32  width            (repeated at +0x58)
 *   +0x40  u32  height           (repeated at +0x5c)
 *   +0x44  u32  depth
 *   +0x54  u8   0x40 + MTLTextureType (0x41 1D array, 0x42 2D, 0x47 3D, ...)
 *   +0x55  u8   MTLTextureUsage
 *   +0x56  u16  MTLPixelFormat
 *   +0x64  u16  mipmap level count (unconfirmed: it and +0x66 have only been 1)
 *   +0x66  u16  sample count (unconfirmed)
 *   +0x68  u16  array length (3 for WindowServer's 1D-array colour lookup table)
 *   +0x6a  u16  MTLResourceOptions (storage mode in bits 4..7)
 *
 * A texture made from a buffer (newTextureWithDescriptor:offset:bytesPerRow:,
 * slot type 0x08) has the buffer's memory from a byte offset (kat tex-buffer):
 *
 *   +0x08  u32  the object's own id
 *   +0x0c  u32  the buffer's object id
 *   +0x10  u64  byte offset into the buffer
 *   +0x18  u64  bytes per row
 *   +0x20  descriptor, as at +0x54 above (width +0x24, height +0x28, ...)
 *
 * A texture made from an IOSurface (slot type 0x05) has no placement of its
 * own: it names the surface, and the descriptor sits 0x40 bytes earlier.
 *
 *   +0x00  u32  IOSurface id (IOSurfaceGetID)
 *   +0x10  u32  the object's own id
 *   +0x14  descriptor, as at +0x54 above (usage +0x15, format +0x16,
 *          width +0x18, height +0x1c, depth +0x20, options +0x2a)
 *   +0x34  u16  plane (newTextureWithDescriptor:iosurface:plane:); the
 *               upper two bytes are never written
 *
 * The surface itself is object <IOSurface id> of task 0, which holds every
 * surface's memory; its object data (slot type 0x04, 0x24 bytes) is
 *
 *   +0x00  u64  size (page-rounded allocation)
 *   +0x08  u32  page in task 0's address space (low 32 bits)
 *   +0x0c  u32  IOSurface pixel format (four-character code, e.g. 'BGRA')
 *   +0x10  u32  plane count (1 for single-plane surfaces)
 *   +0x14  one 16-byte entry per plane:
 *            +0  u32  byte offset of the plane in the surface's memory
 *            +4  u32  width
 *            +8  u32  height
 *            +c  u16  bytes per row
 *            +f  u8   bytes per element (+e has been 0)
 *          so plane 0's width is at +0x18 and its bytes per row at +0x20.
 *
 * Matched against surfaces of known size and row stride
 * (tests/guest/kat.m, `kat iosurface-probe`) and a two-plane '420v' surface
 * (`kat iosurface-planes`: plane 1 at +0x24, offset 0x5000). The desktop
 * wallpaper is such a surface: 16-bit Y at 3840x2160 and CbCr at 1920x1080.
 */
#ifndef LIBAPPLEGFX_PROTOCOL_TEXTURE_OBJECT_H
#define LIBAPPLEGFX_PROTOCOL_TEXTURE_OBJECT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define LAGFX_TEXTURE_OBJECT_MIN_LEN 0x70u
#define LAGFX_IOSURFACE_TEXTURE_MIN_LEN 0x38u
#define LAGFX_IOSURFACE_RECORD_MIN_LEN 0x24u     /* one plane */
#define LAGFX_IOSURFACE_MAX_PLANES 4u
#define LAGFX_IOSURFACE_RECORD_LEN (0x14u + 0x10u * LAGFX_IOSURFACE_MAX_PLANES)
#define LAGFX_BUFFER_TEXTURE_MIN_LEN 0x38u

/* MTLStorageMode */
#define LAGFX_STORAGE_SHARED   0u
#define LAGFX_STORAGE_MANAGED  1u
#define LAGFX_STORAGE_PRIVATE  2u

typedef struct {
    uint32_t texture_type;    /* MTLTextureType */
    uint32_t array_length, mip_levels, sample_count;
    uint32_t width, height, depth;
    uint32_t bytes_per_row;
    uint32_t pixel_format;    /* MTLPixelFormat */
    uint32_t usage;           /* MTLTextureUsage */
    uint32_t storage;         /* MTLStorageMode */
    uint32_t n_place;
    struct { uint64_t size, page; } place[4];
    uint64_t base_offset;     /* byte offset of texel (0,0) in the placement */
    uint32_t buffer_ref;      /* buffer-backed: the buffer's object id, else 0 */
    uint32_t surface_id;      /* IOSurface-backed: the surface's id, else 0 */
    uint32_t plane;           /* IOSurface-backed: the surface plane it views */
} lagfx_texture_object_t;

/* One plane of an IOSurface. */
typedef struct {
    uint32_t offset;          /* bytes from the start of the surface's memory */
    uint32_t width, height;
    uint32_t bytes_per_row, bytes_per_element;
} lagfx_iosurface_plane_t;

/* An IOSurface's record in task 0. width .. bytes_per_element repeat plane 0. */
typedef struct {
    uint64_t size, page;
    uint32_t fourcc;
    uint32_t width, height;
    uint32_t bytes_per_row, bytes_per_element;
    uint32_t plane_count;     /* 1 for single-plane surfaces */
    lagfx_iosurface_plane_t planes[LAGFX_IOSURFACE_MAX_PLANES];
} lagfx_iosurface_record_t;

/* Decode; false when the bytes don't look like a texture object. */
bool lagfx_parse_texture_object(const uint8_t *b, size_t len, lagfx_texture_object_t *out);

/* Decode an IOSurface texture object. Leaves bytes_per_row and the placement
 * empty; lagfx_texture_object_attach_surface fills them. */
bool lagfx_parse_iosurface_texture_object(const uint8_t *b, size_t len,
                                          lagfx_texture_object_t *out);

/* Decode a buffer texture object (slot type 0x08). Leaves the placement
 * empty: it is the buffer's (buffer_ref), from base_offset. */
bool lagfx_parse_buffer_texture_object(const uint8_t *b, size_t len, lagfx_texture_object_t *out);

bool lagfx_parse_iosurface_record(const uint8_t *b, size_t len, lagfx_iosurface_record_t *out);

/* Give an IOSurface texture its surface's memory (placement in task 0) and
 * row stride. False when the surface can't hold the texture. */
bool lagfx_texture_object_attach_surface(lagfx_texture_object_t *t,
                                         const lagfx_iosurface_record_t *s);

#endif /* LIBAPPLEGFX_PROTOCOL_TEXTURE_OBJECT_H */
