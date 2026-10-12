/*
 * libapplegfx-vulkan — texture object decoder (unit)
 * tests/texture-object-unit.c
 *
 * SPDX-License-Identifier: MIT
 *
 * Synthetic objects built field by field (see src/protocol/texture_object.h).
 */

#include "protocol/texture_object.h"

#include <stdio.h>
#include <string.h>

static int failures;

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } \
} while (0)

static void put16(uint8_t *b, size_t at, uint32_t v) { b[at] = (uint8_t)v; b[at + 1] = (uint8_t)(v >> 8); }
static void put32(uint8_t *b, size_t at, uint32_t v) { put16(b, at, v); put16(b, at + 2, v >> 16); }
static void put64(uint8_t *b, size_t at, uint64_t v) { put32(b, at, (uint32_t)v); put32(b, at + 4, (uint32_t)(v >> 32)); }

static void make(uint8_t *b, uint32_t w, uint32_t h, uint32_t bpr, uint32_t fmt,
                 uint32_t usage, uint32_t storage, uint64_t size, uint64_t page) {
    memset(b, 0, 0x80);
    put64(b, 0x00, size);
    put64(b, 0x08, page | 0x0001000100000000ull);   /* high bits are not part of the page */
    put32(b, 0x34, bpr);
    put32(b, 0x3c, w);
    put32(b, 0x40, h);
    put32(b, 0x44, 1);
    b[0x54] = 0x42;
    b[0x55] = (uint8_t)usage;
    put16(b, 0x64, 1);
    put16(b, 0x66, 1);
    put16(b, 0x68, 1);
    put16(b, 0x56, fmt);
    put32(b, 0x58, w);
    put32(b, 0x5c, h);
    put16(b, 0x6a, storage << 4);
}

static void make_ios_tex(uint8_t *b, uint32_t sid, uint32_t w, uint32_t h, uint32_t fmt,
                         uint32_t usage, uint32_t storage) {
    memset(b, 0, 0x38);
    put32(b, 0x00, sid);
    put32(b, 0x10, 4);          /* own id */
    b[0x14] = 0x42;
    b[0x15] = (uint8_t)usage;
    put16(b, 0x16, fmt);
    put32(b, 0x18, w);
    put32(b, 0x1c, h);
    put32(b, 0x20, 1);
    put16(b, 0x2a, storage << 4);
}

static void make_surface(uint8_t *b, uint64_t size, uint32_t page, uint32_t w, uint32_t h,
                         uint32_t bpr) {
    memset(b, 0, 0x24);
    put64(b, 0x00, size);
    put32(b, 0x08, page);
    put32(b, 0x0c, 0x42475241u);   /* 'BGRA' */
    put32(b, 0x10, 1);
    put32(b, 0x18, w);
    put32(b, 0x1c, h);
    put16(b, 0x20, bpr);
    b[0x23] = 4;
}

static void test_iosurface(void) {
    uint8_t tb[0x38], sb[0x24];
    lagfx_texture_object_t t;
    lagfx_iosurface_record_t s;

    /* 100x20 BGRA surface with a padded 512-byte row stride */
    make_ios_tex(tb, 3, 100, 20, 80, 1, 1);
    CHECK(lagfx_parse_iosurface_texture_object(tb, sizeof tb, &t));
    CHECK(t.surface_id == 3 && t.width == 100 && t.height == 20 && t.pixel_format == 80);
    CHECK(t.usage == 1 && t.storage == LAGFX_STORAGE_MANAGED && t.n_place == 0);
    make_surface(sb, 0x3000, 0x8d1, 100, 20, 512);
    CHECK(lagfx_parse_iosurface_record(sb, sizeof sb, &s));
    CHECK(s.size == 0x3000 && s.page == 0x8d1 && s.fourcc == 0x42475241u);
    CHECK(s.width == 100 && s.height == 20 && s.bytes_per_row == 512 && s.bytes_per_element == 4);
    CHECK(lagfx_texture_object_attach_surface(&t, &s));
    CHECK(t.bytes_per_row == 512 && t.n_place == 1);
    CHECK(t.place[0].page == 0x8d1 && t.place[0].size == 0x3000);

    /* texture larger than its surface */
    make_ios_tex(tb, 3, 101, 20, 80, 1, 1);
    CHECK(lagfx_parse_iosurface_texture_object(tb, sizeof tb, &t));
    CHECK(!lagfx_texture_object_attach_surface(&t, &s));

    /* rows that don't fit the allocation */
    make_surface(sb, 0x2000, 0x8d1, 100, 20, 512);
    CHECK(!lagfx_parse_iosurface_record(sb, sizeof sb, &s));

    /* no surface id / not a descriptor */
    make_ios_tex(tb, 0, 40, 24, 80, 1, 1);
    CHECK(!lagfx_parse_iosurface_texture_object(tb, sizeof tb, &t));
    make_ios_tex(tb, 3, 40, 24, 80, 1, 1);
    tb[0x14] = 0;
    CHECK(!lagfx_parse_iosurface_texture_object(tb, sizeof tb, &t));
}

int main(void) {
    uint8_t b[0x80];
    lagfx_texture_object_t t;

    make(b, 64, 64, 256, 80, 5, 1, 0x4000, 0x101);
    CHECK(lagfx_parse_texture_object(b, sizeof b, &t));
    CHECK(t.width == 64 && t.height == 64 && t.depth == 1);
    CHECK(t.bytes_per_row == 256 && t.pixel_format == 80 && t.usage == 5);
    CHECK(t.storage == LAGFX_STORAGE_MANAGED);
    CHECK(t.n_place == 1 && t.place[0].size == 0x4000 && t.place[0].page == 0x101);

    CHECK(t.texture_type == 2 && t.array_length == 1 && t.mip_levels == 1 && t.sample_count == 1);
    CHECK(t.base_offset == 0);

    /* a 167x24 A8 text mask 0x100 bytes into an 8 KiB allocation (+0x18 is
     * its length, not a second placement) */
    make(b, 167, 24, 192, 1, 1, 1, 0x2000, 0xb1f);
    put64(b, 0x10, 0x100);
    put64(b, 0x18, 0x1200);
    CHECK(lagfx_parse_texture_object(b, sizeof b, &t));
    CHECK(t.n_place == 1 && t.place[0].size == 0x2000 && t.place[0].page == 0xb1f);
    CHECK(t.base_offset == 0x100 && t.pixel_format == 1 && t.bytes_per_row == 192);

    /* a 1D array of 3 layers (WindowServer's colour lookup table shape) */
    make(b, 16384, 1, 65536, 55, 1, 0, 0x30000, 0x1a9a);
    b[0x54] = 0x41;
    put16(b, 0x68, 3);
    CHECK(lagfx_parse_texture_object(b, sizeof b, &t));
    CHECK(t.texture_type == 1 && t.array_length == 3 && t.pixel_format == 55 && t.width == 16384);

    make(b, 48, 32, 192, 70, 4, 2, 0x2000, 0x2a0);
    CHECK(lagfx_parse_texture_object(b, sizeof b, &t));
    CHECK(t.width == 48 && t.height == 32 && t.pixel_format == 70);
    CHECK(t.storage == LAGFX_STORAGE_PRIVATE && t.place[0].page == 0x2a0);

    /* mismatched duplicate dimensions -> not a texture object we know */
    make(b, 64, 64, 256, 80, 5, 1, 0x4000, 0x101);
    put32(b, 0x58, 65);
    CHECK(!lagfx_parse_texture_object(b, sizeof b, &t));

    /* too short */
    make(b, 64, 64, 256, 80, 5, 1, 0x4000, 0x101);
    CHECK(!lagfx_parse_texture_object(b, 0x40, &t));

    test_iosurface();

    /* a buffer texture (slot type 0x08) as kat tex-buffer made it: 8x8 BGRA8
     * at offset 0x300 of buffer 1, 0x200 bytes per row */
    static const uint8_t bt[0x38] = {
        0x09, 0, 0, 0, 0x40, 0, 0, 0, 0x02, 0, 0, 0, 0x01, 0, 0, 0,
        0x00, 0x03, 0, 0, 0, 0, 0, 0, 0x00, 0x02, 0, 0, 0, 0, 0, 0,
        0x42, 0x01, 0x50, 0x00, 0x08, 0, 0, 0, 0x08, 0, 0, 0, 0x01, 0, 0, 0,
        0x01, 0, 0x01, 0, 0x01, 0, 0, 0,
    };
    CHECK(lagfx_parse_buffer_texture_object(bt, sizeof bt, &t));
    CHECK(t.buffer_ref == 1 && t.base_offset == 0x300 && t.bytes_per_row == 0x200);
    CHECK(t.width == 8 && t.height == 8 && t.pixel_format == 80 && t.texture_type == 2);
    CHECK(t.storage == LAGFX_STORAGE_SHARED && t.n_place == 0);
    CHECK(!lagfx_parse_buffer_texture_object(bt, 0x30, &t));

    if (failures) return 1;
    printf("texture-object-unit: OK\n");
    return 0;
}
