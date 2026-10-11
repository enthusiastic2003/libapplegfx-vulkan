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
    put16(b, 0x56, fmt);
    put32(b, 0x58, w);
    put32(b, 0x5c, h);
    put16(b, 0x6a, storage << 4);
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

    if (failures) return 1;
    printf("texture-object-unit: OK\n");
    return 0;
}
