/*
 * libapplegfx-vulkan — serialized MTLVertexDescriptor decoder
 * src/protocol/vertex_descriptor.h
 *
 * SPDX-License-Identifier: MIT
 *
 * A render-pipeline descriptor blob carries its vertex descriptor as two
 * record arrays (attributes, then buffer layouts). An array is
 *
 *     u32 count, u32 offset[count]   (offsets relative to the array start;
 *                                     offset[0] == 4 + 4*count)
 *
 * and each record is
 *
 *     u8 nfields, then nfields x { u8 key, u8 size (always 4), u32 value }
 *
 * with keys in increasing order and zero-valued fields omitted. Key 0 is the
 * element index. Attribute records: 1 = MTLVertexFormat, 2 = offset,
 * 3 = bufferIndex. Layout records: 3 = stride.
 */
#ifndef LIBAPPLEGFX_PROTOCOL_VERTEX_DESCRIPTOR_H
#define LIBAPPLEGFX_PROTOCOL_VERTEX_DESCRIPTOR_H

#include <stdint.h>

typedef struct {
    uint32_t index;   /* [[attribute(n)]] */
    uint32_t fmt;     /* MTLVertexFormat */
    uint32_t off;
    uint32_t bufidx;  /* Metal buffer index the attribute is fetched from */
} lagfx_vtxdesc_attr_t;

typedef struct {
    uint32_t index;   /* Metal buffer index */
    uint32_t stride;
} lagfx_vtxdesc_layout_t;

/* Decode the vertex descriptor from a pipeline descriptor blob. Attributes are
 * returned sorted by index; layouts deduplicated by index. Returns the number
 * of attributes found (0 = no vertex descriptor recognised). */
uint32_t lagfx_parse_vertex_descriptor(const uint8_t *blob, uint32_t len,
                                       lagfx_vtxdesc_attr_t *attrs, uint32_t attr_cap,
                                       lagfx_vtxdesc_layout_t *layouts, uint32_t layout_cap,
                                       uint32_t *n_layouts);

#endif /* LIBAPPLEGFX_PROTOCOL_VERTEX_DESCRIPTOR_H */
