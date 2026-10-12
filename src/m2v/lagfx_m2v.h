/*
 * libapplegfx-vulkan — C interface to the metal2vulkan bridge
 * src/m2v/lagfx_m2v.h
 *
 * SPDX-License-Identifier: MIT
 *
 * Implemented in rust/lagfx-m2v over third_party/metal2vulkan (a git
 * submodule), linked when the build defines LAGFX_HAVE_M2V. Metal values are
 * passed raw (MTLPixelFormat, MTLBlendFactor, ...). Errors come back as text in
 * the caller's buffer; handles are freed with their *_free.
 */
#ifndef LAGFX_M2V_H
#define LAGFX_M2V_H

#include <stddef.h>
#include <stdint.h>

typedef struct M2vDevice lagfx_m2v_device_t;
typedef struct M2vPipeline lagfx_m2v_pipeline_t;

/* Describe the physical device to the translator. The VkDevice must enable
 * every core (1.0-1.3) feature the device supports; no extension is assumed.
 * `get_instance_proc_addr` is vkGetInstanceProcAddr. NULL on failure. */
lagfx_m2v_device_t *lagfx_m2v_device_new(const void *get_instance_proc_addr, uint64_t instance,
                                         uint64_t physical_device, uint32_t api_version,
                                         char *err, size_t err_len);
void lagfx_m2v_device_free(lagfx_m2v_device_t *dev);

typedef struct {
    uint32_t index, format /* MTLVertexFormat */, offset, buffer_index;
} lagfx_m2v_vertex_attribute_t;

typedef struct {
    uint32_t buffer_index;
    uint32_t stride;         /* 0 = dynamic */
    uint32_t step_function;  /* MTLVertexStepFunction */
    uint32_t step_rate;
} lagfx_m2v_vertex_layout_t;

/* A render pipeline: each stage's library holds that stage's one function
 * (as the guest's pipeline objects reference them); colour attachment 0. */
typedef struct {
    const uint8_t *vertex_lib;
    size_t vertex_lib_len;
    const uint8_t *fragment_lib;     /* NULL: no fragment stage */
    size_t fragment_lib_len;
    uint32_t color_format;           /* MTLPixelFormat; 0 = none */
    uint32_t blending;
    uint32_t src_rgb, dst_rgb, rgb_op, src_alpha, dst_alpha, alpha_op;
    uint32_t write_mask;             /* MTLColorWriteMask */
    uint32_t has_vertex_descriptor;
    uint32_t n_attributes;
    const lagfx_m2v_vertex_attribute_t *attributes;
    uint32_t n_layouts;
    const lagfx_m2v_vertex_layout_t *layouts;
} lagfx_m2v_render_desc_t;

/* Compile; NULL on failure or refusal, with the reason in `err`. */
lagfx_m2v_pipeline_t *lagfx_m2v_compile_render(const lagfx_m2v_device_t *dev,
                                               const lagfx_m2v_render_desc_t *desc,
                                               char *err, size_t err_len);
uint32_t lagfx_m2v_pipeline_stage_count(const lagfx_m2v_pipeline_t *p);
/* Stage i's SPIR-V words (vertex first); NULL past the last stage. */
const uint32_t *lagfx_m2v_pipeline_spirv(const lagfx_m2v_pipeline_t *p, uint32_t i,
                                         size_t *n_words);
void lagfx_m2v_pipeline_free(lagfx_m2v_pipeline_t *p);

/* ---- pipeline facts ---- */

typedef struct {
    uint32_t mag_filter, min_filter, mipmap_mode;          /* VkFilter / VkSamplerMipmapMode */
    uint32_t address_u, address_v, address_w;              /* VkSamplerAddressMode */
    float mip_lod_bias;
    uint32_t anisotropy_enable;
    float max_anisotropy;
    uint32_t compare_enable, compare_op;
    float min_lod, max_lod;
    uint32_t border_color;                                 /* VkBorderColor */
    uint32_t reduction;                                    /* VkSamplerReductionMode */
} lagfx_m2v_vk_sampler_t;

typedef struct {
    uint32_t set, binding;
    uint32_t descriptor_type;   /* VkDescriptorType */
    uint32_t count;
    uint32_t stages;            /* VkShaderStageFlags */
    uint32_t flags;             /* VkDescriptorBindingFlags */
    uint32_t has_immutable_sampler;
    lagfx_m2v_vk_sampler_t immutable_sampler;
} lagfx_m2v_layout_binding_t;

/* Layout bindings (returns the total; fills up to cap) and the push range. */
uint32_t lagfx_m2v_pipeline_layout(const lagfx_m2v_pipeline_t *p, lagfx_m2v_layout_binding_t *out,
                                   uint32_t cap, uint32_t *push_size, uint32_t *push_stages);
/* Stage i's entry point into out; returns its VkShaderStageFlags. */
uint32_t lagfx_m2v_pipeline_entry(const lagfx_m2v_pipeline_t *p, uint32_t i, char *out, size_t cap);
const uint8_t *lagfx_m2v_pipeline_constant_blob(const lagfx_m2v_pipeline_t *p, uint32_t i,
                                                size_t *len);

typedef struct {
    uint32_t binding, buffer_index;
    uint32_t input_rate;        /* VkVertexInputRate */
    uint32_t stride;            /* 0 = per bind */
} lagfx_m2v_vertex_binding_t;

typedef struct {
    uint32_t location, binding;
    uint32_t format;            /* VkFormat */
    uint32_t offset;
} lagfx_m2v_vertex_attribute_out_t;

/* Fixed-function vertex input (arrays of 16): returns n bindings. */
uint32_t lagfx_m2v_pipeline_vertex_input(const lagfx_m2v_pipeline_t *p,
                                         lagfx_m2v_vertex_binding_t *bindings,
                                         lagfx_m2v_vertex_attribute_out_t *attributes,
                                         uint32_t *n_attributes);
uint32_t lagfx_m2v_pipeline_topology(const lagfx_m2v_pipeline_t *p);

/* ---- draws ---- */

typedef struct {
    uint32_t index;
    uint32_t is_bytes;
    uint64_t length, offset, gpu_address;
    uint64_t device_address;            /* host VkBuffer address of the allocation; 0 = none */
} lagfx_m2v_buffer_t;

typedef struct {
    uint32_t index;
    uint32_t texture_type, pixel_format, usage;            /* MTLTextureType / ...Format / ...Usage */
    uint32_t mip_levels, array_length, sample_count;
} lagfx_m2v_texture_t;

typedef struct {
    uint32_t index;
    uint32_t min_filter, mag_filter, mip_filter;
    uint32_t s_address, t_address, r_address;
    uint32_t border_color, normalized, max_anisotropy, compare_function;
    float lod_min, lod_max;
} lagfx_m2v_sampler_t;

typedef struct {
    uint32_t n_buffers;
    const lagfx_m2v_buffer_t *buffers;
    uint32_t n_textures;
    const lagfx_m2v_texture_t *textures;
    uint32_t n_samplers;
    const lagfx_m2v_sampler_t *samplers;
} lagfx_m2v_stage_t;

enum {
    LAGFX_M2V_W_UNSUPPORTED = 0, LAGFX_M2V_W_BUFFER, LAGFX_M2V_W_INLINE_BYTES, LAGFX_M2V_W_TEXTURE,
    LAGFX_M2V_W_SAMPLER, LAGFX_M2V_W_COMBINED, LAGFX_M2V_W_NEUTRAL, LAGFX_M2V_W_CONSTANT_BLOB,
    LAGFX_M2V_W_BYTES,
};

typedef struct {
    uint32_t set, binding, element, descriptor_type;
    uint32_t kind;                      /* LAGFX_M2V_W_* */
    uint32_t stage, slot;               /* 0 vertex, 1 fragment; Metal index */
    uint32_t view_type, view_format, view_layout, view_aspect;
    uint32_t swizzle[4];                /* VkComponentSwizzle r g b a */
    uint32_t sampler_kind;              /* 0 `sampler`, 1 immutable, 2 neutral */
    lagfx_m2v_vk_sampler_t sampler;
    uint32_t image_neutral;             /* combined write: the image is a neutral */
    uint32_t neutral_kind;              /* 1 zero buffer, 2 sink, 3 image, 4 texel buffer, 5 sampler */
    uint32_t neutral_min_bytes, neutral_view_type, neutral_sampled_type, neutral_format,
             neutral_compare;
    uint32_t has_range;
    uint64_t range_offset, range_size;
    int32_t bytes_index;
} lagfx_m2v_write_t;

typedef struct {
    uint32_t stages, offset, size, data_offset;
} lagfx_m2v_push_t;

typedef struct M2vDraw lagfx_m2v_draw_t;

/* An indexed draw's index buffer (MTLIndexType 0 UInt16 / 1 UInt32). */
typedef struct {
    uint32_t index_type;
    uint64_t buffer_length;
    uint64_t offset;          /* bytes */
    int32_t base_vertex;
} lagfx_m2v_indexed_t;

/* Plan one drawPrimitives; NULL (reason in err) when it can't (yet). */
lagfx_m2v_draw_t *lagfx_m2v_plan_draw(const lagfx_m2v_device_t *dev, const lagfx_m2v_pipeline_t *p,
                                      const lagfx_m2v_stage_t stages[2], uint32_t vertex_start,
                                      uint32_t vertex_count, uint32_t instance_count,
                                      uint32_t base_instance, const lagfx_m2v_indexed_t *indexed,
                                      char *err, size_t err_len);
const lagfx_m2v_write_t *lagfx_m2v_draw_writes(const lagfx_m2v_draw_t *d, uint32_t *n);
const uint8_t *lagfx_m2v_draw_bytes(const lagfx_m2v_draw_t *d, uint32_t i, size_t *len);
const lagfx_m2v_push_t *lagfx_m2v_draw_push(const lagfx_m2v_draw_t *d, uint32_t *n,
                                            const uint8_t **data);
const uint32_t *lagfx_m2v_draw_specialization(const lagfx_m2v_draw_t *d, const uint8_t **data,
                                              size_t *data_len, uint32_t *n_entries);
const uint64_t *lagfx_m2v_draw_vertex_buffers(const lagfx_m2v_draw_t *d, uint32_t *n);
/* {kind 0 draw / 1 indexed, count, instance_count, first vertex or index,
 * vertex offset, first_instance, VkIndexType, 0}; *index_offset for indexed. */
const uint32_t *lagfx_m2v_draw_call(const lagfx_m2v_draw_t *d, uint64_t *index_offset);
void lagfx_m2v_draw_free(lagfx_m2v_draw_t *d);
/* The pipeline the draw was planned on: the one given or a variant of it
 * (owned by that one); use its SPIR-V and layout. */
const lagfx_m2v_pipeline_t *lagfx_m2v_draw_pipeline(const lagfx_m2v_draw_t *d);
/* Push holes {kind 1 root table / 2 execution status, stages, push offset}:
 * each takes a u64 device address. */
const uint32_t *lagfx_m2v_draw_holes(const lagfx_m2v_draw_t *d, uint32_t *n);
/* Root table i (bytes, stages); write the sink buffer's device address at
 * lagfx_m2v_root_sink_offset() before uploading. NULL past the last. */
const uint8_t *lagfx_m2v_draw_root_table(const lagfx_m2v_draw_t *d, uint32_t i, uint32_t *stages,
                                         size_t *len);
uint32_t lagfx_m2v_root_sink_offset(void);

#endif /* LAGFX_M2V_H */
