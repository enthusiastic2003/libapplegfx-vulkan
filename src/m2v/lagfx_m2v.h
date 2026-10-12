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

#endif /* LAGFX_M2V_H */
