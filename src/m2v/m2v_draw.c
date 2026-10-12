/*
 * libapplegfx-vulkan — render pipelines and draws through metal2vulkan
 * src/m2v/m2v_draw.c
 *
 * SPDX-License-Identifier: MIT
 *
 * See m2v_draw.h. Per draw: describe what the task has bound (buffers by their
 * object's length and guest address, textures by their decoded descriptor,
 * samplers by their decoded state), let metal2vulkan plan it, then make what
 * the plan asks for — buffer ranges copied from guest memory, image views,
 * samplers, neutral stand-ins, push constants, a pipeline per specialization —
 * and record the draw into the pass's target. Per-draw resources are created
 * and freed around a synchronous submit, like the older draw path.
 */

#include "m2v_draw.h"

#if defined(LAGFX_HAVE_M2V) && defined(LAGFX_HAVE_VULKAN)

#include "lagfx_m2v.h"

#include "common/le.h"
#include "common/log.h"
#include "device.h"
#include "handlers/compute/compute_draw_internal.h"
#include "handlers/compute/guest_texture.h"
#include "handlers/compute/task_translate.h"
#include "protocol/depth_stencil.h"
#include "protocol/object_resolver.h"
#include "protocol/pipeline_descriptor.h"
#include "protocol/sampler_state.h"
#include "protocol/texture_object.h"
#include "protocol/vertex_descriptor.h"
#include "vulkan/command.h"
#include "vulkan/instance.h"
#include "vulkan/render_target.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LAYOUT 64u
#define MAX_SETS 4u
#define MAX_VARIANTS 16u
#define MAX_PIPES 256u

typedef struct {
    VkFormat format;
    uint32_t topology;
    uint64_t spec_hash;
    VkFormat stencil_format;     /* VK_FORMAT_UNDEFINED: no stencil attachment */
    uint64_t ds_hash;            /* the depth-stencil state baked in */
    VkPipeline pipeline;
} variant_t;

typedef struct m2v_pipe {
    uint64_t key;
    lagfx_m2v_pipeline_t *pipe;
    uint32_t n_sets;
    VkDescriptorSetLayout dsl[MAX_SETS];
    VkPipelineLayout layout;
    lagfx_m2v_layout_binding_t b[MAX_LAYOUT];
    uint32_t nb;
    VkSampler immutable[MAX_LAYOUT];
    uint32_t push_size, push_stages;
    uint32_t n_stages;
    VkShaderModule mod[2];
    char entry[2][128];
    uint32_t stage_flags[2];
    lagfx_m2v_vertex_binding_t vb[16];
    uint32_t nvb;
    lagfx_m2v_vertex_attribute_out_t va[16];
    uint32_t nva;
    lagfx_pso_color_attachment_t blend;
    bool has_blend;
    variant_t var[MAX_VARIANTS];
    uint32_t nvar;
    bool owns_pipe;           /* false for a variant: its parent owns the shim object */
    struct m2v_child { const lagfx_m2v_pipeline_t *pipe; struct m2v_pipe *mp; } child[8];
    uint32_t nchild;
} m2v_pipe_t;

static struct {
    uint64_t key;
    m2v_pipe_t *pipe;   /* NULL: refused */
} g_pipes[MAX_PIPES];
static uint32_t g_npipes;

static uint64_t fnv(uint64_t h, const void *data, size_t n) {
    const uint8_t *b = data;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001b3ull; }
    return h;
}

/* ------------------------------------------------------------------ helpers */

static uint32_t find_mem(struct lagfx_vk_state *vk, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(vk->phys_device, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

/* Host-visible buffer usable as any buffer descriptor or vertex buffer,
 * holding `len` bytes of `data` (zeros when NULL), padded to 16. */
static bool make_buffer(struct lagfx_vk_state *vk, const void *data, VkDeviceSize len,
                        VkBuffer *buf, VkDeviceMemory *mem) {
    VkDeviceSize size = (len + 15u) & ~(VkDeviceSize)15u;
    if (size == 0u) size = 16u;
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
               | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT
               | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT
               | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (vkCreateBuffer(vk->device, &bci, NULL, buf) != VK_SUCCESS) return false;
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(vk->device, *buf, &mr);
    uint32_t mt = find_mem(vk, mr.memoryTypeBits,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateFlagsInfo fi = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
    };
    VkMemoryAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &fi,
        .allocationSize = mr.size,
        .memoryTypeIndex = mt,
    };
    if (mt == UINT32_MAX || vkAllocateMemory(vk->device, &ai, NULL, mem) != VK_SUCCESS) {
        vkDestroyBuffer(vk->device, *buf, NULL);
        return false;
    }
    vkBindBufferMemory(vk->device, *buf, *mem, 0);
    void *m = NULL;
    if (vkMapMemory(vk->device, *mem, 0, size, 0, &m) == VK_SUCCESS) {
        memset(m, 0, (size_t)size);
        if (data && len) memcpy(m, data, (size_t)len);
        vkUnmapMemory(vk->device, *mem);
    }
    return true;
}

static uint64_t buffer_address(struct lagfx_vk_state *vk, VkBuffer b) {
    VkBufferDeviceAddressInfo ai = { .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = b };
    return (uint64_t)vkGetBufferDeviceAddress(vk->device, &ai);
}

static VkSampler make_sampler(struct lagfx_vk_state *vk, const lagfx_m2v_vk_sampler_t *s) {
    /* cache: samplers are few and long-lived */
    static struct { uint64_t h; VkSampler s; } cache[256];
    static uint32_t n;
    uint64_t h = fnv(0xcbf29ce484222325ull, s, sizeof *s);
    for (uint32_t i = 0; i < n; i++) if (cache[i].h == h) return cache[i].s;
    VkSamplerReductionModeCreateInfo red = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO,
        .reductionMode = (VkSamplerReductionMode)s->reduction,
    };
    VkSamplerCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = s->reduction ? &red : NULL,
        .magFilter = (VkFilter)s->mag_filter,
        .minFilter = (VkFilter)s->min_filter,
        .mipmapMode = (VkSamplerMipmapMode)s->mipmap_mode,
        .addressModeU = (VkSamplerAddressMode)s->address_u,
        .addressModeV = (VkSamplerAddressMode)s->address_v,
        .addressModeW = (VkSamplerAddressMode)s->address_w,
        .mipLodBias = s->mip_lod_bias,
        .anisotropyEnable = s->anisotropy_enable ? VK_TRUE : VK_FALSE,
        .maxAnisotropy = s->max_anisotropy,
        .compareEnable = s->compare_enable ? VK_TRUE : VK_FALSE,
        .compareOp = (VkCompareOp)s->compare_op,
        .minLod = s->min_lod,
        .maxLod = s->max_lod,
        .borderColor = (VkBorderColor)s->border_color,
    };
    VkSampler out = VK_NULL_HANDLE;
    if (vkCreateSampler(vk->device, &ci, NULL, &out) != VK_SUCCESS) return VK_NULL_HANDLE;
    if (n < 256u) { cache[n].h = h; cache[n].s = out; n++; }
    return out;
}

/* A buffer placement is a four-entry scatter/gather table of
 * {u64 size, u32 page, u32 flags}.  Object references may point at successive
 * rows (a controlled guest KAT saw ref 4 at row 0 and ref 5 at row 1).  Flag
 * bit 16 marks the final range; bytes after it belong to another object. */
typedef struct {
    uint32_t n;
    uint64_t size[4], page[4];
    uint64_t length;
} placement_t;

static bool buffer_placement(lagfx_protocol_t *p, const lagfx_task_entry_t *task, uint32_t ref,
                             placement_t *out) {
    uint8_t type = 0;
    uint64_t va = 0, gpa = 0;
    uint8_t d[4u * 16u];
    memset(out, 0, sizeof *out);
    if (!lagfx_resolve_object_data(p, task, ref, &type, &va, &gpa) || va == 0u
        || !lagfx_task_read_virtual(p, task, va, sizeof d, d))
        return false;
    if (getenv("LAGFX_BINDLOG")) {
        static uint32_t dumped[64];
        static unsigned nd;
        bool seen = false;
        for (unsigned i = 0; i < nd && i < 64u; i++) seen |= dumped[i] == ref;
        if (!seen) {
            dumped[nd++ % 64u] = ref;
            char hx[3 * 64 + 1];
            size_t hn = 0;
            for (uint32_t k = 0; k < 64u; k++)
                hn += (size_t)snprintf(hx + hn, sizeof hx - hn, "%02x ", d[k]);
            LAGFX_WARN("M2VBUF task=%u ref=0x%x type=0x%02x: %s", task->id, ref, type, hx);
        }
    }
    for (uint32_t i = 0; i < 4u; i++) {
        const uint8_t *e = d + i * 16u;
        uint64_t size = lagfx_le64(e);
        uint64_t page = lagfx_le32(e + 8u);
        uint32_t flags = lagfx_le32(e + 12u);
        if (size == 0u || size > (1ull << 32) || page < 0x10u) continue;
        out->size[out->n] = size;
        out->page[out->n] = page;
        out->n++;
        out->length += size;
        if (flags & 0x00010000u) break;
    }
    return out->n != 0u;
}

/* Read `len` bytes at logical offset `off` of the buffer; past the end reads zeros. */
static void read_buffer(lagfx_protocol_t *p, const lagfx_task_entry_t *task,
                        const placement_t *pl, uint64_t off, uint64_t len, uint8_t *dst) {
    memset(dst, 0, (size_t)len);
    uint64_t base = 0;
    for (uint32_t e = 0; e < pl->n && len > 0u; e++) {
        if (off >= base + pl->size[e]) { base += pl->size[e]; continue; }
        uint64_t local = off - base;
        uint64_t n = pl->size[e] - local;
        if (n > len) n = len;
        for (uint64_t done = 0; done < n;) {
            uint64_t va = (pl->page[e] << 12) + local + done;
            uint32_t chunk = (uint32_t)((n - done) > 0x10000u ? 0x10000u : (n - done));
            (void)lagfx_task_read_virtual(p, task, va, chunk, dst + done);
            done += chunk;
        }
        dst += n; off += n; len -= n;
        base += pl->size[e];
    }
}

/* Neutral stand-ins, made once. */
static VkBuffer g_zero_buf, g_sink_buf, g_status_buf;
static VkDeviceMemory g_zero_mem, g_sink_mem, g_status_mem;
static VkSampler g_neutral_sampler;

static VkBuffer neutral_buffer(struct lagfx_vk_state *vk) {
    if (!g_zero_buf) make_buffer(vk, NULL, 1u << 16, &g_zero_buf, &g_zero_mem);
    return g_zero_buf;
}

static uint64_t sink_address(struct lagfx_vk_state *vk) {
    if (!g_sink_buf) make_buffer(vk, NULL, 4096u, &g_sink_buf, &g_sink_mem);
    return g_sink_buf ? buffer_address(vk, g_sink_buf) : 0u;
}

static uint64_t status_address(struct lagfx_vk_state *vk) {
    if (!g_status_buf) make_buffer(vk, NULL, 4096u, &g_status_buf, &g_status_mem);
    return g_status_buf ? buffer_address(vk, g_status_buf) : 0u;
}


/* A neutral image of any view type: 1x1(x1), zeros, one level, the layers the
 * view type needs; sampled (SHADER_READ_ONLY) or storage (GENERAL). Cached. */
typedef struct {
    uint32_t view_type, sampled_type, format, compare;
    VkImage image;
    VkDeviceMemory mem;
    VkImageView view;
    VkImageLayout layout;
} neutral_img_t;

static neutral_img_t g_nimg[32];
static uint32_t g_nnimg;

static bool submit_once(struct lagfx_vk_state *vk, VkCommandBuffer cb) {
    VkFence f = VK_NULL_HANDLE;
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                        .pCommandBuffers = &cb };
    bool ok = vkCreateFence(vk->device, &fci, NULL, &f) == VK_SUCCESS
              && vkQueueSubmit(vk->graphics_queue, 1, &si, f) == VK_SUCCESS
              && vkWaitForFences(vk->device, 1, &f, VK_TRUE, 2000000000ull) == VK_SUCCESS;
    if (f) vkDestroyFence(vk->device, f, NULL);
    return ok;
}

static const neutral_img_t *neutral_image_any(struct lagfx_vk_state *vk, uint32_t view_type,
                                              uint32_t sampled_type, uint32_t storage_format,
                                              uint32_t compare) {
    for (uint32_t i = 0; i < g_nnimg; i++)
        if (g_nimg[i].view_type == view_type && g_nimg[i].sampled_type == sampled_type
            && g_nimg[i].format == storage_format && g_nimg[i].compare == compare)
            return &g_nimg[i];
    if (g_nnimg >= 32u) return NULL;
    static const VkFormat by_type[4] = { VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_SINT,
                                         VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32_UINT };
    VkFormat fmt = storage_format ? (VkFormat)storage_format
                 : compare ? VK_FORMAT_D32_SFLOAT : by_type[sampled_type < 4u ? sampled_type : 0u];
    VkImageAspectFlags aspect = compare ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageType itype = VK_IMAGE_TYPE_2D;
    uint32_t layers = 1;
    VkImageCreateFlags flags = 0;
    switch ((VkImageViewType)view_type) {
    case VK_IMAGE_VIEW_TYPE_1D: case VK_IMAGE_VIEW_TYPE_1D_ARRAY: itype = VK_IMAGE_TYPE_1D; break;
    case VK_IMAGE_VIEW_TYPE_3D: itype = VK_IMAGE_TYPE_3D; break;
    case VK_IMAGE_VIEW_TYPE_CUBE: layers = 6; flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT; break;
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: layers = 6; flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT; break;
    default: break;
    }
    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                            | (storage_format ? VK_IMAGE_USAGE_STORAGE_BIT : 0u);
    neutral_img_t n = { view_type, sampled_type, storage_format, compare };
    n.layout = storage_format ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = flags, .imageType = itype,
        .format = fmt, .extent = { 1, 1, 1 }, .mipLevels = 1, .arrayLayers = layers,
        .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    if (vkCreateImage(vk->device, &ici, NULL, &n.image) != VK_SUCCESS) return NULL;
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(vk->device, n.image, &mr);
    VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                .allocationSize = mr.size,
                                .memoryTypeIndex = find_mem(vk, mr.memoryTypeBits, 0) };
    if (vkAllocateMemory(vk->device, &ai, NULL, &n.mem) != VK_SUCCESS) return NULL;
    vkBindImageMemory(vk->device, n.image, n.mem, 0);
    VkImageSubresourceRange range = { aspect, 0, 1, 0, layers };
    VkCommandBuffer cb = VK_NULL_HANDLE;
    if (lagfx_vk_cmdbuf_alloc(vk, &cb) != LAGFX_OK) return NULL;
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkBeginCommandBuffer(cb, &bi);
    VkImageMemoryBarrier b = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = n.image, .subresourceRange = range,
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, NULL, 0, NULL, 1, &b);
    if (compare) {
        VkClearDepthStencilValue z = { 0 };
        vkCmdClearDepthStencilImage(cb, n.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &z, 1, &range);
    } else {
        VkClearColorValue z = { { 0 } };
        vkCmdClearColorImage(cb, n.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &z, 1, &range);
    }
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout = n.layout;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, 0, NULL, 0, NULL, 1, &b);
    vkEndCommandBuffer(cb);
    bool ok = submit_once(vk, cb);
    lagfx_vk_cmdbuf_free(vk, cb);
    if (!ok) return NULL;
    VkImageViewCreateInfo vci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = n.image,
        .viewType = (VkImageViewType)view_type, .format = fmt, .subresourceRange = range,
    };
    if (vkCreateImageView(vk->device, &vci, NULL, &n.view) != VK_SUCCESS) return NULL;
    g_nimg[g_nnimg] = n;
    return &g_nimg[g_nnimg++];
}

/* Textures that aren't single-layer 2D (1D, arrays, 3D, cubes): an image of
 * their own type and layers, contents uploaded once from guest memory (all
 * layers, back to back), cached by (task, object, memory). */
typedef struct {
    uint32_t task, ref;
    uint64_t where;             /* texel (0,0): placement address + offset */
    VkImage image;
    VkDeviceMemory mem;
    VkFormat format;
    uint32_t layers;
    uint32_t gen;            /* task command buffer the contents were read for */
} other_tex_t;

static other_tex_t g_other[64];
static uint32_t g_nother;

static const other_tex_t *other_texture(lagfx_protocol_t *p, lagfx_task_entry_t *task,
                                        struct lagfx_vk_state *vk, uint32_t ref) {
    lagfx_texture_object_t o;
    if (!lagfx_guest_texture_info(p, task, ref, &o) || !o.n_place || o.surface_id) return NULL;
    other_tex_t *cached = NULL;
    uint64_t where = (o.place[0].page << 12) + o.base_offset;
    for (uint32_t i = 0; i < g_nother; i++)
        if (g_other[i].task == task->id && g_other[i].ref == ref && g_other[i].where == where)
            cached = &g_other[i];
    /* contents are re-read once per command buffer: the guest may fill the
     * texture after creating it */
    if (cached && cached->gen == task->cmdbuf_gen + 1u) return cached;
    if (!cached && g_nother >= 64u) return NULL;
    uint32_t bpp = lagfx_metal_pixel_format_bytes(o.pixel_format);
    VkFormat fmt = lagfx_metal_pixel_format_to_vk(o.pixel_format);
    if (!bpp || fmt == VK_FORMAT_UNDEFINED) return NULL;
    VkImageType itype = VK_IMAGE_TYPE_2D;
    VkImageCreateFlags flags = 0;
    uint32_t layers = o.array_length ? o.array_length : 1u;
    uint32_t depth = 1u;
    switch (o.texture_type) {
    case 0: case 1: itype = VK_IMAGE_TYPE_1D; break;                       /* 1D, 1D array */
    case 5: case 6: flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT; layers *= 6u; break;
    case 7: itype = VK_IMAGE_TYPE_3D; depth = o.depth ? o.depth : 1u; break;
    default: break;
    }
    uint64_t row = (uint64_t)o.width * bpp;
    if (o.bytes_per_row < row) return NULL;
    uint64_t slice = (uint64_t)o.bytes_per_row * o.height;          /* one layer / depth slice */
    uint32_t nslices = itype == VK_IMAGE_TYPE_3D ? depth : layers;
    uint64_t total = row * o.height * nslices;
    if (total > (64u << 20)) return NULL;
    uint8_t *px = malloc((size_t)total);
    if (!px) return NULL;
    placement_t pl = { .n = 1, .size = { o.place[0].size }, .page = { o.place[0].page },
                       .length = o.place[0].size };
    for (uint32_t z = 0; z < nslices; z++)
        for (uint32_t y = 0; y < o.height; y++)
            read_buffer(p, task, &pl, o.base_offset + z * slice + (uint64_t)y * o.bytes_per_row, row,
                        px + ((size_t)z * o.height + y) * row);
    if (getenv("LAGFX_SRCCHECK")) {
        size_t nz = 0;
        for (size_t k = 0; k < total; k++) nz += px[k] != 0u;
        LAGFX_WARN("SRCCHECK other texture 0x%x type %u %ux%u layers %u: %zu/%llu bytes non-zero",
                   ref, o.texture_type, o.width, o.height, layers, nz, (unsigned long long)total);
    }
    if (cached) {
        /* replace the old image (no draw is in flight: draws are synchronous) */
        vkDestroyImage(vk->device, cached->image, NULL);
        vkFreeMemory(vk->device, cached->mem, NULL);
        cached->image = VK_NULL_HANDLE;
        cached->mem = VK_NULL_HANDLE;
    }
    other_tex_t t = { task->id, ref, where, VK_NULL_HANDLE, VK_NULL_HANDLE, fmt, layers,
                      task->cmdbuf_gen + 1u };
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = flags, .imageType = itype,
        .format = fmt, .extent = { o.width, itype == VK_IMAGE_TYPE_1D ? 1u : o.height, depth },
        .mipLevels = 1, .arrayLayers = itype == VK_IMAGE_TYPE_3D ? 1u : layers,
        .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
    };
    VkBuffer sb = VK_NULL_HANDLE;
    VkDeviceMemory sm = VK_NULL_HANDLE;
    bool ok = vkCreateImage(vk->device, &ici, NULL, &t.image) == VK_SUCCESS;
    if (ok) {
        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(vk->device, t.image, &mr);
        VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                    .allocationSize = mr.size,
                                    .memoryTypeIndex = find_mem(vk, mr.memoryTypeBits, 0) };
        ok = vkAllocateMemory(vk->device, &ai, NULL, &t.mem) == VK_SUCCESS
             && vkBindImageMemory(vk->device, t.image, t.mem, 0) == VK_SUCCESS
             && make_buffer(vk, px, total, &sb, &sm);
    }
    free(px);
    VkCommandBuffer cb = VK_NULL_HANDLE;
    if (ok) ok = lagfx_vk_cmdbuf_alloc(vk, &cb) == LAGFX_OK;
    if (ok) {
        VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        vkBeginCommandBuffer(cb, &bi);
        VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, ici.arrayLayers };
        VkImageMemoryBarrier b = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = t.image, .subresourceRange = range,
        };
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, NULL, 0, NULL, 1, &b);
        VkBufferImageCopy c = {
            .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, ici.arrayLayers },
            .imageExtent = ici.extent,
        };
        vkCmdCopyBufferToImage(cb, sb, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 0, NULL, 0, NULL, 1, &b);
        vkEndCommandBuffer(cb);
        ok = submit_once(vk, cb);
        lagfx_vk_cmdbuf_free(vk, cb);
    }
    if (sb) { vkDestroyBuffer(vk->device, sb, NULL); vkFreeMemory(vk->device, sm, NULL); }
    if (!ok) {
        if (t.image) vkDestroyImage(vk->device, t.image, NULL);
        if (t.mem) vkFreeMemory(vk->device, t.mem, NULL);
        if (cached) *cached = (other_tex_t){ 0 };
        return NULL;
    }
    LAGFX_LOG("M2V: texture 0x%x (task %u) type %u %ux%ux%u layers %u uploaded", ref, task->id,
              o.texture_type, o.width, o.height, depth, layers);
    if (cached) {
        *cached = t;
        return cached;
    }
    g_other[g_nother] = t;
    return &g_other[g_nother++];
}

static VkSampler neutral_sampler(struct lagfx_vk_state *vk) {
    if (!g_neutral_sampler) {
        lagfx_m2v_vk_sampler_t s = { .max_anisotropy = 1.0f, .max_lod = 0.25f };
        g_neutral_sampler = make_sampler(vk, &s);
    }
    return g_neutral_sampler;
}

/* ----------------------------------------------------------------- pipelines */

static void pipe_destroy(struct lagfx_vk_state *vk, m2v_pipe_t *mp) {
    if (!mp) return;
    for (uint32_t i = 0; i < mp->nvar; i++) vkDestroyPipeline(vk->device, mp->var[i].pipeline, NULL);
    for (uint32_t i = 0; i < 2u; i++) if (mp->mod[i]) vkDestroyShaderModule(vk->device, mp->mod[i], NULL);
    if (mp->layout) vkDestroyPipelineLayout(vk->device, mp->layout, NULL);
    for (uint32_t i = 0; i < mp->n_sets; i++)
        if (mp->dsl[i]) vkDestroyDescriptorSetLayout(vk->device, mp->dsl[i], NULL);
    for (uint32_t i = 0; i < mp->nchild; i++) pipe_destroy(vk, mp->child[i].mp);
    if (mp->owns_pipe) lagfx_m2v_pipeline_free(mp->pipe);
    free(mp);
}

static m2v_pipe_t *pipe_create(struct lagfx_vk_state *vk, lagfx_m2v_pipeline_t *pipe) {
    m2v_pipe_t *mp = calloc(1, sizeof *mp);
    if (!mp) return NULL;
    mp->pipe = pipe;
    mp->owns_pipe = true;
    uint32_t total = lagfx_m2v_pipeline_layout(pipe, mp->b, MAX_LAYOUT, &mp->push_size,
                                               &mp->push_stages);
    if (total > MAX_LAYOUT) { LAGFX_WARN("M2V: layout of %u bindings too large", total); goto fail; }
    mp->nb = total;
    for (uint32_t i = 0; i < mp->nb; i++) {
        if (mp->b[i].set >= MAX_SETS) { LAGFX_WARN("M2V: set %u unsupported", mp->b[i].set); goto fail; }
        if (mp->b[i].set + 1u > mp->n_sets) mp->n_sets = mp->b[i].set + 1u;
        if (mp->b[i].has_immutable_sampler)
            mp->immutable[i] = make_sampler(vk, &mp->b[i].immutable_sampler);
    }
    for (uint32_t s = 0; s < mp->n_sets; s++) {
        VkDescriptorSetLayoutBinding lb[MAX_LAYOUT];
        VkDescriptorBindingFlags fl[MAX_LAYOUT];
        uint32_t n = 0;
        bool any_flags = false;
        for (uint32_t i = 0; i < mp->nb; i++) {
            if (mp->b[i].set != s) continue;
            lb[n] = (VkDescriptorSetLayoutBinding){
                .binding = mp->b[i].binding,
                .descriptorType = (VkDescriptorType)mp->b[i].descriptor_type,
                .descriptorCount = mp->b[i].count,
                .stageFlags = mp->b[i].stages,
                .pImmutableSamplers = mp->immutable[i] ? &mp->immutable[i] : NULL,
            };
            fl[n] = mp->b[i].flags;
            any_flags |= fl[n] != 0u;
            n++;
        }
        VkDescriptorSetLayoutBindingFlagsCreateInfo fci = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
            .bindingCount = n,
            .pBindingFlags = fl,
        };
        VkDescriptorSetLayoutCreateInfo ci = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = any_flags ? &fci : NULL,
            .bindingCount = n,
            .pBindings = lb,
        };
        if (vkCreateDescriptorSetLayout(vk->device, &ci, NULL, &mp->dsl[s]) != VK_SUCCESS) goto fail;
    }
    VkPushConstantRange pr = { .stageFlags = mp->push_stages, .offset = 0, .size = mp->push_size };
    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = mp->n_sets,
        .pSetLayouts = mp->dsl,
        .pushConstantRangeCount = mp->push_size ? 1u : 0u,
        .pPushConstantRanges = &pr,
    };
    if (vkCreatePipelineLayout(vk->device, &plci, NULL, &mp->layout) != VK_SUCCESS) goto fail;
    mp->n_stages = lagfx_m2v_pipeline_stage_count(pipe);
    if (mp->n_stages < 1u || mp->n_stages > 2u) goto fail;
    for (uint32_t i = 0; i < mp->n_stages; i++) {
        size_t nw = 0;
        const uint32_t *w = lagfx_m2v_pipeline_spirv(pipe, i, &nw);
        VkShaderModuleCreateInfo sci = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = nw * 4u,
            .pCode = w,
        };
        if (!w || vkCreateShaderModule(vk->device, &sci, NULL, &mp->mod[i]) != VK_SUCCESS) goto fail;
        mp->stage_flags[i] = lagfx_m2v_pipeline_entry(pipe, i, mp->entry[i], sizeof mp->entry[i]);
    }
    mp->nvb = lagfx_m2v_pipeline_vertex_input(pipe, mp->vb, mp->va, &mp->nva);
    return mp;
fail:
    pipe_destroy(vk, mp);
    return NULL;
}

void *lagfx_m2v_pipeline_get(lagfx_protocol_t *p, lagfx_task_entry_t *task,
                             struct lagfx_vk_state *vk, uint32_t reference,
                             const uint8_t *vlib, size_t vlen,
                             const uint8_t *flib, size_t flen) {
    if (!vk || !vk->m2v || !vlib) return NULL;
    lagfx_m2v_render_desc_t d = {
        .vertex_lib = vlib, .vertex_lib_len = vlen,
        .fragment_lib = flib, .fragment_lib_len = flib ? flen : 0u,
        .color_format = 80u /* BGRA8Unorm */, .write_mask = 0xfu,
        .src_rgb = 1u, .src_alpha = 1u,
    };
    if (task->pending_pipeline.has_blend) {
        const lagfx_pso_color_attachment_t *b = &task->pending_pipeline.blend;
        if (b->pixel_format) d.color_format = b->pixel_format;
        d.blending = b->blending;
        d.src_rgb = b->src_rgb; d.dst_rgb = b->dst_rgb; d.rgb_op = b->rgb_op;
        d.src_alpha = b->src_alpha; d.dst_alpha = b->dst_alpha; d.alpha_op = b->alpha_op;
        d.write_mask = b->write_mask;
    }
    uint8_t blob[512];
    lagfx_vtxdesc_attr_t va[8];
    lagfx_vtxdesc_layout_t vl[8];
    uint32_t nl = 0;
    uint32_t bl = lagfx_read_pso_blob(p, task, reference, blob, sizeof blob, NULL);
    uint32_t na = bl ? lagfx_parse_vertex_descriptor(blob, bl, va, 8u, vl, 8u, &nl) : 0u;
    lagfx_m2v_vertex_attribute_t ma[8];
    lagfx_m2v_vertex_layout_t ml[8];
    for (uint32_t i = 0; i < na; i++)
        ma[i] = (lagfx_m2v_vertex_attribute_t){ va[i].index, va[i].fmt, va[i].off, va[i].bufidx };
    for (uint32_t i = 0; i < nl; i++) {
        /* layout record key 1: MTLVertexStepFunction (absent: PerVertex), key 2:
         * step rate (absent: 1); kat vtxstep-probe */
        ml[i] = (lagfx_m2v_vertex_layout_t){
            vl[i].index, vl[i].stride,
            (vl[i].present & 2u) ? vl[i].value[1] : 1u,
            (vl[i].present & 4u) ? vl[i].value[2] : 1u };
        if (getenv("LAGFX_VTXLOG"))
            LAGFX_WARN("VTXLAYOUT ref=0x%x buffer %u stride %u keys 0x%x: [1]=%u [2]=%u [4]=%u [5]=%u",
                       reference, vl[i].index, vl[i].stride, vl[i].present, vl[i].value[1],
                       vl[i].value[2], vl[i].value[4], vl[i].value[5]);
    }
    if (na) {
        d.has_vertex_descriptor = 1u;
        d.n_attributes = na; d.attributes = ma;
        d.n_layouts = nl; d.layouts = ml;
    }

    uint64_t key = fnv(0xcbf29ce484222325ull, vlib, vlen);
    if (flib) key = fnv(key, flib, flen);
    key = fnv(key, &d.color_format, sizeof(uint32_t) * 9u);   /* colour format .. write mask */
    key = fnv(key, ma, sizeof ma[0] * na);
    key = fnv(key, ml, sizeof ml[0] * nl);
    for (uint32_t i = 0; i < g_npipes && i < MAX_PIPES; i++)
        if (g_pipes[i].key == key) return g_pipes[i].pipe;

    char err[512] = "";
    lagfx_m2v_pipeline_t *mp = lagfx_m2v_compile_render((lagfx_m2v_device_t *)vk->m2v, &d, err, sizeof err);
    m2v_pipe_t *out = NULL;
    if (!mp) {
        LAGFX_WARN("M2V pipe=0x%x refused: %s", reference, err);
    } else if (!(out = pipe_create(vk, mp))) {
        LAGFX_WARN("M2V pipe=0x%x: Vulkan objects failed", reference);
    } else {
        out->key = key;
        out->blend = task->pending_pipeline.blend;
        out->has_blend = task->pending_pipeline.has_blend;
    }
    uint32_t slot = g_npipes < MAX_PIPES ? g_npipes++ : (uint32_t)(key % MAX_PIPES);
    if (g_npipes > MAX_PIPES) g_npipes = MAX_PIPES;
    if (g_pipes[slot].pipe && slot != g_npipes - 1u) pipe_destroy(vk, g_pipes[slot].pipe);
    g_pipes[slot].key = key;
    g_pipes[slot].pipe = out;
    return out;
}

/* MTLBlendFactor -> VkBlendFactor */
static VkBlendFactor blend_factor(uint8_t f) {
    static const VkBlendFactor map[] = {
        VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_SRC_COLOR,
        VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR, VK_BLEND_FACTOR_SRC_ALPHA,
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_FACTOR_DST_COLOR,
        VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR, VK_BLEND_FACTOR_DST_ALPHA,
        VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA, VK_BLEND_FACTOR_SRC_ALPHA_SATURATE,
        VK_BLEND_FACTOR_CONSTANT_COLOR, VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
        VK_BLEND_FACTOR_CONSTANT_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA,
        VK_BLEND_FACTOR_SRC1_COLOR, VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR,
        VK_BLEND_FACTOR_SRC1_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA,
    };
    return f < sizeof map / sizeof map[0] ? map[f] : VK_BLEND_FACTOR_ONE;
}

static VkBlendOp blend_op(uint8_t o) {
    switch (o) {
    case 1: return VK_BLEND_OP_SUBTRACT;
    case 2: return VK_BLEND_OP_REVERSE_SUBTRACT;
    case 3: return VK_BLEND_OP_MIN;
    case 4: return VK_BLEND_OP_MAX;
    default: return VK_BLEND_OP_ADD;
    }
}

static VkPipeline variant(struct lagfx_vk_state *vk, m2v_pipe_t *mp, VkFormat fmt, uint32_t topology,
                          const uint8_t *spec, size_t spec_len, const uint32_t (*entries)[3],
                          uint32_t n_entries, VkFormat sfmt, const lagfx_depth_stencil_t *dss) {
    uint64_t sh = fnv(0xcbf29ce484222325ull, spec, spec_len);
    sh = fnv(sh, entries, sizeof entries[0] * n_entries);
    uint64_t dh = sfmt != VK_FORMAT_UNDEFINED ? fnv(0xcbf29ce484222325ull, dss, sizeof *dss) : 0u;
    for (uint32_t i = 0; i < mp->nvar; i++)
        if (mp->var[i].format == fmt && mp->var[i].topology == topology && mp->var[i].spec_hash == sh
            && mp->var[i].stencil_format == sfmt && mp->var[i].ds_hash == dh)
            return mp->var[i].pipeline;
    if (mp->nvar >= MAX_VARIANTS) return VK_NULL_HANDLE;

    VkSpecializationMapEntry me[64];
    uint32_t nme = n_entries < 64u ? n_entries : 64u;
    for (uint32_t i = 0; i < nme; i++)
        me[i] = (VkSpecializationMapEntry){ entries[i][0], entries[i][1], entries[i][2] };
    VkSpecializationInfo si = { nme, me, spec_len, spec };
    VkPipelineShaderStageCreateInfo st[2];
    for (uint32_t i = 0; i < mp->n_stages; i++)
        st[i] = (VkPipelineShaderStageCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = (VkShaderStageFlagBits)mp->stage_flags[i],
            .module = mp->mod[i],
            .pName = mp->entry[i],
            .pSpecializationInfo = nme ? &si : NULL,
        };
    VkVertexInputBindingDescription vbd[16];
    VkVertexInputAttributeDescription vad[16];
    for (uint32_t i = 0; i < mp->nvb; i++)
        vbd[i] = (VkVertexInputBindingDescription){ mp->vb[i].binding, mp->vb[i].stride,
                                                    (VkVertexInputRate)mp->vb[i].input_rate };
    for (uint32_t i = 0; i < mp->nva; i++)
        vad[i] = (VkVertexInputAttributeDescription){ mp->va[i].location, mp->va[i].binding,
                                                      (VkFormat)mp->va[i].format, mp->va[i].offset };
    VkPipelineVertexInputStateCreateInfo vis = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = mp->nvb, .pVertexBindingDescriptions = vbd,
        .vertexAttributeDescriptionCount = mp->nva, .pVertexAttributeDescriptions = vad,
    };
    VkPipelineInputAssemblyStateCreateInfo ias = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = (VkPrimitiveTopology)topology,
    };
    VkPipelineViewportStateCreateInfo vps = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1, .scissorCount = 1,
    };
    VkPipelineRasterizationStateCreateInfo rs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .lineWidth = 1.0f,
    };
    VkPipelineMultisampleStateCreateInfo ms = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    const lagfx_pso_color_attachment_t *b = &mp->blend;
    VkPipelineColorBlendAttachmentState cba = {
        .blendEnable = mp->has_blend && b->blending ? VK_TRUE : VK_FALSE,
        .srcColorBlendFactor = blend_factor(mp->has_blend ? b->src_rgb : 1),
        .dstColorBlendFactor = blend_factor(mp->has_blend ? b->dst_rgb : 0),
        .colorBlendOp = blend_op(mp->has_blend ? b->rgb_op : 0),
        .srcAlphaBlendFactor = blend_factor(mp->has_blend ? b->src_alpha : 1),
        .dstAlphaBlendFactor = blend_factor(mp->has_blend ? b->dst_alpha : 0),
        .alphaBlendOp = blend_op(mp->has_blend ? b->alpha_op : 0),
        /* MTLColorWriteMask R8 G4 B2 A1 -> VkColorComponentFlags R1 G2 B4 A8 */
        .colorWriteMask = mp->has_blend
            ? (((b->write_mask >> 3) & 1u) | (((b->write_mask >> 2) & 1u) << 1)
               | (((b->write_mask >> 1) & 1u) << 2) | ((b->write_mask & 1u) << 3))
            : 0xfu,
    };
    VkPipelineColorBlendStateCreateInfo cbs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1, .pAttachments = &cba,
    };
    VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                             VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_STENCIL_REFERENCE };
    bool stencil = sfmt != VK_FORMAT_UNDEFINED;
    VkPipelineDynamicStateCreateInfo ds = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = stencil ? 4u : 3u, .pDynamicStates = dyn,
    };
    /* Metal's compare functions and stencil operations are numbered as Vulkan's */
    VkPipelineDepthStencilStateCreateInfo dsi = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .stencilTestEnable = stencil ? VK_TRUE : VK_FALSE,
    };
    if (stencil) {
        const lagfx_stencil_face_t *f[2] = { &dss->front, &dss->back };
        VkStencilOpState *o[2] = { &dsi.front, &dsi.back };
        for (int k = 0; k < 2; k++)
            *o[k] = (VkStencilOpState){ (VkStencilOp)f[k]->stencil_fail, (VkStencilOp)f[k]->pass,
                                        (VkStencilOp)f[k]->depth_fail, (VkCompareOp)f[k]->compare,
                                        f[k]->read_mask, f[k]->write_mask, 0u };
    }
    VkPipelineRenderingCreateInfo rci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1, .pColorAttachmentFormats = &fmt,
        .stencilAttachmentFormat = sfmt,
    };
    VkGraphicsPipelineCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &rci,
        .stageCount = mp->n_stages, .pStages = st,
        .pVertexInputState = &vis, .pInputAssemblyState = &ias, .pViewportState = &vps,
        .pRasterizationState = &rs, .pMultisampleState = &ms, .pColorBlendState = &cbs,
        .pDepthStencilState = stencil ? &dsi : NULL,
        .pDynamicState = &ds, .layout = mp->layout,
    };
    VkPipeline pl = VK_NULL_HANDLE;
    VkResult vr = vkCreateGraphicsPipelines(vk->device, VK_NULL_HANDLE, 1, &ci, NULL, &pl);
    if (vr != VK_SUCCESS) {
        LAGFX_WARN("M2V: vkCreateGraphicsPipelines failed (%d)", (int)vr);
        return VK_NULL_HANDLE;
    }
    mp->var[mp->nvar++] = (variant_t){ fmt, topology, sh, sfmt, dh, pl };
    return pl;
}

/* ------------------------------------------------------------------- stencil */

/* Live tracing (diagnostic): /tmp/lagfx_trace.txt holds pipeline refs in the
 * LAGFX_SKIP_PIPES format; draws with those pipelines log their bindings as
 * LAGFX_BINDLOG does, plus each texture object. Re-read every draw, since
 * pipeline refs differ per boot. */
static bool traced_pipeline(uint32_t ref) {
    char buf[256], pref[16];
    FILE *f = fopen("/tmp/lagfx_trace.txt", "r");
    if (!f) return false;
    size_t got = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[got] = '\0';
    snprintf(pref, sizeof pref, "0x%x", ref);
    size_t pl = strlen(pref);
    for (const char *h = strstr(buf, pref); h; h = strstr(h + 1, pref))
        if ((h == buf || h[-1] == ',') && (h[pl] == '\0' || h[pl] == ',' || h[pl] == '\n'))
            return true;
    return false;
}

static void trace_texture(lagfx_protocol_t *p, lagfx_task_entry_t *task, int stage,
                          uint32_t index, uint32_t ref) {
    lagfx_texture_object_t o;
    uint8_t type = 0, raw[0x80];
    uint64_t va = 0, gpa = 0;
    bool have = lagfx_guest_texture_info(p, task, ref, &o);
    bool rd = lagfx_resolve_object_data(p, task, ref, &type, &va, &gpa) && va
              && lagfx_task_read_virtual(p, task, va, sizeof raw, raw);
    if (have)
        LAGFX_WARN("TRACE task=%u pipe=0x%x stage%d texture[%u] ref=0x%x type=0x%02x %ux%u fmt=%u "
                   "ttype=%u bpr=%u offset=0x%llx surface=%u places=%u {0x%llx@0x%llx}",
                   task->id, task->pending_pipeline.reference, stage, index, ref, type, o.width, o.height,
                   o.pixel_format, o.texture_type, o.bytes_per_row,
                   (unsigned long long)o.base_offset, o.surface_id, o.n_place,
                   (unsigned long long)o.place[0].size, (unsigned long long)o.place[0].page);
    else
        LAGFX_WARN("TRACE pipe=0x%x stage%d texture[%u] ref=0x%x type=0x%02x undecoded",
                   task->pending_pipeline.reference, stage, index, ref, type);
    for (uint32_t row = 0; rd && row < sizeof raw; row += 32u) {
        char hx[3 * 32 + 1];
        size_t hn = 0;
        for (uint32_t k = row; k < row + 32u; k++)
            hn += (size_t)snprintf(hx + hn, sizeof hx - hn, "%02x ", raw[k]);
        LAGFX_WARN("TRACE   +%02x: %s", row, hx);
    }
}

/* A host stencil image per (task, stencil texture, size). Its contents live
 * only on the host: the clock's renderer stores none (store DontCare). */
typedef struct {
    uint32_t task, ref, width, height;
    VkFormat format;
    VkImage image;
    VkDeviceMemory mem;
    VkImageView view;
    VkImageLayout layout;
} stencil_img_t;

static stencil_img_t g_stencil[32];
static uint32_t g_nstencil;

static VkFormat stencil_format(struct lagfx_vk_state *vk) {
    static VkFormat f = VK_FORMAT_MAX_ENUM;
    if (f != VK_FORMAT_MAX_ENUM) return f;
    const VkFormat cand[] = { VK_FORMAT_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT,
                              VK_FORMAT_D24_UNORM_S8_UINT };
    f = VK_FORMAT_UNDEFINED;
    for (size_t i = 0; i < sizeof cand / sizeof cand[0]; i++) {
        VkFormatProperties fp;
        vkGetPhysicalDeviceFormatProperties(vk->phys_device, cand[i], &fp);
        if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
            f = cand[i];
            break;
        }
    }
    LAGFX_WARN("M2V: stencil attachments use VkFormat %d", (int)f);
    return f;
}

static stencil_img_t *stencil_image(struct lagfx_vk_state *vk, uint32_t task, uint32_t ref,
                                    uint32_t w, uint32_t h) {
    for (uint32_t i = 0; i < g_nstencil; i++)
        if (g_stencil[i].task == task && g_stencil[i].ref == ref && g_stencil[i].width == w
            && g_stencil[i].height == h)
            return &g_stencil[i];
    VkFormat fmt = stencil_format(vk);
    if (fmt == VK_FORMAT_UNDEFINED) return NULL;
    stencil_img_t *e = &g_stencil[g_nstencil < 32u ? g_nstencil++ : (task + ref) % 32u];
    if (e->image) {
        vkDestroyImageView(vk->device, e->view, NULL);
        vkDestroyImage(vk->device, e->image, NULL);
        vkFreeMemory(vk->device, e->mem, NULL);
    }
    *e = (stencil_img_t){ task, ref, w, h, fmt, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE,
                          VK_IMAGE_LAYOUT_UNDEFINED };
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
        .format = fmt, .extent = { w, h, 1 }, .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkMemoryRequirements mr;
    if (vkCreateImage(vk->device, &ici, NULL, &e->image) != VK_SUCCESS) goto fail;
    vkGetImageMemoryRequirements(vk->device, e->image, &mr);
    VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                .allocationSize = mr.size,
                                .memoryTypeIndex = find_mem(vk, mr.memoryTypeBits, 0) };
    if (vkAllocateMemory(vk->device, &ai, NULL, &e->mem) != VK_SUCCESS) goto fail;
    vkBindImageMemory(vk->device, e->image, e->mem, 0);
    VkImageViewCreateInfo vci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = e->image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
        .subresourceRange = { VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1 },
    };
    if (vkCreateImageView(vk->device, &vci, NULL, &e->view) != VK_SUCCESS) goto fail;
    return e;
fail:
    if (e->image) vkDestroyImage(vk->device, e->image, NULL);
    if (e->mem) vkFreeMemory(vk->device, e->mem, NULL);
    *e = (stencil_img_t){ 0 };
    return NULL;
}

/* The task's bound depth-stencil state (Metal's default without one). */
static void bound_depth_stencil(lagfx_protocol_t *p, const lagfx_task_entry_t *task,
                                lagfx_depth_stencil_t *out) {
    lagfx_depth_stencil_default(out);
    uint8_t type = 0, d[LAGFX_DEPTH_STENCIL_OBJECT_LEN];
    uint64_t va = 0, gpa = 0;
    if (task->depth_stencil_ref
        && lagfx_resolve_object_data(p, task, task->depth_stencil_ref, &type, &va, &gpa) && va
        && lagfx_task_read_virtual(p, task, va, sizeof d, d)
        && !lagfx_parse_depth_stencil_object(d, sizeof d, out))
        lagfx_depth_stencil_default(out);
}

/* ---------------------------------------------------------------------- draw */

#define MAX_TEMP 128u

typedef struct {
    struct lagfx_vk_state *vk;
    VkBuffer buf[MAX_TEMP];
    VkDeviceMemory mem[MAX_TEMP];
    uint32_t nbuf;
    VkImageView view[MAX_TEMP];
    uint32_t nview;
} temps_t;

static VkBuffer temp_buffer(temps_t *t, const void *data, VkDeviceSize len) {
    if (t->nbuf >= MAX_TEMP) return VK_NULL_HANDLE;
    if (!make_buffer(t->vk, data, len, &t->buf[t->nbuf], &t->mem[t->nbuf])) return VK_NULL_HANDLE;
    return t->buf[t->nbuf++];
}

static void temps_free(temps_t *t) {
    for (uint32_t i = 0; i < t->nbuf; i++) {
        vkDestroyBuffer(t->vk->device, t->buf[i], NULL);
        vkFreeMemory(t->vk->device, t->mem[i], NULL);
    }
    for (uint32_t i = 0; i < t->nview; i++) vkDestroyImageView(t->vk->device, t->view[i], NULL);
}

typedef struct {
    lagfx_m2v_buffer_t buffers[LAGFX_MAX_BINDING_SLOTS];
    placement_t placement[LAGFX_MAX_BINDING_SLOTS];
    uint32_t buffer_ref[LAGFX_MAX_BINDING_SLOTS];
    uint32_t nb;
    lagfx_m2v_texture_t textures[LAGFX_MAX_BINDING_SLOTS];
    uint32_t texture_ref[LAGFX_MAX_BINDING_SLOTS];
    uint32_t nt;
    lagfx_m2v_sampler_t samplers[LAGFX_MAX_BINDING_SLOTS];
    uint32_t ns;
} stage_bind_t;

static void describe_stage(lagfx_protocol_t *p, lagfx_task_entry_t *task,
                           const lagfx_binding_slot_t *bufs, const lagfx_binding_slot_t *texs,
                           const lagfx_binding_slot_t *sams, stage_bind_t *s) {
    memset(s, 0, sizeof *s);
    for (uint32_t i = 0; i < LAGFX_MAX_BINDING_SLOTS; i++) {
        if (!bufs[i].valid || bufs[i].ref == 0u) continue;
        placement_t pl;
        if (!buffer_placement(p, task, bufs[i].ref, &pl)) continue;
        s->placement[s->nb] = pl;
        s->buffer_ref[s->nb] = bufs[i].ref;
        s->buffers[s->nb++] = (lagfx_m2v_buffer_t){
            .index = i, .length = pl.length, .offset = bufs[i].offset,
            .gpu_address = pl.page[0] << 12,
        };
    }
    for (uint32_t i = 0; texs && i < LAGFX_MAX_BINDING_SLOTS; i++) {
        if (!texs[i].valid || texs[i].ref == 0u) continue;
        lagfx_texture_object_t obj;
        if (!lagfx_guest_texture_info(p, task, texs[i].ref, &obj)) continue;
        s->texture_ref[s->nt] = texs[i].ref;
        s->textures[s->nt++] = (lagfx_m2v_texture_t){
            .index = i, .texture_type = obj.texture_type, .pixel_format = obj.pixel_format,
            .usage = obj.usage ? obj.usage : 1u,
            .mip_levels = obj.mip_levels ? obj.mip_levels : 1u,
            .array_length = obj.array_length ? obj.array_length : 1u,
            .sample_count = obj.sample_count ? obj.sample_count : 1u,
        };
    }
    for (uint32_t i = 0; sams && i < LAGFX_MAX_BINDING_SLOTS; i++) {
        if (!sams[i].valid || sams[i].ref == 0u) continue;
        uint8_t st = 0;
        uint64_t sva = 0, sgpa = 0;
        uint8_t sb[LAGFX_SAMPLER_OBJECT_LEN];
        lagfx_sampler_desc_t sd;
        if (!lagfx_resolve_object_data(p, task, sams[i].ref, &st, &sva, &sgpa) || sva == 0u
            || !lagfx_task_read_virtual(p, task, sva, sizeof sb, sb)
            || !lagfx_parse_sampler_object(sb, sizeof sb, &sd))
            continue;
        s->samplers[s->ns++] = (lagfx_m2v_sampler_t){
            .index = i, .min_filter = sd.min_filter, .mag_filter = sd.mag_filter,
            .mip_filter = sd.mip_filter, .s_address = sd.address[0], .t_address = sd.address[1],
            .r_address = sd.address[2], .border_color = sd.border_color,
            .normalized = sd.normalized, .max_anisotropy = sd.max_anisotropy,
            .lod_min = sd.lod_min, .lod_max = sd.lod_max,
        };
    }
}

static const stage_bind_t *g_stage[2];

static int find_buffer(const stage_bind_t *s, uint32_t index) {
    for (uint32_t i = 0; i < s->nb; i++) if (s->buffers[i].index == index) return (int)i;
    return -1;
}

static int find_texture(const stage_bind_t *s, uint32_t index) {
    for (uint32_t i = 0; i < s->nt; i++) if (s->textures[i].index == index) return (int)i;
    return -1;
}

/* LAGFX_SRCCHECK: read back a large texture a draw samples and log how much of
 * it is non-black (diagnosis). */
static void src_check(struct lagfx_vk_state *vk, lagfx_vk_iosurface_t *ios,
                      lagfx_task_entry_t *task, uint32_t ref, uint32_t slot) {
    lagfx_vk_render_target_t rt;
    size_t bytes = (size_t)ios->width * ios->height * 4u, stride = 0;
    uint8_t *px = malloc(bytes);
    if (!px) return;
    if (lagfx_vk_render_target_wrap(ios->image, ios->view, ios->memory, ios->width, ios->height,
                                    ios->format, &rt) == LAGFX_OK) {
        rt.layout = ios->layout;
        if (lagfx_vk_render_target_readback(vk, &rt, px, bytes, &stride) == LAGFX_OK) {
            ios->layout = rt.layout;
            if (!stride) stride = (size_t)ios->width * 4u;
            unsigned nz = 0, n = 0;
            for (uint32_t y = 0; y < ios->height; y += 8u)
                for (uint32_t xx = 0; xx < ios->width; xx += 8u) {
                    const uint8_t *q = px + (size_t)y * stride + (size_t)xx * 4u;
                    n++; nz += (q[0] | q[1] | q[2]) != 0u;
                }
            LAGFX_WARN("SRCCHECK pipe=0x%x slot %u ref 0x%x %ux%u nonblack=%u%% drawn=%u gen=%u/%u",
                       task->pending_pipeline.reference, slot, ref, ios->width, ios->height,
                       n ? 100u * nz / n : 0u, ios->gpu_drawn, ios->upload_gen, task->cmdbuf_gen);
        }
    }
    free(px);
}

#define FAIL_AT(n) do { why = #n; goto out; } while (0)

bool lagfx_m2v_draw(lagfx_protocol_t *p, lagfx_task_entry_t *task, struct lagfx_vk_state *vk,
                    void *pipeline, lagfx_vk_iosurface_t *target, uint32_t *target_layout,
                    const lagfx_draw_region_t *region, uint32_t prim, uint32_t vertex_start,
                    uint32_t vertex_count, uint32_t instance_count) {
    m2v_pipe_t *mp = pipeline;
    if (!p || !task || !vk || !mp || !target || !target_layout) return false;
    static stage_bind_t sb[2];
    describe_stage(p, task, task->bindings.vertex_buffers, task->bindings.vertex_textures, NULL, &sb[0]);
    describe_stage(p, task, task->bindings.fragment_buffers, task->bindings.fragment_textures,
                   task->bindings.fragment_samplers, &sb[1]);
    g_stage[0] = &sb[0];
    g_stage[1] = &sb[1];
    lagfx_m2v_stage_t st[2];
    for (int i = 0; i < 2; i++)
        st[i] = (lagfx_m2v_stage_t){ sb[i].nb, sb[i].buffers, sb[i].nt, sb[i].textures,
                                     sb[i].ns, sb[i].samplers };
    char err[512] = "";
    temps_t t = { .vk = vk };
    /* an indexed draw's index buffer */
    lagfx_m2v_indexed_t ix = { 0 };
    placement_t ipl;
    uint64_t index_source_offset = 0;
    uint32_t index_source_length = 0;
    const lagfx_m2v_indexed_t *ixp = NULL;
    if (task->pending_draw.indexed && getenv("LAGFX_DISABLE_M2V_INDEXED") == NULL) {
        if (!buffer_placement(p, task, task->pending_draw.index_buffer_ref, &ipl)) return false;
        ix.index_type = task->pending_draw.index_type;
        uint64_t index_bytes = (uint64_t)task->pending_draw.index_count
                             * (ix.index_type == 1u ? 4u : 2u);
        if (index_bytes == 0u || index_bytes > (16u << 20)) return false;
        /* Snapshot precisely the range named by the draw.  In WindowServer's
         * pooled buffers the object placement can be stale/smaller than the
         * command offset while opcode 0x3b supplies the current backing VA.
         * lagfx_read_vtx_source() knows that override.  Binding the snapshot
         * from offset zero also avoids inventing ring/modulo semantics. */
        index_source_offset = task->pending_draw.index_buffer_offset;
        index_source_length = (uint32_t)index_bytes;
        ix.buffer_length = index_source_length;
        ix.offset = 0;
        ix.base_vertex = task->pending_draw.base_vertex;
        /* Some late WindowServer commands retain an object whose placement no
         * longer contains the named slice.  Do not turn zero/obvious garbage
         * indices into a degenerate or wild native draw.  The old unindexed
         * interpretation is a useful per-draw fallback for these three-vertex
         * screen triangles, while valid pooled quads remain indexed. */
        uint8_t probe[24] = {0};
        uint32_t probe_len = index_source_length < sizeof probe
                           ? index_source_length : (uint32_t)sizeof probe;
        const char *probe_how = NULL;
        uint32_t probe_got = lagfx_read_vtx_source(p, task,
                                                   task->pending_draw.index_buffer_ref,
                                                   index_source_offset, probe_len, probe,
                                                   &probe_how, 0);
        bool plausible = probe_got == probe_len;
        uint32_t distinct = 0, max_index = 0;
        uint32_t probe_count = task->pending_draw.index_count;
        uint32_t max_probe_count = probe_len / (ix.index_type == 1u ? 4u : 2u);
        if (probe_count > max_probe_count) probe_count = max_probe_count;
        for (uint32_t i = 0; i < probe_count; i++) {
            uint32_t v = ix.index_type == 1u
                       ? lagfx_le32(probe + i * 4u)
                       : lagfx_le16(probe + i * 2u);
            bool seen = false;
            for (uint32_t j = 0; j < i; j++) {
                uint32_t old = ix.index_type == 1u
                             ? lagfx_le32(probe + j * 4u)
                             : lagfx_le16(probe + j * 2u);
                seen |= old == v;
            }
            distinct += !seen;
            if (v > max_index) max_index = v;
        }
        if (probe_count >= 3u && distinct < 3u) plausible = false;
        if (task->pending_draw.index_count <= 64u && max_index > 4096u) plausible = false;
        if (plausible) {
            ixp = &ix;
        } else {
            static unsigned n_bad;
            if (n_bad < 40u && getenv("LAGFX_INDEXLOG")) {
                /* lookup: the same offset read flat from the first entry's page */
                uint8_t flat[12] = { 0 };
                bool fok = ipl.n && lagfx_task_read_virtual(p, task, (ipl.page[0] << 12)
                                                            + index_source_offset, sizeof flat, flat);
                LAGFX_WARN("INDEXFLAT ref=0x%x n=%u e0=%llx@%llx e1=%llx@%llx len=%llx off=0x%llx ok=%d "
                           "[%u %u %u %u %u %u]", task->pending_draw.index_buffer_ref, ipl.n,
                           (unsigned long long)ipl.size[0], (unsigned long long)ipl.page[0],
                           (unsigned long long)ipl.size[1], (unsigned long long)ipl.page[1],
                           (unsigned long long)ipl.length, (unsigned long long)index_source_offset,
                           fok, lagfx_le16(flat), lagfx_le16(flat + 2), lagfx_le16(flat + 4),
                           lagfx_le16(flat + 6), lagfx_le16(flat + 8), lagfx_le16(flat + 10));
            }
            if (n_bad++ < 40u)
                LAGFX_WARN("INDEXSRC stale t%u ref=0x%x off=0x%llx got=%u/%u distinct=%u max=%u via=%s; unindexed fallback",
                           task->id, task->pending_draw.index_buffer_ref,
                           (unsigned long long)index_source_offset, probe_got, probe_len,
                           distinct, max_index, probe_how ? probe_how : "?");
        }
        vertex_start = 0;
    } else if (task->pending_draw.indexed) {
        /* A/B diagnostic: reproduce the pre-indexed M2V behavior exactly by
         * issuing this as an unindexed draw.  This is intentionally wrong
         * geometry, but establishes whether native indexed handling causes a
         * WindowServer regression. */
        static bool warned;
        if (!warned) {
            LAGFX_WARN("M2V indexed draws disabled; using legacy unindexed fallback");
            warned = true;
        }
        vertex_start = 0;
    }
    lagfx_m2v_draw_t *dr = lagfx_m2v_plan_draw((lagfx_m2v_device_t *)vk->m2v, mp->pipe, st,
                                               vertex_start, vertex_count,
                                               instance_count ? instance_count : 1u,
                                               task->pending_draw.first_instance, ixp,
                                               err, sizeof err);
    if (!dr && strstr(err, "MissingDeviceAddress")) {
        /* the shader reads buffers by address: give every bound buffer a host
         * copy of its whole allocation with a device address, and plan again */
        for (int i = 0; i < 2; i++)
            for (uint32_t k = 0; k < sb[i].nb; k++) {
                uint64_t len = sb[i].placement[k].length;
                if (len > (32u << 20)) len = 32u << 20;
                uint8_t *data = malloc((size_t)len ? (size_t)len : 16u);
                if (!data) continue;
                read_buffer(p, task, &sb[i].placement[k], 0u, len, data);
                VkBuffer b = temp_buffer(&t, data, len);
                free(data);
                if (b) sb[i].buffers[k].device_address = buffer_address(vk, b);
            }
        err[0] = 0;
        dr = lagfx_m2v_plan_draw((lagfx_m2v_device_t *)vk->m2v, mp->pipe, st, vertex_start,
                                 vertex_count, instance_count ? instance_count : 1u,
                                               task->pending_draw.first_instance, ixp,
                                 err, sizeof err);
    }
    if (!dr) {
        temps_free(&t);
        static unsigned n_warn;
        if (n_warn++ < 50u)
            LAGFX_WARN("M2V draw not planned: %s (indexed=%d ref=0x%x len=%llu off=%llu count=%u type=%u)",
                       err, ixp != NULL, task->pending_draw.index_buffer_ref,
                       (unsigned long long)ix.buffer_length, (unsigned long long)ix.offset,
                       vertex_count, ix.index_type);
        return false;
    }
    const lagfx_m2v_pipeline_t *used = lagfx_m2v_draw_pipeline(dr);
    if (used && used != mp->pipe) {
        /* a variant: its own Vulkan objects, made once */
        m2v_pipe_t *child = NULL;
        for (uint32_t i = 0; i < mp->nchild; i++)
            if (mp->child[i].pipe == used) child = mp->child[i].mp;
        if (!child && mp->nchild < 8u
            && (child = pipe_create(vk, (lagfx_m2v_pipeline_t *)used)) != NULL) {
            child->owns_pipe = false;
            child->blend = mp->blend;
            child->has_blend = mp->has_blend;
            mp->child[mp->nchild].pipe = used;
            mp->child[mp->nchild++].mp = child;
        }
        if (!child) {
            lagfx_m2v_draw_free(dr);
            temps_free(&t);
            return false;
        }
        mp = child;
    }

    bool ok = false;
    const char *why = "submit";
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet sets[MAX_SETS] = { 0 };
    uint32_t nw = 0;
    const lagfx_m2v_write_t *w = lagfx_m2v_draw_writes(dr, &nw);

    /* descriptor pool sized for this draw */
    VkDescriptorPoolSize ps[16];
    uint32_t nps = 0;
    for (uint32_t i = 0; i < mp->nb; i++) {
        uint32_t k = 0;
        while (k < nps && ps[k].type != (VkDescriptorType)mp->b[i].descriptor_type) k++;
        if (k == nps) { if (nps >= 16u) FAIL_AT(1); ps[nps++] = (VkDescriptorPoolSize){
                            (VkDescriptorType)mp->b[i].descriptor_type, 0 }; }
        ps[k].descriptorCount += mp->b[i].count;
    }
    if (mp->n_sets) {
        VkDescriptorPoolCreateInfo pci = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = mp->n_sets, .poolSizeCount = nps, .pPoolSizes = ps,
        };
        if (vkCreateDescriptorPool(vk->device, &pci, NULL, &pool) != VK_SUCCESS) FAIL_AT(2);
        VkDescriptorSetAllocateInfo ai = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = pool, .descriptorSetCount = mp->n_sets, .pSetLayouts = mp->dsl,
        };
        if (vkAllocateDescriptorSets(vk->device, &ai, sets) != VK_SUCCESS) FAIL_AT(3);
    }

    {
        const char *bl = getenv("LAGFX_BINDLOG");
        if ((bl && (strcmp(bl, "all") == 0
                    || strtoul(bl, NULL, 0) == task->pending_pipeline.reference))
            || traced_pipeline(task->pending_pipeline.reference)) {
            for (int si = 0; si < 2; si++)
                for (uint32_t k = 0; k < sb[si].nt; k++)
                    trace_texture(p, task, si, sb[si].textures[k].index, sb[si].texture_ref[k]);
            for (int si = 0; si < 2; si++) {
                for (uint32_t k = 0; k < sb[si].nb; k++) {
                    uint8_t head[384];
                    read_buffer(p, task, &sb[si].placement[k], sb[si].buffers[k].offset, sizeof head, head);
                    char hx[2048];
                    size_t hn = 0;
                    for (uint32_t q = 0; q < sizeof head; q += 4u) {
                        float f; memcpy(&f, head + q, 4);
                        hn += (size_t)snprintf(hx + hn, sizeof hx - hn, "%g ", f);
                        if (hn > sizeof hx - 16) break;
                    }
                    LAGFX_WARN("M2VBIND pipe=0x%x stage%d buffer[%u] ref=0x%x len=%llu off=%llu dev=%llx f32: %s",
                               task->pending_pipeline.reference, si, sb[si].buffers[k].index,
                               sb[si].buffer_ref[k], (unsigned long long)sb[si].buffers[k].length,
                               (unsigned long long)sb[si].buffers[k].offset,
                               (unsigned long long)sb[si].buffers[k].device_address, hx);
                }
                for (uint32_t k = 0; k < sb[si].ns; k++)
                    LAGFX_WARN("M2VBIND pipe=0x%x stage%d sampler[%u] min=%u mag=%u mip=%u addr=%u/%u/%u "
                               "normalized=%u border=%u lod=%g..%g",
                               task->pending_pipeline.reference, si, sb[si].samplers[k].index,
                               sb[si].samplers[k].min_filter, sb[si].samplers[k].mag_filter,
                               sb[si].samplers[k].mip_filter, sb[si].samplers[k].s_address,
                               sb[si].samplers[k].t_address, sb[si].samplers[k].r_address,
                               sb[si].samplers[k].normalized, sb[si].samplers[k].border_color,
                               (double)sb[si].samplers[k].lod_min, (double)sb[si].samplers[k].lod_max);
                for (uint32_t k = 0; k < sb[si].nt; k++)
                    LAGFX_WARN("M2VBIND pipe=0x%x stage%d texture[%u] ref=0x%x fmt=%u",
                               task->pending_pipeline.reference, si, sb[si].textures[k].index,
                               sb[si].texture_ref[k], sb[si].textures[k].pixel_format);
            }
            for (uint32_t i = 0; i < nw; i++)
                LAGFX_WARN("M2VBIND pipe=0x%x write set%u.%u type=%u kind=%u stage%u slot=%u "
                           "neutral=%u/%u view=%u fmt=%u range=%llu+%llu sampler=%u",
                           task->pending_pipeline.reference, w[i].set, w[i].binding,
                           w[i].descriptor_type, w[i].kind, w[i].stage, w[i].slot,
                           w[i].image_neutral, w[i].neutral_kind, w[i].view_type, w[i].view_format,
                           (unsigned long long)w[i].range_offset, (unsigned long long)w[i].range_size,
                           w[i].sampler_kind);
        }
    }
    VkWriteDescriptorSet wr[MAX_LAYOUT * 2];
    VkDescriptorBufferInfo bi[MAX_LAYOUT * 2];
    VkDescriptorImageInfo ii[MAX_LAYOUT * 2];
    uint32_t nwr = 0;
    for (uint32_t i = 0; i < nw && nwr < MAX_LAYOUT * 2u; i++) {
        const lagfx_m2v_write_t *x = &w[i];
        if (x->set >= mp->n_sets) FAIL_AT(4);
        VkWriteDescriptorSet *o = &wr[nwr];
        *o = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = sets[x->set],
            .dstBinding = x->binding, .dstArrayElement = x->element, .descriptorCount = 1,
            .descriptorType = (VkDescriptorType)x->descriptor_type,
        };
        bool is_image = o->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
                     || o->descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                     || o->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER
                     || o->descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        switch (x->kind) {
        case LAGFX_M2V_W_BUFFER: {
            const stage_bind_t *s = g_stage[x->stage < 2u ? x->stage : 0u];
            int k = find_buffer(s, x->slot);
            if (k < 0) FAIL_AT(5);
            uint64_t off = x->has_range ? x->range_offset : s->buffers[k].offset;
            uint64_t len = x->has_range ? x->range_size : 0x10000u;
            if (len == VK_WHOLE_SIZE || len > (64u << 20)) len = 64u << 20;
            uint8_t *data = malloc((size_t)len ? (size_t)len : 16u);
            if (!data) FAIL_AT(6);
            read_buffer(p, task, &s->placement[k], off, len, data);
            VkBuffer b = temp_buffer(&t, data, len);
            free(data);
            if (!b) FAIL_AT(7);
            bi[nwr] = (VkDescriptorBufferInfo){ b, 0, len ? len : VK_WHOLE_SIZE };
            o->pBufferInfo = &bi[nwr];
            break;
        }
        case LAGFX_M2V_W_CONSTANT_BLOB:
        case LAGFX_M2V_W_BYTES: {
            size_t len = 0;
            const uint8_t *data = x->kind == LAGFX_M2V_W_BYTES
                ? lagfx_m2v_draw_bytes(dr, (uint32_t)x->bytes_index, &len)
                : lagfx_m2v_pipeline_constant_blob(mp->pipe, x->stage, &len);
            VkBuffer b = temp_buffer(&t, data, len ? len : 16u);
            if (!b) FAIL_AT(8);
            bi[nwr] = (VkDescriptorBufferInfo){ b, 0, x->has_range ? x->range_size : VK_WHOLE_SIZE };
            o->pBufferInfo = &bi[nwr];
            break;
        }
        case LAGFX_M2V_W_TEXTURE:
        case LAGFX_M2V_W_COMBINED:
        case LAGFX_M2V_W_SAMPLER:
        case LAGFX_M2V_W_NEUTRAL: {
            VkImageView view = VK_NULL_HANDLE;
            VkImageLayout lay = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkSampler smp = VK_NULL_HANDLE;
            bool want_image = x->kind == LAGFX_M2V_W_TEXTURE || x->kind == LAGFX_M2V_W_COMBINED
                              || (x->kind == LAGFX_M2V_W_NEUTRAL && x->neutral_kind == 3u);
            if (x->kind == LAGFX_M2V_W_NEUTRAL && (x->neutral_kind == 1u || x->neutral_kind == 2u)) {
                VkBuffer b = neutral_buffer(vk);
                if (!b) FAIL_AT(9);
                bi[nwr] = (VkDescriptorBufferInfo){ b, 0, x->neutral_min_bytes ? x->neutral_min_bytes : 16u };
                o->pBufferInfo = &bi[nwr];
                break;
            }
            if (x->kind == LAGFX_M2V_W_NEUTRAL && x->neutral_kind == 4u) FAIL_AT(10); /* texel buffer */
            if (want_image) {
                bool neutral_img = x->kind == LAGFX_M2V_W_NEUTRAL || x->image_neutral;
                if (neutral_img) {
                    const neutral_img_t *ni = neutral_image_any(vk, x->neutral_view_type,
                                                                x->neutral_sampled_type,
                                                                x->neutral_format, x->neutral_compare);
                    if (!ni) FAIL_AT(12);
                    view = ni->view;
                    lay = ni->layout;
                } else {
                    const stage_bind_t *s = g_stage[x->stage < 2u ? x->stage : 0u];
                    int k = find_texture(s, x->slot);
                    if (k < 0) FAIL_AT(13);
                    if (s->textures[k].texture_type != 2u || s->textures[k].array_length > 1u) {
                        const other_tex_t *ot = other_texture(p, task, vk, s->texture_ref[k]);
                        if (!ot || (VkFormat)x->view_format != ot->format || t.nview >= MAX_TEMP)
                            FAIL_AT(29);
                        uint32_t vlayers = (x->view_type == VK_IMAGE_VIEW_TYPE_1D_ARRAY
                                            || x->view_type == VK_IMAGE_VIEW_TYPE_2D_ARRAY
                                            || x->view_type == VK_IMAGE_VIEW_TYPE_CUBE
                                            || x->view_type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY)
                                           ? ot->layers : 1u;
                        VkImageViewCreateInfo vci = {
                            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = ot->image,
                            .viewType = (VkImageViewType)x->view_type, .format = ot->format,
                            .components = { x->swizzle[0], x->swizzle[1], x->swizzle[2], x->swizzle[3] },
                            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, vlayers },
                        };
                        if (vkCreateImageView(vk->device, &vci, NULL, &t.view[t.nview]) != VK_SUCCESS)
                            FAIL_AT(30);
                        view = t.view[t.nview++];
                        lay = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                        goto image_done;
                    }
                    lagfx_vk_iosurface_t *ios = lagfx_guest_texture_source(p, task, vk, s->texture_ref[k]);
                    if (!ios) FAIL_AT(14);
                    if (getenv("LAGFX_SRCCHECK") && ios->width >= 1024u)
                        src_check(vk, ios, task, s->texture_ref[k], x->slot);
                    if (ios == target) {
                        ios = lagfx_guest_texture_snapshot(vk, target, (VkImageLayout)*target_layout);
                        if (!ios) FAIL_AT(15);
                    }
                    if (x->view_type != VK_IMAGE_VIEW_TYPE_2D) FAIL_AT(16);
                    bool identity = true;
                    for (int c = 0; c < 4; c++)
                        identity &= x->swizzle[c] == VK_COMPONENT_SWIZZLE_IDENTITY
                                    || x->swizzle[c] == (uint32_t)(VK_COMPONENT_SWIZZLE_R + c);
                    if ((VkFormat)x->view_format == ios->format && identity) {
                        view = ios->view;
                    } else if ((VkFormat)x->view_format == ios->format && t.nview < MAX_TEMP) {
                        VkImageViewCreateInfo vci = {
                            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = ios->image,
                            .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = ios->format,
                            .components = { x->swizzle[0], x->swizzle[1], x->swizzle[2], x->swizzle[3] },
                            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
                        };
                        if (vkCreateImageView(vk->device, &vci, NULL, &t.view[t.nview]) != VK_SUCCESS)
                            FAIL_AT(17);
                        view = t.view[t.nview++];
                    } else {
                        static unsigned n_fmt;
                        if (n_fmt++ < 20u)
                            LAGFX_WARN("M2V: view format %u differs from image format %u",
                                       x->view_format, (unsigned)ios->format);
                        FAIL_AT(18);
                    }
                    lay = ios->layout == VK_IMAGE_LAYOUT_GENERAL ? VK_IMAGE_LAYOUT_GENERAL
                                                                  : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                }
            }
image_done:
            if (x->kind == LAGFX_M2V_W_SAMPLER || x->kind == LAGFX_M2V_W_COMBINED
                || (x->kind == LAGFX_M2V_W_NEUTRAL && x->neutral_kind == 5u)) {
                if (x->kind == LAGFX_M2V_W_NEUTRAL || x->sampler_kind == 2u)
                    smp = neutral_sampler(vk);
                else if (x->sampler_kind == 0u)
                    smp = make_sampler(vk, &x->sampler);
                /* immutable: the layout holds it */
            }
            (void)is_image;
            ii[nwr] = (VkDescriptorImageInfo){ smp, view, lay };
            o->pImageInfo = &ii[nwr];
            break;
        }
        default:
            FAIL_AT(19);
        }
        nwr++;
    }
    if (nwr) vkUpdateDescriptorSets(vk->device, nwr, wr, 0, NULL);

    const uint8_t *spec = NULL;
    size_t spec_len = 0;
    uint32_t n_spec = 0;
    const uint32_t *se = lagfx_m2v_draw_specialization(dr, &spec, &spec_len, &n_spec);
    uint32_t topology = prim <= 4u ? prim : 3u;   /* MTLPrimitiveType == VkPrimitiveTopology here */
    /* the pass's stencil attachment and the bound depth-stencil state */
    stencil_img_t *sti = task->stencil_target_ref
        ? stencil_image(vk, task->id, task->stencil_target_ref, target->width, target->height)
        : NULL;
    lagfx_depth_stencil_t dss;
    bound_depth_stencil(p, task, &dss);
    VkPipeline pl = variant(vk, mp, target->format, topology, spec, spec_len,
                            (const uint32_t (*)[3])se, n_spec,
                            sti ? sti->format : VK_FORMAT_UNDEFINED, &dss);
    if (!pl) FAIL_AT(20);

    /* vertex buffers */
    uint32_t nvbuf = 0;
    const uint64_t *vbs = lagfx_m2v_draw_vertex_buffers(dr, &nvbuf);
    VkBuffer vbuf[16];
    VkDeviceSize voff[16];
    uint32_t vbind[16];
    for (uint32_t i = 0; i < nvbuf && i < 16u; i++) {
        int k = find_buffer(&sb[0], (uint32_t)vbs[i * 3 + 1]);
        if (k < 0) FAIL_AT(21);
        uint64_t off = vbs[i * 3 + 2];
        uint64_t len = sb[0].placement[k].length > off ? sb[0].placement[k].length - off : 16u;
        if (len > (16u << 20)) len = 16u << 20;
        uint8_t *data = malloc((size_t)len);
        if (!data) FAIL_AT(22);
        read_buffer(p, task, &sb[0].placement[k], off, len, data);
        vbuf[i] = temp_buffer(&t, data, len);
        free(data);
        if (!vbuf[i]) FAIL_AT(23);
        voff[i] = 0;
        vbind[i] = (uint32_t)vbs[i * 3];
    }

    /* The exact index slice, preferably from the latest BackingUpdate VA. */
    VkBuffer ibuf = VK_NULL_HANDLE;
    if (ixp) {
        uint32_t len = index_source_length;
        uint8_t *data = malloc(len);
        if (!data) FAIL_AT(31);
        const char *how = NULL;
        uint32_t got = lagfx_read_vtx_source(p, task,
                                             task->pending_draw.index_buffer_ref,
                                             index_source_offset, len, data, &how, 0);
        if (got != len) {
            free(data);
            FAIL_AT(31);
        }
        if (getenv("LAGFX_INDEXLOG")) {
            static unsigned n;
            if (n++ < 160u)
                LAGFX_WARN("INDEXSRC t%u ref=0x%x off=0x%llx len=%u via=%s head=[%u,%u,%u]",
                           task->id, task->pending_draw.index_buffer_ref,
                           (unsigned long long)index_source_offset, len, how ? how : "?",
                           len >= 2u ? (unsigned)lagfx_le16(data) : 0u,
                           len >= 4u ? (unsigned)lagfx_le16(data + 2u) : 0u,
                           len >= 6u ? (unsigned)lagfx_le16(data + 4u) : 0u);
        }
        ibuf = temp_buffer(&t, data, len);
        free(data);
        if (!ibuf) FAIL_AT(32);
    }

    /* device addresses the push constants carry */
    uint32_t nholes = 0;
    const uint32_t *holes = lagfx_m2v_draw_holes(dr, &nholes);
    uint64_t hole_addr[8] = { 0 };
    if (nholes > 8u) FAIL_AT(24);
    for (uint32_t h = 0; h < nholes; h++) {
        if (holes[h * 3] == 2u) {
            hole_addr[h] = status_address(vk);
        } else {
            for (uint32_t r = 0;; r++) {
                uint32_t rst = 0;
                size_t rlen = 0;
                const uint8_t *rt = lagfx_m2v_draw_root_table(dr, r, &rst, &rlen);
                if (!rt) break;
                if (!(rst & holes[h * 3 + 1])) continue;
                uint8_t *copy = malloc(rlen);
                if (!copy) FAIL_AT(25);
                memcpy(copy, rt, rlen);
                uint64_t sink = sink_address(vk);
                uint32_t so = lagfx_m2v_root_sink_offset();
                if (so + 8u <= rlen) memcpy(copy + so, &sink, 8);
                VkBuffer rb = temp_buffer(&t, copy, rlen);
                free(copy);
                if (!rb) FAIL_AT(26);
                hole_addr[h] = buffer_address(vk, rb);
                break;
            }
        }
        if (!hole_addr[h]) FAIL_AT(27);
    }

    /* record */
    VkCommandBuffer cb = VK_NULL_HANDLE;
    if (lagfx_vk_cmdbuf_alloc(vk, &cb) != LAGFX_OK) FAIL_AT(28);
    VkCommandBufferBeginInfo cbi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cb, &cbi);
    VkImageMemoryBarrier bar = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
        .oldLayout = (VkImageLayout)*target_layout,
        .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = target->image,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1, &bar);
    VkRenderingAttachmentInfo att = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = target->view,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    };
    VkRenderingAttachmentInfo satt = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    };
    if (sti) {
        bool combined = sti->format != VK_FORMAT_S8_UINT;
        VkImageMemoryBarrier sb = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                           | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .oldLayout = sti->layout,
            .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = sti->image,
            .subresourceRange = { VK_IMAGE_ASPECT_STENCIL_BIT
                                  | (combined ? VK_IMAGE_ASPECT_DEPTH_BIT : 0u), 0, 1, 0, 1 },
        };
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
                             | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                             0, 0, NULL, 0, NULL, 1, &sb);
        satt.imageView = sti->view;
        /* a pass that loads (or doesn't care about) a stencil we never saw starts at 0 */
        if (task->stencil_clear_pending || sti->layout == VK_IMAGE_LAYOUT_UNDEFINED) {
            satt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            satt.clearValue.depthStencil.stencil =
                task->stencil_clear_pending ? task->stencil_clear_value & 0xffu : 0u;
        }
    }
    VkRenderingInfo ri = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = { { 0, 0 }, { target->width, target->height } },
        .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &att,
        .pStencilAttachment = sti ? &satt : NULL,
    };
    vkCmdBeginRendering(cb, &ri);
    if (sti) {
        vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_BIT, task->stencil_ref_front & 0xffu);
        vkCmdSetStencilReference(cb, VK_STENCIL_FACE_BACK_BIT, task->stencil_ref_back & 0xffu);
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl);
    {
        uint32_t vx = 0, vy = 0, vw = target->width, vh = target->height;
        uint32_t sx = 0, sy = 0, sw = target->width, sh = target->height;
        if (region && region->vp_w && region->vp_x < target->width && region->vp_y < target->height) {
            vx = region->vp_x; vy = region->vp_y; vw = region->vp_w; vh = region->vp_h;
            if (vx + vw > target->width) vw = target->width - vx;
            if (vy + vh > target->height) vh = target->height - vy;
        }
        if (region && region->sc_w && region->sc_x < target->width && region->sc_y < target->height) {
            sx = region->sc_x; sy = region->sc_y; sw = region->sc_w; sh = region->sc_h;
            if (sx + sw > target->width) sw = target->width - sx;
            if (sy + sh > target->height) sh = target->height - sy;
        }
        /* Metal's y-up clip space: the Vulkan viewport with its height negated */
        VkViewport vp = { (float)vx, (float)(vy + vh), (float)vw, -(float)vh, 0.0f, 1.0f };
        VkRect2D sc = { { (int32_t)sx, (int32_t)sy }, { sw, sh } };
        vkCmdSetViewport(cb, 0, 1, &vp);
        vkCmdSetScissor(cb, 0, 1, &sc);
        float bc[4] = { 0, 0, 0, 0 };
        vkCmdSetBlendConstants(cb, bc);
    }
    if (mp->n_sets)
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, mp->layout, 0, mp->n_sets,
                                sets, 0, NULL);
    {
        uint32_t npw = 0;
        const uint8_t *pd = NULL;
        const lagfx_m2v_push_t *pw = lagfx_m2v_draw_push(dr, &npw, &pd);
        for (uint32_t i = 0; i < npw; i++) {
            uint8_t bytes[256];
            uint32_t n = pw[i].size < sizeof bytes ? pw[i].size : (uint32_t)sizeof bytes;
            memcpy(bytes, pd + pw[i].data_offset, n);
            for (uint32_t h = 0; h < nholes; h++) {
                uint32_t at = holes[h * 3 + 2];
                if (at < pw[i].offset || at + 8u > pw[i].offset + n) continue;
                memcpy(bytes + (at - pw[i].offset), &hole_addr[h], 8);
            }
            vkCmdPushConstants(cb, mp->layout, mp->push_stages, pw[i].offset, n, bytes);
        }
    }
    for (uint32_t i = 0; i < nvbuf && i < 16u; i++)
        vkCmdBindVertexBuffers(cb, vbind[i], 1, &vbuf[i], &voff[i]);
    uint64_t ioff = 0;
    const uint32_t *dc = lagfx_m2v_draw_call(dr, &ioff);
    if (dc[0] == 1u) {
        vkCmdBindIndexBuffer(cb, ibuf, ioff, (VkIndexType)dc[6]);
        vkCmdDrawIndexed(cb, dc[1], dc[2], dc[3], (int32_t)dc[4], dc[5]);
    } else {
        vkCmdDraw(cb, dc[1], dc[2], dc[3], dc[5]);
    }
    vkCmdEndRendering(cb);
    /* back to sampleable, as the rest of the host expects */
    VkImageMemoryBarrier post = bar;
    post.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    post.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    post.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    post.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &post);
    vkEndCommandBuffer(cb);

    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                        .pCommandBuffers = &cb };
    if (vkCreateFence(vk->device, &fci, NULL, &fence) == VK_SUCCESS
        && vkQueueSubmit(vk->graphics_queue, 1, &si, fence) == VK_SUCCESS) {
        VkResult wr2 = vkWaitForFences(vk->device, 1, &fence, VK_TRUE, 10000000000ull);
        if (wr2 == VK_SUCCESS) {
            ok = true;
            if (sti) {
                sti->layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                task->stencil_clear_pending = false;
            }
        } else {
            LAGFX_WARN("M2V draw: fence wait failed (%d); waiting for the device", (int)wr2);
            vkDeviceWaitIdle(vk->device);
        }
    }
    if (fence) vkDestroyFence(vk->device, fence, NULL);
    lagfx_vk_cmdbuf_free(vk, cb);
    *target_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
out:
    if (!ok) {
        static unsigned n_fail;
        if (n_fail++ < 100u)
            LAGFX_WARN("M2V draw failed at %s (pipe=0x%x, %u writes)", why,
                       task->pending_pipeline.reference, nw);
    }
    if (pool) vkDestroyDescriptorPool(vk->device, pool, NULL);
    temps_free(&t);
    lagfx_m2v_draw_free(dr);
    return ok;
}

#endif /* LAGFX_HAVE_M2V && LAGFX_HAVE_VULKAN */
