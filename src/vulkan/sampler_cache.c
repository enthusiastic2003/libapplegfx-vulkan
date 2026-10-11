/*
 * libapplegfx-vulkan — VkSampler cache for decoded guest samplers
 * src/vulkan/sampler_cache.c
 *
 * SPDX-License-Identifier: MIT
 *
 * See sampler_cache.h. Guests use a handful of distinct samplers, so a small
 * linear cache is enough; it is never evicted.
 */

#include "sampler_cache.h"

#include "common/log.h"
#include "instance.h"

#include <string.h>

#define CACHE_SIZE 64u

static struct {
    VkDevice device;
    lagfx_sampler_desc_t desc;
    VkSampler sampler;
} cache[CACHE_SIZE];
static uint32_t cache_n;

static VkSamplerAddressMode address_mode(uint8_t a) {
    switch (a) {
    case LAGFX_SAMPLER_REPEAT:               return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case LAGFX_SAMPLER_MIRROR_REPEAT:        return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case LAGFX_SAMPLER_MIRROR_CLAMP_TO_EDGE: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
    case LAGFX_SAMPLER_CLAMP_TO_ZERO:
    case LAGFX_SAMPLER_CLAMP_TO_BORDER:      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    default:                                 return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    }
}

static VkSampler create(VkDevice device, const lagfx_sampler_desc_t *d) {
    VkSamplerCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = d->mag_filter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        .minFilter = d->min_filter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        .mipmapMode = d->mip_filter == 2u ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                          : VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = address_mode(d->address[0]),
        .addressModeV = address_mode(d->address[1]),
        .addressModeW = address_mode(d->address[2]),
        .minLod = d->lod_min,
        .maxLod = d->lod_max > 1000.0f ? VK_LOD_CLAMP_NONE : d->lod_max,
        .borderColor = d->border_color == 2u ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE
                     : d->border_color == 1u ? VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK
                                             : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = d->normalized ? VK_FALSE : VK_TRUE,
        /* anisotropy needs the samplerAnisotropy feature; not enabled */
    };
    for (int i = 0; i < 3; i++)
        if (d->address[i] == LAGFX_SAMPLER_CLAMP_TO_ZERO)
            ci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    if (d->mip_filter == 0u) {
        /* Not mipmapped: stay on level 0, but let the LOD go above 0 so the
         * min filter still applies when minifying. */
        ci.minLod = 0.0f;
        ci.maxLod = 0.25f;
    }
    if (ci.unnormalizedCoordinates) {
        /* Vulkan's limits for unnormalized coordinates */
        ci.minFilter = ci.magFilter;
        ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        ci.minLod = ci.maxLod = 0.0f;
        VkSamplerAddressMode *m[3] = { &ci.addressModeU, &ci.addressModeV, &ci.addressModeW };
        for (int i = 0; i < 3; i++)
            if (*m[i] != VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER)
                *m[i] = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    }
    VkSampler s = VK_NULL_HANDLE;
    if (vkCreateSampler(device, &ci, NULL, &s) != VK_SUCCESS) return VK_NULL_HANDLE;
    return s;
}

VkSampler lagfx_vk_sampler_get(struct lagfx_vk_state *vk, const lagfx_sampler_desc_t *d) {
    if (!vk || !d || vk->device == VK_NULL_HANDLE) return VK_NULL_HANDLE;
    for (uint32_t i = 0; i < cache_n; i++)
        if (cache[i].device == vk->device && memcmp(&cache[i].desc, d, sizeof *d) == 0)
            return cache[i].sampler;
    VkSampler s = create(vk->device, d);
    if (s == VK_NULL_HANDLE) {
        LAGFX_WARN("sampler_cache: vkCreateSampler failed");
        return VK_NULL_HANDLE;
    }
    if (cache_n < CACHE_SIZE) {
        cache[cache_n].device = vk->device;
        cache[cache_n].desc = *d;
        cache[cache_n].sampler = s;
        cache_n++;
    } else {
        LAGFX_WARN("sampler_cache: full; sampler not cached (leaks)");
    }
    LAGFX_LOG("sampler_cache: new sampler min=%u mag=%u mip=%u addr=%u/%u/%u border=%u norm=%d "
              "lod=%g..%g", d->min_filter, d->mag_filter, d->mip_filter, d->address[0],
              d->address[1], d->address[2], d->border_color, d->normalized, d->lod_min, d->lod_max);
    return s;
}
