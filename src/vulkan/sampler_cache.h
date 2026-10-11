/*
 * libapplegfx-vulkan — VkSampler cache for decoded guest samplers
 * src/vulkan/sampler_cache.h
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef LIBAPPLEGFX_VULKAN_SAMPLER_CACHE_H
#define LIBAPPLEGFX_VULKAN_SAMPLER_CACHE_H

#include "protocol/sampler_state.h"

#include <vulkan/vulkan.h>

struct lagfx_vk_state;

/* A VkSampler matching `d`, created on first use and kept for the device's
 * lifetime. VK_NULL_HANDLE on failure. */
VkSampler lagfx_vk_sampler_get(struct lagfx_vk_state *vk, const lagfx_sampler_desc_t *d);

#endif /* LIBAPPLEGFX_VULKAN_SAMPLER_CACHE_H */
