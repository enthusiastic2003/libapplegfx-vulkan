/*
 * libapplegfx-vulkan — guest memory imported as Vulkan buffers
 * src/vulkan/guest_map.h
 *
 * SPDX-License-Identifier: MIT
 *
 * A texture's guest pages (scattered in guest RAM, which QEMU backs with a
 * shareable memfd) are mapped into one contiguous range of this process and
 * imported with VK_EXT_external_memory_host, so the GPU copies between the
 * texture's image and guest memory directly: no CPU copy, no per-page
 * read_memory/write_memory calls.
 */
#ifndef LIBAPPLEGFX_VULKAN_GUEST_MAP_H
#define LIBAPPLEGFX_VULKAN_GUEST_MAP_H

#include <stdbool.h>
#include <stdint.h>

#ifdef LAGFX_HAVE_VULKAN
#include <vulkan/vulkan.h>

struct lagfx_vk_state;

typedef struct lagfx_guest_map {
    VkBuffer buffer;          /* the pages, in order, as one buffer */
    VkDeviceMemory memory;
    uint8_t *base;            /* where they are mapped in this process */
    uint32_t npages;
    uint64_t *gpa;            /* each page's guest-physical address */
} lagfx_guest_map_t;

/* Map `npages` guest pages (their GPAs, and the file and offset holding
 * each) and import them as a buffer. NULL when the device can't import
 * host memory or a step fails; the caller copies instead. */
lagfx_guest_map_t *lagfx_guest_map_create(struct lagfx_vk_state *vk, uint32_t npages,
                                          const uint64_t *gpa, const int *fd,
                                          const uint64_t *offset);

/* Whether `m` maps exactly these pages. */
bool lagfx_guest_map_same(const lagfx_guest_map_t *m, uint32_t npages, const uint64_t *gpa);

void lagfx_guest_map_destroy(struct lagfx_vk_state *vk, lagfx_guest_map_t *m);

#endif /* LAGFX_HAVE_VULKAN */

#endif /* LIBAPPLEGFX_VULKAN_GUEST_MAP_H */
