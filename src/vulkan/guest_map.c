/*
 * libapplegfx-vulkan — guest memory imported as Vulkan buffers
 * src/vulkan/guest_map.c
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "guest_map.h"
#include "instance.h"
#include "common/log.h"

#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#ifdef LAGFX_HAVE_VULKAN

#define PAGE 4096u

static uint32_t pick_type(struct lagfx_vk_state *vk, uint32_t bits) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(vk->phys_device, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if (bits & (1u << i)) return i;
    return UINT32_MAX;
}

lagfx_guest_map_t *lagfx_guest_map_create(struct lagfx_vk_state *vk, uint32_t npages,
                                          const uint64_t *gpa, const int *fd,
                                          const uint64_t *offset) {
    if (!vk || !vk->have_host_import || npages == 0u) return NULL;
    size_t len = (size_t)npages * PAGE;
    lagfx_guest_map_t *m = calloc(1, sizeof *m);
    if (!m) return NULL;
    m->gpa = malloc(sizeof(uint64_t) * npages);
    if (!m->gpa) goto fail;
    memcpy(m->gpa, gpa, sizeof(uint64_t) * npages);
    m->npages = npages;

    /* Reserve the range, then map each run of consecutive file pages over it. */
    void *base = mmap(NULL, len, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) goto fail;
    m->base = base;
    for (uint32_t i = 0; i < npages;) {
        uint32_t j = i + 1u;
        while (j < npages && fd[j] == fd[i] && offset[j] == offset[i] + (uint64_t)(j - i) * PAGE)
            j++;
        void *at = mmap(m->base + (size_t)i * PAGE, (size_t)(j - i) * PAGE, PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_FIXED, fd[i], (off_t)offset[i]);
        if (at == MAP_FAILED) goto fail;
        i = j;
    }

    const VkExternalMemoryHandleTypeFlagBits ht =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    VkMemoryHostPointerPropertiesEXT hp = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT,
    };
    if (vk->get_host_pointer_props(vk->device, ht, m->base, &hp) != VK_SUCCESS
        || hp.memoryTypeBits == 0u)
        goto fail;

    VkExternalMemoryBufferCreateInfo ebi = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, .handleTypes = ht,
    };
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &ebi, .size = len,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (vkCreateBuffer(vk->device, &bci, NULL, &m->buffer) != VK_SUCCESS) goto fail;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(vk->device, m->buffer, &req);
    uint32_t type = pick_type(vk, req.memoryTypeBits & hp.memoryTypeBits);
    if (type == UINT32_MAX || req.size > len) goto fail;
    VkImportMemoryHostPointerInfoEXT imp = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
        .handleType = ht, .pHostPointer = m->base,
    };
    VkMemoryAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &imp,
        .allocationSize = len, .memoryTypeIndex = type,
    };
    if (vkAllocateMemory(vk->device, &ai, NULL, &m->memory) != VK_SUCCESS) goto fail;
    if (vkBindBufferMemory(vk->device, m->buffer, m->memory, 0) != VK_SUCCESS) goto fail;
    return m;

fail: {
        static unsigned n_fail;
        if (n_fail++ < 5u)
            LAGFX_WARN("guest_map: can't import %u guest pages; copying instead", npages);
    }
    lagfx_guest_map_destroy(vk, m);
    return NULL;
}

bool lagfx_guest_map_same(const lagfx_guest_map_t *m, uint32_t npages, const uint64_t *gpa) {
    return m && m->npages == npages && memcmp(m->gpa, gpa, sizeof(uint64_t) * npages) == 0;
}

void lagfx_guest_map_destroy(struct lagfx_vk_state *vk, lagfx_guest_map_t *m) {
    if (!m) return;
    if (m->buffer) vkDestroyBuffer(vk->device, m->buffer, NULL);
    if (m->memory) vkFreeMemory(vk->device, m->memory, NULL);
    if (m->base) munmap(m->base, (size_t)m->npages * PAGE);
    free(m->gpa);
    free(m);
}

#endif /* LAGFX_HAVE_VULKAN */
