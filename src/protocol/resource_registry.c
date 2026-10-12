/*
 * libapplegfx-vulkan — resource reference registry implementation
 * src/protocol/resource_registry.c
 *
 * Copyright © 2026 Matthew Jackson
 * SPDX-License-Identifier: MIT
 */

#include "resource_registry.h"
#include "../common/log.h"

#include <string.h>

static uint32_t index_home(uint32_t ref, uint32_t task_id) {
    uint32_t h = ref * 0x9e3779b1u ^ task_id * 0x85ebca77u;
    h ^= h >> 15;
    return h & (LAGFX_RESOURCE_INDEX_SLOTS - 1u);
}

/* Index slot holding (ref, task), or the empty slot where it would go. */
static uint32_t index_find(const lagfx_resource_registry_t *reg, uint32_t ref, uint32_t task_id) {
    uint32_t s = index_home(ref, task_id);
    for (;;) {
        uint32_t v = reg->index[s];
        if (v == 0u) return s;
        const lagfx_resource_entry_t *e = &reg->entries[v - 1u];
        if (e->ref == ref && e->task_id == task_id) return s;
        s = (s + 1u) & (LAGFX_RESOURCE_INDEX_SLOTS - 1u);
    }
}

static void index_rebuild(lagfx_resource_registry_t *reg) {
    memset(reg->index, 0, sizeof reg->index);
    for (uint32_t i = 0; i < reg->count; ++i) {
        const lagfx_resource_entry_t *e = &reg->entries[i];
        reg->index[index_find(reg, e->ref, e->task_id)] = i + 1u;
    }
}

void lagfx_resource_register(lagfx_resource_registry_t *reg,
                             uint32_t ref,
                             lagfx_resource_type_t type,
                             uint32_t task_id,
                             uint64_t gpu_addr,
                             uint64_t size) {
    if (!reg) {
        return;
    }

    uint32_t slot = index_find(reg, ref, task_id);
    if (reg->index[slot] != 0u) {
        lagfx_resource_entry_t *e = &reg->entries[reg->index[slot] - 1u];
        e->type     = type;
        e->gpu_addr = gpu_addr;
        e->size     = size;
        e->surface_id = 0u;
        LAGFX_TRACE("resource_register: updated ref=0x%x task=%u "
                     "type=%u gpu_addr=0x%llx size=%llu",
                     ref, task_id, (unsigned)type,
                     (unsigned long long)gpu_addr,
                     (unsigned long long)size);
        return;
    }

    if (reg->count >= LAGFX_MAX_RESOURCES) {
        LAGFX_WARN("resource_register: table full (max=%u) ref=0x%x "
                    "task=%u",
                    LAGFX_MAX_RESOURCES, ref, task_id);
        return;
    }

    lagfx_resource_entry_t *e = &reg->entries[reg->count++];
    memset(e, 0, sizeof(*e));
    e->ref         = ref;
    e->type        = type;
    e->task_id     = task_id;
#ifdef LAGFX_HAVE_VULKAN
    e->image       = VK_NULL_HANDLE;
    e->view        = VK_NULL_HANDLE;
#endif
    e->gpu_addr    = gpu_addr;
    e->size        = size;
    reg->index[slot] = reg->count;

    LAGFX_TRACE("resource_register: inserted ref=0x%x task=%u type=%u "
                  "gpu_addr=0x%llx size=%llu (total=%u)",
                  ref, task_id, (unsigned)type,
                  (unsigned long long)gpu_addr,
                  (unsigned long long)size,
                  reg->count);
}

lagfx_resource_entry_t *lagfx_resource_lookup(lagfx_resource_registry_t *reg,
                                               uint32_t ref,
                                               uint32_t task_id) {
    if (!reg) {
        return NULL;
    }
    uint32_t v = reg->index[index_find(reg, ref, task_id)];
    return v ? &reg->entries[v - 1u] : NULL;
}

/* M1 B9 — task-agnostic texture lookup. IOSurfaces are global, shared by
 * surface_id across the system, and are registered under a hardcoded task_id=0
 * (the single-task convention) while draw-time texture binding looks up under
 * the real task_id — so an exact (ref,task) lookup always misses. Textures are
 * keyed by surface_id alone; match on ref + TEXTURE type, ignoring task_id.
 * Type-filtered so a same-numbered buffer ref in another task can't collide. */
lagfx_resource_entry_t *lagfx_resource_lookup_texture(lagfx_resource_registry_t *reg,
                                                       uint32_t ref) {
    if (!reg) {
        return NULL;
    }
    for (uint32_t i = 0; i < reg->count; ++i) {
        if (reg->entries[i].ref == ref
            && reg->entries[i].type == LAGFX_RESOURCE_TYPE_TEXTURE) {
            return &reg->entries[i];
        }
    }
    return NULL;
}

void lagfx_resource_unregister(lagfx_resource_registry_t *reg,
                               uint32_t ref,
                               uint32_t task_id) {
    if (!reg) {
        return;
    }

    for (uint32_t i = 0; i < reg->count; ++i) {
        if (reg->entries[i].ref == ref
            && reg->entries[i].task_id == task_id) {
            if (i + 1 < reg->count) {
                reg->entries[i] = reg->entries[reg->count - 1];
            }
            memset(&reg->entries[reg->count - 1], 0,
                   sizeof(lagfx_resource_entry_t));
            reg->count--;
            index_rebuild(reg);
            LAGFX_TRACE("resource_unregister: ref=0x%x task=%u "
                         "(remaining=%u)",
                         ref, task_id, reg->count);
            return;
        }
    }

    LAGFX_TRACE("resource_unregister: ref=0x%x task=%u not found",
                 ref, task_id);
}

void lagfx_resource_clear_task(lagfx_resource_registry_t *reg,
                               uint32_t task_id) {
    if (!reg) {
        return;
    }

    uint32_t write_idx = 0;
    for (uint32_t read_idx = 0; read_idx < reg->count; ++read_idx) {
        if (reg->entries[read_idx].task_id != task_id) {
            if (write_idx != read_idx) {
                reg->entries[write_idx] = reg->entries[read_idx];
            }
            write_idx++;
        }
    }

    uint32_t removed = reg->count - write_idx;
    if (removed > 0) {
        memset(&reg->entries[write_idx], 0,
               removed * sizeof(lagfx_resource_entry_t));
    }
    reg->count = write_idx;
    if (removed > 0) index_rebuild(reg);

    LAGFX_TRACE("resource_clear_task: task=%u removed=%u remaining=%u",
                 task_id, removed, reg->count);
}
