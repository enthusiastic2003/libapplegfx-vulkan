/*
 * libapplegfx-vulkan — resource registry (unit)
 * tests/resource-registry-unit.c
 *
 * SPDX-License-Identifier: MIT
 *
 * The real registry (src/protocol/resource_registry.c): more live entries than
 * the old 256-entry table held, lookups through the index after removals move
 * entries, and per-task clearing.
 */

#include "protocol/resource_registry.h"

#include <stdio.h>
#include <stdlib.h>

static int failures;

#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } \
} while (0)

static uint64_t addr(uint32_t ref, uint32_t task) { return ((uint64_t)task << 32) | ref; }

/* every (ref, task) registered so far is found with its own address */
static int all_found(lagfx_resource_registry_t *reg, uint32_t nref, uint32_t ntask,
                     uint32_t skip_task) {
    int bad = 0;
    for (uint32_t t = 0; t < ntask; t++)
        for (uint32_t r = 1; r <= nref; r++) {
            lagfx_resource_entry_t *e = lagfx_resource_lookup(reg, r, t);
            if (t == skip_task) { bad += e != NULL; continue; }
            bad += !e || e->gpu_addr != addr(r, t);
        }
    return bad;
}

int main(void) {
    lagfx_resource_registry_t *reg = calloc(1, sizeof *reg);
    if (!reg) return 1;

    /* 3000 refs in each of 4 tasks: far past the old 256-entry limit */
    const uint32_t nref = 3000, ntask = 4;
    for (uint32_t t = 0; t < ntask; t++)
        for (uint32_t r = 1; r <= nref; r++)
            lagfx_resource_register(reg, r, LAGFX_RESOURCE_TYPE_TEXTURE, t, addr(r, t), 64);
    CHECK(reg->count == nref * ntask);
    CHECK(all_found(reg, nref, ntask, ~0u) == 0);
    CHECK(lagfx_resource_lookup(reg, nref + 1, 0) == NULL);
    CHECK(lagfx_resource_lookup(reg, 1, ntask) == NULL);

    /* re-registering updates in place */
    lagfx_resource_register(reg, 7, LAGFX_RESOURCE_TYPE_BUFFER, 2, 0x1234, 8);
    lagfx_resource_entry_t *e = lagfx_resource_lookup(reg, 7, 2);
    CHECK(reg->count == nref * ntask);
    CHECK(e && e->type == LAGFX_RESOURCE_TYPE_BUFFER && e->gpu_addr == 0x1234);
    lagfx_resource_register(reg, 7, LAGFX_RESOURCE_TYPE_TEXTURE, 2, addr(7, 2), 64);

    /* removal moves the last entry into the hole; both stay reachable */
    lagfx_resource_unregister(reg, 5, 1);
    CHECK(lagfx_resource_lookup(reg, 5, 1) == NULL);
    CHECK(lagfx_resource_lookup(reg, nref, ntask - 1) != NULL);
    lagfx_resource_register(reg, 5, LAGFX_RESOURCE_TYPE_TEXTURE, 1, addr(5, 1), 64);
    CHECK(all_found(reg, nref, ntask, ~0u) == 0);

    /* clearing one task leaves the others intact */
    lagfx_resource_clear_task(reg, 2);
    CHECK(reg->count == nref * (ntask - 1));
    CHECK(all_found(reg, nref, ntask, 2) == 0);

    /* the table still refuses past its capacity */
    for (uint32_t r = 1; reg->count < LAGFX_MAX_RESOURCES; r++)
        lagfx_resource_register(reg, r, LAGFX_RESOURCE_TYPE_BUFFER, 100, 0, 0);
    lagfx_resource_register(reg, 1, LAGFX_RESOURCE_TYPE_BUFFER, 101, 0, 0);
    CHECK(reg->count == LAGFX_MAX_RESOURCES);
    CHECK(lagfx_resource_lookup(reg, 1, 101) == NULL);
    CHECK(all_found(reg, nref, ntask, 2) == 0);

    free(reg);
    if (failures) return 1;
    printf("resource registry: OK\n");
    return 0;
}
