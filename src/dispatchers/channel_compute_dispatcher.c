/*
 * libapplegfx-vulkan — Compute channel dispatcher (ch 1-4)
 * src/dispatchers/channel_compute_dispatcher.c
 *
 * Copyright © 2026 Matthew Jackson
 * SPDX-License-Identifier: MIT
 *
 * Per-channel ring drain for compute vchans. Each child channel
 * (chan_id 1..4) has its own command ring registered via
 * FIFORingDescriptor at:
 *
 *   shared_control + 0x400 + 20 * (chan_id - 1)
 *
 * where shared_control is the page registered at BAR0+0x101c
 * (ring_shared_page_pfn). See PROTOCOL.md §3, §5 and
 * paravirt-re/library/state-machines/FIFORingDescriptor.md (the
 * authoritative reference verified 2026-04-25 against a live xp dump).
 *
 * Wire format and read semantics on each ring mirror the root-channel
 * drainer (channel_0_dispatcher.c). The descriptor's read_ptr is
 * authoritative — host advances it after draining.
 */

#include "cmd_dump.h"
#include "channel_compute_dispatcher.h"
#include "ring_common.h"
#include "../doorbell.h"
#include "../device.h"
#include "../common/le.h"
#include "../common/log.h"
#include "protocol/opcodes.h"
#include "protocol/state.h"
#include "protocol/object_resolver.h"
#include "handlers/handlers.h"
#include "handlers/iosurface/iosurface.h"
#include "handlers/compute/task_translate.h"

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

/* LAGFX_DUMP_CHEV=1: hex-dump channel events (0x34..0x36) and object lists
 * (0x25) at warn level. */
static void dump_chev(const lagfx_protocol_t *p, const lagfx_cmd_header_t *hdr, const char *what) {
    if (!getenv("LAGFX_DUMP_CHEV") || !hdr->payload) return;
    char hx[3 * 48 + 1];
    size_t hn = 0;
    for (uint32_t k = 0; k < hdr->payload_size && k < 48u; k++)
        hn += (size_t)snprintf(hx + hn, sizeof hx - hn, "%02x ", hdr->payload[k]);
    hx[hn] = 0;
    LAGFX_WARN("CHEV %s op=0x%02x ch=%u stamp=0x%08x len=%u: %s", what, (unsigned)hdr->opcode,
               (unsigned)p->current_chan_id, hdr->stamp, (unsigned)hdr->payload_size, hx);
}

/* Resolve only the object named by 0x25 and dump its declared bytes.  This is
 * a bounded object-table lookup, not a guest-memory scan. */
static void dump_registered_pipeline(lagfx_protocol_t *p, lagfx_task_entry_t *task,
                                     uint32_t object_id, const lagfx_cmd_header_t *hdr) {
    if (!getenv("LAGFX_PIPE_CREATE_TRACE") || !task) return;

    uint8_t type = 0;
    uint64_t va = 0, gpa = 0;
    if (!lagfx_resolve_object_data(p, task, object_id, &type, &va, &gpa) || !va) {
        LAGFX_WARN("PSOREG unresolved t%u obj=0x%x stamp=0x%08x ch=%u",
                   task->id, object_id, hdr->stamp, (unsigned)p->current_chan_id);
        return;
    }
    LAGFX_WARN("PSOREG slot t%u obj=0x%x type=0x%02x va=0x%llx gpa=0x%llx "
               "stamp=0x%08x ch=%u",
               task->id, object_id, type, (unsigned long long)va,
               (unsigned long long)gpa, hdr->stamp, (unsigned)p->current_chan_id);
    if (type != LAGFX_APV_TYPE_PIPELINE) return;

    uint8_t head[16] = {0};
    if (!lagfx_task_read_virtual(p, task, va, sizeof head, head)) {
        LAGFX_WARN("PSOREG unreadable t%u obj=0x%x va=0x%llx",
                   task->id, object_id, (unsigned long long)va);
        return;
    }
    uint32_t declared = lagfx_le32(head + 4);
    if (declared < 16u || declared > 512u) declared = 16u;
    uint8_t blob[512] = {0};
    if (!lagfx_task_read_virtual(p, task, va, declared, blob)) return;

    p->diag.pso_reg_serial++;
    p->diag.pso_reg_task = task->id;
    p->diag.pso_reg_object = object_id;
    p->diag.pso_reg_stamp = hdr->stamp;
    p->diag.pso_reg_channel = p->current_chan_id;
    LAGFX_WARN("PSOREG #%u t%u obj=0x%x slotType=0x%02x objKind=0x%x declared=%u "
               "va=0x%llx gpa=0x%llx stamp=0x%08x ch=%u",
               p->diag.pso_reg_serial, task->id, object_id, type,
               lagfx_le32(blob), declared, (unsigned long long)va,
               (unsigned long long)gpa, hdr->stamp, (unsigned)p->current_chan_id);
    for (uint32_t off = 0; off < declared; off += 32u) {
        uint32_t row = declared - off < 32u ? declared - off : 32u;
        char hx[3 * 32 + 1]; size_t hn = 0;
        for (uint32_t k = 0; k < row; k++)
            hn += (size_t)snprintf(hx + hn, sizeof hx - hn, "%02x ", blob[off + k]);
        hx[hn] = 0;
        LAGFX_WARN("PSOREG #%u +%03x: %s", p->diag.pso_reg_serial, off, hx);
    }
}

/* Phase 6b: 0x25 CmdSetObjectAndPlacementList payload is
 * `{taskId: u32, objectId: u32}`
 * (see paravirt-re/library/apv-object-entry-parser-scoping-2026-05-17.md).
 * Insert the objectId into the per-task active-objects registry so
 * op_0x74 SetRenderPipelineState can query it before falling back to
 * the heap-VA lookup. Idempotent — duplicate objectIds are a no-op. */
static void
register_active_object(lagfx_task_entry_t *task, uint32_t object_id) {
    if (!task) return;
    /* Linear scan for dedup. The bounded set (64 entries) is fine for
     * Apple's observed traffic (~10s of unique objectIds per task). */
    for (uint32_t i = 0; i < task->active_objects.count; i++) {
        if (task->active_objects.object_ids[i] == object_id) return;
    }
    uint32_t capacity = (uint32_t)(sizeof task->active_objects.object_ids
                                   / sizeof task->active_objects.object_ids[0]);
    if (task->active_objects.count >= capacity) {
        LAGFX_WARN("register_active_object: task=%u registry full (%u); dropping objectId=0x%x",
                   task->id, capacity, object_id);
        return;
    }
    task->active_objects.object_ids[task->active_objects.count++] = object_id;
}


#define LAGFX_COMPUTE_DRAIN_MAX_CMDS 128u
#define LAGFX_COMPUTE_MAX_CMD_BYTES  4096u

/* VirtualChannel ring size per FIFORingDescriptor.md (§"ring size"):
 * 0x10000 (64 KiB). See ring_common.h for the descriptor read +
 * PFN-array resolve helpers.
 */
#define LAGFX_COMPUTE_RING_SIZE  0x10000u

/* CmdGetComputeInfo (child opcode 0x3b) writes keyed u32 pairs into a
 * task-virtual reply buffer.  The request's count is capacity, not the number
 * of keys the host must invent; terminate the three known Tahoe keys with a
 * zero pair when there is room. */
static bool
write_task_virtual(lagfx_protocol_t *p, const lagfx_task_entry_t *task,
                   uint64_t va, const uint8_t *src, uint32_t len) {
    if (!p || !task || !src || !p->dev) return false;
    lagfx_device_t *dev = (lagfx_device_t *)p->dev;
    if (!dev->desc.shell.write_memory) return false;

    uint32_t done = 0;
    while (done < len) {
        uint64_t cur = va + done;
        uint32_t chunk = 0x1000u - (uint32_t)(cur & 0xfffu);
        if (chunk > len - done) chunk = len - done;
        uint64_t gpa = 0;
        if (!lagfx_task_translate(p, task, cur, &gpa)
            || !dev->desc.shell.write_memory(dev->desc.shell.opaque, gpa,
                                              chunk, src + done))
            return false;
        done += chunk;
    }
    return true;
}

static void
reply_compute_info(lagfx_protocol_t *p, const lagfx_cmd_header_t *hdr) {
    if (!hdr->payload || hdr->payload_size < 24u) {
        LAGFX_WARN("compute: CmdGetComputeInfo short payload=%u",
                   (unsigned)hdr->payload_size);
        return;
    }

    const uint32_t task_id = lagfx_le32(hdr->payload + 0);
    const uint32_t pipeline_ref = lagfx_le32(hdr->payload + 4);
    const uint32_t key_table_len = lagfx_le32(hdr->payload + 8);
    const uint32_t capacity = lagfx_le32(hdr->payload + 12);
    const uint64_t reply_va = lagfx_le64(hdr->payload + 16);
    lagfx_task_entry_t *task = lagfx_protocol_find_task(p, task_id);
    if (!task || reply_va == 0u || capacity == 0u) {
        LAGFX_WARN("compute: CmdGetComputeInfo invalid task=%u pipe=0x%x "
                   "keys=%u capacity=%u reply=0x%llx",
                   task_id, pipeline_ref, key_table_len, capacity,
                   (unsigned long long)reply_va);
        return;
    }

    /* Metal pipeline properties.  Width 32 is the virtual Apple SIMD ABI.
     * 512 is conservative until per-kernel reflection supplies an exact
     * maximum; zero static memory is valid for the currently observed MPS
     * kernels and is preferable to leaving the property undefined. */
    static const struct { uint32_t key, value; } caps[] = {
        { 1u, 512u }, /* maxTotalThreadsPerThreadgroup */
        { 3u,  32u }, /* threadExecutionWidth */
        { 4u,   0u }, /* staticThreadgroupMemoryLength */
    };
    uint8_t reply[32] = {0}; /* three pairs plus terminating zero pair */
    uint32_t wrote = 0;
    for (uint32_t i = 0; i < sizeof caps / sizeof caps[0]; i++) {
        if (caps[i].key >= key_table_len || wrote >= capacity) continue;
        lagfx_put_le32(reply + wrote * 8u, caps[i].key);
        lagfx_put_le32(reply + wrote * 8u + 4u, caps[i].value);
        wrote++;
    }
    uint32_t pairs = wrote + (wrote < capacity ? 1u : 0u);
    if (!write_task_virtual(p, task, reply_va, reply, pairs * 8u)) {
        LAGFX_WARN("compute: CmdGetComputeInfo write failed task=%u "
                   "pipe=0x%x reply=0x%llx", task_id, pipeline_ref,
                   (unsigned long long)reply_va);
        return;
    }
    LAGFX_LOG("compute: CmdGetComputeInfo task=%u pipe=0x%x keys=%u "
              "capacity=%u wrote=%u reply=0x%llx max=512 width=32",
              task_id, pipeline_ref, key_table_len, capacity, wrote,
              (unsigned long long)reply_va);
}

/* Dispatch a single command to appropriate handler. */
static void dispatch_command(lagfx_protocol_t *p, const lagfx_cmd_header_t *hdr) {
    if (!p || !hdr) {
        return;
    }

    LAGFX_TRACE("compute dispatch: opcode=0x%04x stamp=0x%08x", hdr->opcode, hdr->stamp);

    if (getenv("LAGFX_DUMP_ALLCMD") && hdr->opcode != 0x37u) {
        char hx[3 * 32 + 1];
        size_t hn = 0;
        for (uint32_t k = 0; hdr->payload && k < hdr->payload_size && k < 32u; k++)
            hn += (size_t)snprintf(hx + hn, sizeof hx - hn, "%02x ", hdr->payload[k]);
        hx[hn] = 0;
        LAGFX_WARN("ALLCMD compute op=0x%02x ch=%u len=%u: %s", (unsigned)hdr->opcode,
                   (unsigned)p->current_chan_id, (unsigned)hdr->payload_size, hx);
    }
    switch (hdr->opcode) {
        /* CmdExecIndirect2 on a compute channel — the inner stream is
         * where Metal commands live. See render-decoder-handlers.md. */
        case LAGFX_OP_EXEC_INDIRECT2:  /* 0x20 */
            lagfx_compute_exec_cmdbuf(p, hdr);
            break;

        /* Kext-side per-channel exec variant per
         * M4-inner-opcode-implementation-guide.md §1.1 — same outer
         * payload as 0x20, kext emits this on the per-channel exec
         * rings (ch 1..4). Conflicts with LAGFX_OP_CHANNEL_EVENT_37
         * naming in opcodes.h — the kext-disasm pass classified 0x37
         * as "ChannelEventMachine-adjacent" before the M4 RE pass
         * identified it as the per-channel CmdExecIndirect2. The
         * payload's outer layout (12 + dc*24 + rc*16) distinguishes
         * the two at runtime; for now route 0x37 to exec_cmdbuf
         * which handles both empty (event-only) and populated (real
         * exec) payloads gracefully. */
        case LAGFX_OP_CHANNEL_EVENT_37:  /* 0x37 — see comment above */
            lagfx_compute_exec_cmdbuf(p, hdr);
            break;

        /* Channel-event / immediate-vchan opcodes fired by the kext
         * on the Immediate (ch=2) and Uploads/Downloads (ch=3/4)
         * channels. Log + ack — these don't carry render workloads
         * (no inner-PGCmdHeader stream); they're kext-internal lifecycle
         * events. Documented in paravirt-re/library/PROTOCOL.md §13
         * + journey/opcodes-0x35-0x36-0x39.md. */
        case LAGFX_OP_UNMAP_MEMORY_IMMEDIATE: /* 0x22 — CmdUnmapMemoryImmediate */
            LAGFX_LOG("compute: 0x22 CmdUnmapMemoryImmediate ch=%u stamp=0x%08x "
                      "payload_size=%u",
                      (unsigned)p->current_chan_id, hdr->stamp,
                      (unsigned)hdr->payload_size);
            break;
        case LAGFX_OP_SET_OBJECT_LIST:        /* 0x24 CmdSetObjectList */
            /* Stub. Payload per command-buffer-format.md:254 is
             * `objectArray[], count`. Phase 6b will parse the
             * objectArray (APVObjectEntry records) to register
             * pipeline-state metallib bytes. Today: log+ack discards
             * the payload. */
            LAGFX_LOG("compute: 0x24 CmdSetObjectList ch=%u stamp=0x%08x payload_size=%u",
                      (unsigned)p->current_chan_id, hdr->stamp,
                      (unsigned)hdr->payload_size);
            break;
        case LAGFX_OP_SET_OBJECT_PLACEMENT:   /* 0x25 CmdSetObjectAndPlacementList */
            /* Phase 6b — wire data captured 2026-05-28 confirmed
             * payload = `{taskId: u32, objectId: u32}` (NOT the dual-
             * array TLV ENTRY-007 anticipated). Decode the objectId
             * and register it in the per-task active-objects set so
             * op_0x74 can query active membership before doing the
             * heap-VA lookup.
             *
             * Mislabeled enum kept for backwards-compatibility —
             * 0x25 carries object+placement together, not just
             * placement (command-buffer-format.md:255). */
            LAGFX_LOG("compute: 0x25 CmdSetObjectAndPlacementList ch=%u stamp=0x%08x payload_size=%u",
                      (unsigned)p->current_chan_id, hdr->stamp,
                      (unsigned)hdr->payload_size);
            dump_chev(p, hdr, "objects");
            if (hdr->payload && hdr->payload_size >= 8u) {
                uint32_t task_id   = lagfx_le32(hdr->payload + 0);
                uint32_t object_id = lagfx_le32(hdr->payload + 4);
                lagfx_task_entry_t *task = lagfx_protocol_find_task(p, task_id);
                if (task) {
                    register_active_object(task, object_id);
                    dump_registered_pipeline(p, task, object_id, hdr);
                    LAGFX_LOG("compute: 0x25 registered objectId=0x%x in task=%u (active=%u)",
                              object_id, task->id,
                              task->active_objects.count);
                } else {
                    LAGFX_WARN("compute: 0x25 no live task=%u for objectId=0x%x",
                               task_id, object_id);
                }
            }
            break;
        case LAGFX_OP_IOSURFACE_LOOKUP:       /* 0x28 */
            /* Live evidence shows 0x28 on compute channels too — route
             * to the real lookup handler so resource_registry stays
             * in sync with the cross-task IOSurface lifecycle. */
            lagfx_iosurface_lookup(p, hdr);
            break;
        case LAGFX_OP_CHANNEL_EVENT_34:       /* 0x34 */
        case LAGFX_OP_CHANNEL_EVENT_35:       /* 0x35 */
        case LAGFX_OP_CHANNEL_EVENT_36:       /* 0x36 */
            /* ChannelEventMachine-adjacent (kext-internal scheduler
             * pings). Log + ack. 0x37 is handled above as the
             * per-channel CmdExecIndirect2. */
            LAGFX_LOG("compute: channel_event 0x%02x ch=%u stamp=0x%08x payload_size=%u",
                      (unsigned)hdr->opcode, (unsigned)p->current_chan_id,
                      hdr->stamp, (unsigned)hdr->payload_size);
            dump_chev(p, hdr, "event");
            if (getenv("LAGFX_PIPE_CREATE_TRACE") && p->diag.pso_reg_serial) {
                LAGFX_WARN("PSOEV after=#%u reg=t%u/obj=0x%x/regstamp=0x%08x/regch=%u "
                           "event=0x%02x stamp=0x%08x ch=%u",
                           p->diag.pso_reg_serial, p->diag.pso_reg_task,
                           p->diag.pso_reg_object, p->diag.pso_reg_stamp,
                           (unsigned)p->diag.pso_reg_channel, (unsigned)hdr->opcode,
                           hdr->stamp, (unsigned)p->current_chan_id);
            }
            break;
        case LAGFX_OP_MAP_MEMORY_IMMEDIATE:   /* 0x39 — CmdMapMemoryImmediate */
            /* 20 B: {flag u32, gpa=PFN<<12 u64, size u64}. These establish the
             * real object->GPA mappings; decode them to correlate a ~5 MB region
             * to the wallpaper surface 0x10 (its placement descriptor reads black,
             * so the true backing is mapped here). */
            if (hdr->payload && hdr->payload_size >= 20u) {
                uint32_t flag = lagfx_le32(hdr->payload + 0);
                uint64_t gpa  = lagfx_le64(hdr->payload + 4);
                uint64_t sz   = lagfx_le64(hdr->payload + 12);
                LAGFX_LOG("MAPMEM 0x39 stamp=0x%08x flag=0x%x gpa=0x%llx size=%llu%s",
                          hdr->stamp, flag, (unsigned long long)gpa,
                          (unsigned long long)sz,
                          sz >= 4u*1024u*1024u ? " <== BIG" : "");
                if (sz >= 4u*1024u*1024u && gpa != 0u) {
                    bool known = false;
                    for (uint32_t bm = 0; bm < p->big_maps_n && bm < 16u; bm++)
                        if (p->big_maps[bm].gpa == gpa) { known = true; break; }
                    if (!known && p->big_maps_n < 16u) {
                        p->big_maps[p->big_maps_n].gpa = gpa;
                        p->big_maps[p->big_maps_n].size = sz;
                        p->big_maps_n++;
                    }
                }
            } else {
                LAGFX_LOG("compute: 0x39 CmdMapMemoryImmediate ch=%u stamp=0x%08x payload_size=%u",
                          (unsigned)p->current_chan_id, hdr->stamp,
                          (unsigned)hdr->payload_size);
            }
            break;

        case LAGFX_OP_VCHAN_REPLACE_PHYSICAL: { /* 0x3c — per-resource replacePhysical on Immediate vchan */
            /* RE: paravirt-re — single kext emit-site at vaddr 0x14564494
             * in AppleParavirtResource::replacePhysical() (vaddr 0x145643f2,
             * mangled __ZN21AppleParavirtResource15replacePhysicalEv).
             * See paravirt-re/classes/AppleParavirtResource.md and the
             * caller annotated/AppleParavirtMemoryMap-commitIntoGPUPageTable.annotated.asm
             * (loop body at 0x1455dc4e calls this for each resource in
             * task[+0xa0] when [chan+0x114] re-binding flag is set).
             *
             * Wire trailer: 8 bytes `{u32 eventID/taskID, u32 counter}` —
             * same emit-helper family as 0x25/0x36 (single
             * apvgpu_cmd_builder_reserve(8) after channel lock, then
             * `[r12]=eventID; [r12+4]=[chan+0x180]=counter`). Total
             * command size = 12 (header) + 8 (trailer) = 20 bytes;
             * matches the live ring delta `(wp - rp) == 0x14` observed
             * in /tmp/lagfx.log compute-drain traces.
             *
             * Semantics: kext notifies host that a per-task Resource has
             * had its host backing pages relocated and any host-side
             * mapping caches should be invalidated. The downstream side-
             * table publishes (two `vt[+0x1b8]` calls at kext 0x14564598 /
             * 0x145645b6) update `[accel_shared+0x380]`'s per-task tracker
             * with the new task[+0x20] / task[+0x270] pointers — these
             * are kext-internal cleanup, not wire data we need to
             * consume.
             *
             * Log+ack correct because: until lagfx allocates real
             * VkImage/VkBuffer per-resource (Stage 30 work; the
             * pre-refactor real-translation paths in dead-code-to-revive/
             * have been ported as parse-and-trace stubs per
             * project_m5_inflight.md), there is no host-side mapping
             * cache to invalidate. The kext does NOT wait on a
             * 0x3c-specific stamp completion — `stamp` is the standard
             * per-channel monotonic that the universal complete_stamp
             * path already raises after dispatch.
             *
             * TODO (Stage 30): when per-resource Vulkan backing lands,
             * look up the resource by `eventID` (== task->id via
             * 0x14563376) and flush any host-side mapping cache. The
             * `counter` field is the per-channel emit counter and is
             * not load-bearing for host behaviour. */
            uint32_t eventID = 0, counter = 0;
            if (hdr->payload && hdr->payload_size >= 8) {
                eventID = lagfx_le32(hdr->payload + 0);
                counter = lagfx_le32(hdr->payload + 4);
            }
            LAGFX_LOG("compute: 0x3c VchanReplacePhysical ch=%u stamp=0x%08x "
                      "eventID=%u counter=%u payload_size=%u",
                      (unsigned)p->current_chan_id, hdr->stamp,
                      eventID, counter, (unsigned)hdr->payload_size);
            break;
        }

        case 0x3b: {
            /* Child CmdGetComputeInfo:
             * {u32 task, u32 pipeline_ref, u32 key_table_len, u32 capacity,
             *  u64 reply_task_va}.  The old "BackingUpdate" interpretation
             * happened to label the characteristic 5/6 words as constants
             * and therefore never populated the pipeline-state properties. */
            reply_compute_info(p, hdr);
            break;
        }

        default:
            LAGFX_WARN("compute dispatch: ch=%u unknown opcode 0x%04x stamp=0x%08x",
                       (unsigned)p->current_chan_id, hdr->opcode, hdr->stamp);
            /* M2c wire RE (LAGFX_DUMP_SPV): raw payload hex for unknown opcodes.
             * 0x3b flows now that the ring-wrap drop is fixed and is a suspect
             * for the slab-content upload that would fill the composite vertex
             * buffers we read as text/poison. */
            if (getenv("LAGFX_DUMP_SPV") && hdr->payload && hdr->payload_size) {
                char uh[200]; size_t un = 0;
                uint32_t cap = hdr->payload_size < 64u ? hdr->payload_size : 64u;
                for (uint32_t k = 0; k < cap && un + 3 < sizeof(uh); k++)
                    un += (size_t)snprintf(uh + un, sizeof(uh) - un, "%02x ",
                                           hdr->payload[k]);
                LAGFX_LOG("UNKOP 0x%04x len=%u: %s", hdr->opcode,
                          (unsigned)hdr->payload_size, uh);
            }
            p->unknown_opcode_count++;
            break;
    }
}

/* Drain ring buffer for compute channels. Returns number of commands processed. */
size_t channel_compute_dispatcher_drain(lagfx_protocol_t *p, uint32_t chan_id) {
    if (!lagfx_protocol_is_valid(p)) {
        return 0;
    }

    /* Look up the per-channel ring geometry from the FIFORingDescriptor.
     * Helper lives in ring_common.{c,h}; shared with display dispatcher. */
    uint64_t desc_gpa = 0;
    uint32_t write_ptr = 0;
    uint32_t read_ptr = 0;
    uint64_t page0_gpa = 0;
    if (!lagfx_ring_fifo_descriptor_read(p, chan_id, &desc_gpa, &write_ptr,
                                          &read_ptr, &page0_gpa)) {
        return 0;
    }

    uint32_t ring_size = LAGFX_COMPUTE_RING_SIZE;

    if (write_ptr == read_ptr) {
        LAGFX_TRACE("compute drain: ch=%u caught up rp=0x%x wp=0x%x",
                    chan_id, read_ptr, write_ptr);
        return 0;
    }

    LAGFX_LOG("compute drain: ch=%u desc_gpa=0x%llx page0_gpa=0x%llx "
              "ring_size=0x%x rp=0x%x wp=0x%x",
              chan_id,
              (unsigned long long)desc_gpa,
              (unsigned long long)page0_gpa,
              ring_size, read_ptr, write_ptr);

    lagfx_device_t *dev = (lagfx_device_t *)p->dev;
    if (!dev || !dev->desc.shell.read_memory) {
        LAGFX_WARN("compute drain: no shell.read_memory callback");
        return 0;
    }

    uint32_t rp = read_ptr;
    size_t cmds = 0;
    for (unsigned i = 0; i < LAGFX_COMPUTE_DRAIN_MAX_CMDS; ++i) {
        if (rp == write_ptr) break;

        /* Header read MUST be chunk-resolved: at rp=0xffc the 12-byte header
         * straddles the ring-page boundary and the ring's data pages are NOT
         * physically contiguous — the old flat read pulled the length dword
         * from the wrong physical page → length=0 → "bad length … stop" →
         * every later command in the batch silently dropped. */
        uint8_t hdr_buf[LAGFX_CMD_HEADER_BYTES];
        if (!lagfx_ring_read_bytes(p, page0_gpa, ring_size, rp,
                                    LAGFX_CMD_HEADER_BYTES, hdr_buf)) {
            LAGFX_WARN("compute drain: ch=%u header read failed at rp=0x%x",
                       chan_id, rp);
            break;
        }

        uint16_t opcode       = lagfx_le16(hdr_buf + 0);
        uint16_t arg_count_8b = lagfx_le16(hdr_buf + 2);
        uint32_t length       = lagfx_le32(hdr_buf + 4);
        uint32_t stamp        = lagfx_le32(hdr_buf + 8);

        if (length < LAGFX_CMD_HEADER_BYTES ||
            length > LAGFX_COMPUTE_MAX_CMD_BYTES ||
            length > ring_size) {
            LAGFX_WARN("compute drain: ch=%u bad length 0x%x at rp=0x%x "
                       "opcode=0x%04x first8=%02x%02x%02x%02x%02x%02x%02x%02x — stop",
                       chan_id, length, rp, opcode,
                       hdr_buf[0], hdr_buf[1], hdr_buf[2], hdr_buf[3],
                       hdr_buf[4], hdr_buf[5], hdr_buf[6], hdr_buf[7]);
            break;
        }

        /* Read body. For compute (64 KiB ring) a command can straddle
         * a page boundary within the same ring slot — resolve each
         * 4 KiB chunk independently because page0[i] may map to
         * non-contiguous physical pages.
         *
         * Body lands in the protocol's compute scratch (state.h's
         * "Per-dispatcher scratch buffers"). BQL-serialised. */
        uint8_t *cmd_buf = p->scratch_compute;
        bool body_ok = true;
        uint32_t bytes_read = 0;
        while (bytes_read < length) {
            uint64_t chunk_gpa = 0;
            if (!lagfx_ring_resolve_data_gpa(p, page0_gpa, ring_size,
                                        rp + bytes_read, &chunk_gpa)) {
                body_ok = false;
                break;
            }
            uint32_t chunk_off_in_page = (uint32_t)(chunk_gpa & 0xfffu);
            uint32_t chunk_len = 0x1000u - chunk_off_in_page;
            if (chunk_len > length - bytes_read) {
                chunk_len = length - bytes_read;
            }
            if (!dev->desc.shell.read_memory(dev->desc.shell.opaque,
                                              chunk_gpa, chunk_len,
                                              cmd_buf + bytes_read)) {
                LAGFX_WARN("compute drain: ch=%u body chunk DMA failed",
                           chan_id);
                body_ok = false;
                break;
            }
            bytes_read += chunk_len;
        }
        if (!body_ok) break;

        lagfx_cmd_header_t hdr;
        hdr.opcode       = opcode;
        hdr.arg_count_8b = arg_count_8b;
        hdr.length       = length;
        hdr.stamp        = stamp;
        hdr.payload_size = (uint16_t)((length > LAGFX_CMD_HEADER_BYTES)
                                       ? (length - LAGFX_CMD_HEADER_BYTES)
                                       : 0u);
        hdr.payload      = (hdr.payload_size > 0)
                              ? (cmd_buf + LAGFX_CMD_HEADER_BYTES)
                              : NULL;
        lagfx_dump_cmd("compute", &hdr);

        dispatch_command(p, &hdr);

        /* Per-channel stamp slot = chan_id (1..4 → SLOT_COMPUTE_1..4). */
        lagfx_protocol_complete_stamp_slot(p, chan_id, stamp);

        p->total_cmds_seen++;
        cmds++;
        rp = rp + length;  /* absolute, monotonic — see per-channel-ring-pfn-array.md */
    }

    /* Publish the drained position back to the descriptor so the
     * kext's watchdog observes progress (channel-progress-flag-RE.md). */
    lagfx_ring_publish_read_ptr(p, desc_gpa, rp);

    LAGFX_LOG("compute drain: ch=%u drained=%zu new rp=0x%x",
              chan_id, cmds, rp);
    return cmds;
}
