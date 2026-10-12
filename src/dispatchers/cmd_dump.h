/*
 * libapplegfx-vulkan — command dump for wire decoding
 * src/dispatchers/cmd_dump.h
 *
 * SPDX-License-Identifier: MIT
 *
 * LAGFX_DUMP_CMDS=1: log every command a dispatcher receives (channel,
 * opcode, length, first 64 payload bytes) at warn level. A channel name
 * ("ch0", "compute", "display") limits it to that channel.
 */
#ifndef LAGFX_DISPATCHERS_CMD_DUMP_H
#define LAGFX_DISPATCHERS_CMD_DUMP_H

#include "common/log.h"
#include "protocol/opcodes.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline void lagfx_dump_cmd(const char *chan, const lagfx_cmd_header_t *hdr) {
    const char *want = getenv("LAGFX_DUMP_CMDS");
    if (!want || (strcmp(want, "1") != 0 && strcmp(want, chan) != 0)) return;
    char hx[3 * 64 + 1] = "";
    size_t hn = 0;
    for (uint32_t k = 0; hdr->payload && k < hdr->payload_size && k < 64u; k++)
        hn += (size_t)snprintf(hx + hn, sizeof hx - hn, "%02x ", hdr->payload[k]);
    LAGFX_WARN("CMD %s op=0x%04x len=%u: %s", chan, (unsigned)hdr->opcode,
               (unsigned)hdr->payload_size, hx);
}

#endif /* LAGFX_DISPATCHERS_CMD_DUMP_H */
