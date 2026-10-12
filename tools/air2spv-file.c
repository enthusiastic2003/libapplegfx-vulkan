/*
 * libapplegfx-vulkan — translate a saved shader library offline
 * tools/air2spv-file.c
 *
 * SPDX-License-Identifier: MIT
 *
 * Runs the AIR -> SPIR-V translator on a metallib saved by the device
 * (LAGFX_DUMP_MLIB=1 writes /tmp/mlib-pipe<ref>-<stage>-ref<fn>.metallib), the
 * same way the pipeline path does, so translator changes can be checked
 * without booting a guest:
 *
 *   air2spv-file <in.metallib> vtx|frag <out.spv>
 *
 * Translator warnings go to stderr (LAGFX_LOG_FILE=/dev/stderr is set unless
 * already given); validate the output with spirv-val.
 */

#include "air/bitcode_reader.h"
#include "air2spirv/metallib_extract.h"
#include "air2spv/translate.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 4 || (strcmp(argv[2], "vtx") != 0 && strcmp(argv[2], "frag") != 0)) {
        fprintf(stderr, "usage: %s <in.metallib> vtx|frag <out.spv>\n", argv[0]);
        return 2;
    }
    setenv("LAGFX_LOG_FILE", "/dev/stderr", 0);
    setenv("LAGFX_LOG_LEVEL", "warn", 0);

    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc((size_t)len);
    if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) { fclose(f); return 1; }
    fclose(f);

    lagfx_metallib_function_t fns[8] = {0};
    size_t n = 0;
    if (lagfx_metallib_extract_functions(buf, (size_t)len, fns, 8, &n) != LAGFX_OK || n == 0) {
        fprintf(stderr, "not a metallib with functions\n");
        return 1;
    }
    unsigned want = strcmp(argv[2], "vtx") == 0 ? LAGFX_METALLIB_STAGE_VERTEX
                                                : LAGFX_METALLIB_STAGE_FRAGMENT;
    const lagfx_metallib_function_t *fn = NULL;
    for (size_t i = 0; i < n && i < 8; i++)
        if (fns[i].stage == want) { fn = &fns[i]; break; }
    if (!fn) { fprintf(stderr, "no %s function\n", argv[2]); return 1; }
    fprintf(stderr, "function: %s (%zu bytes of bitcode)\n", fn->name,
            (size_t)fn->bitcode_len);

    lagfx_air_module_t *m = NULL;
    if (lagfx_air_module_open(fn->bitcode, fn->bitcode_len, &m) != LAGFX_OK || !m) {
        fprintf(stderr, "air_module_open failed\n");
        return 1;
    }
    uint8_t *spv = NULL;
    size_t spv_sz = 0;
    lagfx_status_t st = lagfx_air2spv_translate_module(m, &spv, &spv_sz);
    lagfx_air_module_free(m);
    if (st != LAGFX_OK || !spv) { fprintf(stderr, "translate failed (%d)\n", (int)st); return 1; }
    f = fopen(argv[3], "wb");
    if (!f || fwrite(spv, 1, spv_sz, f) != spv_sz) { perror(argv[3]); return 1; }
    fclose(f);
    fprintf(stderr, "wrote %zu bytes\n", spv_sz);
    free(spv);
    free(buf);
    return 0;
}
