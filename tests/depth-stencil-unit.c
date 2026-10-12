/* Depth-stencil state object decoder, against kat stencil-probe's object. */
#include "protocol/depth_stencil.h"

#include <stdio.h>

static int failures;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); failures++; } } while (0)

int main(void) {
    /* front: Equal, Replace / Invert / IncrementWrap, masks 0x3c / 0xc3;
     * back: Greater, Zero / DecrementClamp / Keep, masks 0x0f / 0xf0 */
    static const uint8_t b[0x28] = {
        0x04, 0, 0, 0, 0x28, 0, 0, 0, 0x05, 0, 0, 0, 0x37, 0, 0, 0,
        0x52, 0x0d, 0, 0, 0x3c, 0, 0, 0, 0xc3, 0, 0, 0, 0x0c, 0x01, 0, 0,
        0x0f, 0, 0, 0, 0xf0, 0, 0, 0,
    };
    lagfx_depth_stencil_t d;
    CHECK(lagfx_parse_depth_stencil_object(b, sizeof b, &d));
    CHECK(d.depth_compare == 7 && !d.depth_write);
    CHECK(d.front.compare == 2 && d.front.stencil_fail == 2 && d.front.depth_fail == 5
          && d.front.pass == 6 && d.front.read_mask == 0x3c && d.front.write_mask == 0xc3);
    CHECK(d.back.compare == 4 && d.back.stencil_fail == 1 && d.back.depth_fail == 4
          && d.back.pass == 0 && d.back.read_mask == 0x0f && d.back.write_mask == 0xf0);
    CHECK(!lagfx_parse_depth_stencil_object(b, 0x20, &d));
    if (failures) return 1;
    printf("depth-stencil-unit: OK\n");
    return 0;
}
