#include <ApplicationServices/ApplicationServices.h>
#include <stdio.h>

int main(void)
{
    CGDirectDisplayID ids[8], nvidia = kCGNullDirectDisplay;
    uint32_t count = 0;

    if (CGGetOnlineDisplayList(8, ids, &count) != kCGErrorSuccess) {
        return 1;
    }
    for (uint32_t i = 0; i < count; i++) {
        printf("display %u: %zux%zu vendor=%#x product=%#x main=%d mirror=%u\n",
               ids[i], CGDisplayPixelsWide(ids[i]), CGDisplayPixelsHigh(ids[i]),
               CGDisplayVendorNumber(ids[i]), CGDisplayModelNumber(ids[i]),
               CGDisplayIsMain(ids[i]), CGDisplayMirrorsDisplay(ids[i]));
        /* the capture card on HDMI (0x2157), or NVRMFB's forced EDID "NVS" (0x3ad3) */
        if (CGDisplayVendorNumber(ids[i]) == 0x2157 || CGDisplayVendorNumber(ids[i]) == 0x3ad3) {
            nvidia = ids[i];
        }
    }
    if (nvidia == kCGNullDirectDisplay) {
        fprintf(stderr, "NVIDIA HDMI display not found\n");
        return 2;
    }

    CGDisplayConfigRef cfg;
    if (CGBeginDisplayConfiguration(&cfg) != kCGErrorSuccess) {
        return 3;
    }
    for (uint32_t i = 0; i < count; i++) {
        CGConfigureDisplayMirrorOfDisplay(cfg, ids[i], kCGNullDirectDisplay);
    }
    CGConfigureDisplayOrigin(cfg, nvidia, 0, 0);
    int32_t x = -(int32_t)CGDisplayPixelsWide(ids[0]);
    for (uint32_t i = 0; i < count; i++) {
        if (ids[i] != nvidia) {
            CGConfigureDisplayOrigin(cfg, ids[i], x, 0);
            x -= (int32_t)CGDisplayPixelsWide(ids[i]);
        }
    }
    CGError err = CGCompleteDisplayConfiguration(cfg, kCGConfigureForSession);
    printf("NVIDIA display %u made primary for session: %s\n", nvidia,
           err == kCGErrorSuccess ? "ok" : "failed");
    return err == kCGErrorSuccess ? 0 : 4;
}
