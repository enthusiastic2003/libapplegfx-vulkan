#include <ApplicationServices/ApplicationServices.h>
#include <stdio.h>

/* Make the NVIDIA HDMI display the main one and mirror every other display
 * (the boot VGA framebuffer) from it, for the current login session. */
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
        if (CGDisplayVendorNumber(ids[i]) == 0x2157) {
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
    CGConfigureDisplayMirrorOfDisplay(cfg, nvidia, kCGNullDirectDisplay);
    CGConfigureDisplayOrigin(cfg, nvidia, 0, 0);
    for (uint32_t i = 0; i < count; i++) {
        if (ids[i] != nvidia) {
            CGConfigureDisplayMirrorOfDisplay(cfg, ids[i], nvidia);
        }
    }
    CGError err = CGCompleteDisplayConfiguration(cfg, kCGConfigureForSession);
    printf("mirroring from NVIDIA display %u for session: %s\n", nvidia,
           err == kCGErrorSuccess ? "ok" : "failed");
    return err == kCGErrorSuccess ? 0 : 4;
}
