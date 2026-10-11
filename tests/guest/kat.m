// kat.m — known-answer Metal tests for the paravirtual GPU.
//
// Runs INSIDE the macOS guest, off-screen (works over SSH, no GUI session).
// Each case renders into a small texture, reads it back and compares it with
// an exactly computable expected image. Images are also written to
// /tmp/kat-<case>.ppm for inspection.
//
// Build in the guest (Command Line Tools):
//   clang -fobjc-arc -framework Foundation -framework Metal kat.m -o kat
// Run:
//   ./kat            all cases
//   ./kat triangle   one case
// Exit status: number of failed cases.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#define kSize ((NSUInteger)64)
static const int kTol = 4;   // per-channel tolerance (out of 255)

typedef struct { uint8_t b, g, r, a; } bgra_t;

static id<MTLDevice> gDevice;
static id<MTLCommandQueue> gQueue;

// ---------------------------------------------------------------- helpers

static id<MTLTexture> make_target(void) {
    MTLTextureDescriptor *td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                     width:kSize height:kSize mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeManaged;
    return [gDevice newTextureWithDescriptor:td];
}

static MTLRenderPassDescriptor *clear_pass(id<MTLTexture> t, MTLClearColor c) {
    MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = t;
    rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    rpd.colorAttachments[0].clearColor = c;
    return rpd;
}

// Sync the managed texture, commit, wait (5 s), read back. Returns NO on GPU error.
static BOOL finish(id<MTLCommandBuffer> cb, id<MTLTexture> t, bgra_t *px) {
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit synchronizeResource:t];
    [blit endEncoding];
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    [cb addCompletedHandler:^(id<MTLCommandBuffer> c) { dispatch_semaphore_signal(sem); }];
    [cb commit];
    if (dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC))) {
        printf("  command buffer timed out\n");
        return NO;
    }
    if (cb.status != MTLCommandBufferStatusCompleted) {
        printf("  command buffer status %ld: %s\n", (long)cb.status,
               cb.error ? cb.error.localizedDescription.UTF8String : "(no error)");
        return NO;
    }
    [t getBytes:px bytesPerRow:kSize * 4 fromRegion:MTLRegionMake2D(0, 0, kSize, kSize)
    mipmapLevel:0];
    return YES;
}

static void write_ppm(const char *name, const bgra_t *px) {
    char path[256];
    snprintf(path, sizeof path, "/tmp/kat-%s.ppm", name);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P6\n%lu %lu\n255\n", (unsigned long)kSize, (unsigned long)kSize);
    for (NSUInteger i = 0; i < kSize * kSize; i++) {
        uint8_t rgb[3] = {px[i].r, px[i].g, px[i].b};
        fwrite(rgb, 1, 3, f);
    }
    fclose(f);
}

static BOOL close_to(bgra_t p, bgra_t e) {
    return abs(p.r - e.r) <= kTol && abs(p.g - e.g) <= kTol && abs(p.b - e.b) <= kTol
        && abs(p.a - e.a) <= kTol;
}

// Compare against expect(x, y). Pixels on the triangle's edge (where coverage
// depends on rasterization rules) are skipped via the `edge` callback.
static BOOL compare(const char *name, const bgra_t *px,
                    bgra_t (^expect)(int x, int y), BOOL (^edge)(int x, int y)) {
    int bad = 0, checked = 0;
    for (int y = 0; y < (int)kSize; y++)
        for (int x = 0; x < (int)kSize; x++) {
            if (edge && edge(x, y)) continue;
            checked++;
            bgra_t p = px[y * kSize + x], e = expect(x, y);
            if (!close_to(p, e)) {
                if (bad < 4)
                    printf("  (%d,%d): got rgba %d %d %d %d, want %d %d %d %d\n", x, y,
                           p.r, p.g, p.b, p.a, e.r, e.g, e.b, e.a);
                bad++;
            }
        }
    write_ppm(name, px);
    printf("  %d/%d pixels wrong\n", bad, checked);
    return bad == 0;
}

static id<MTLRenderPipelineState> make_pipeline(NSString *src, MTLVertexDescriptor *vd) {
    NSError *err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:src options:nil error:&err];
    if (!lib) { printf("  shader compile: %s\n", err.localizedDescription.UTF8String); return nil; }
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = [lib newFunctionWithName:@"vs"];
    pd.fragmentFunction = [lib newFunctionWithName:@"fs"];
    pd.vertexDescriptor = vd;
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    id<MTLRenderPipelineState> ps = [gDevice newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!ps) printf("  pipeline: %s\n", err.localizedDescription.UTF8String);
    return ps;
}

static const bgra_t kBlack = {0, 0, 0, 255}, kBlue = {255, 0, 0, 255},
                    kRed = {0, 0, 255, 255}, kGreen = {0, 255, 0, 255};

// Lower-left half triangle in NDC: (-1,-1) (1,-1) (-1,1). In texture
// coordinates (origin top-left) it covers pixels with x < y.
static BOOL in_tri(int x, int y) { return x < y; }
static BOOL on_tri_edge(int x, int y) { return abs(x - y) <= 1; }

// ------------------------------------------------------------------ cases

static BOOL case_clear(void) {
    id<MTLTexture> t = make_target();
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    [[cb renderCommandEncoderWithDescriptor:clear_pass(t, MTLClearColorMake(0, 0, 1, 1))]
        endEncoding];
    bgra_t px[kSize * kSize];
    if (!finish(cb, t, px)) return NO;
    return compare("clear", px, ^bgra_t(int x, int y) { return kBlue; }, nil);
}

static NSString *const kTriangleSrc =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "vertex float4 vs(uint vid [[vertex_id]]) {\n"
     "    const float2 p[3] = { float2(-1, -1), float2(1, -1), float2(-1, 1) };\n"
     "    return float4(p[vid], 0, 1);\n"
     "}\n"
     "fragment float4 fs() { return float4(1, 0, 0, 1); }\n";

static BOOL case_triangle(void) {
    id<MTLRenderPipelineState> ps = make_pipeline(kTriangleSrc, nil);
    if (!ps) return NO;
    id<MTLTexture> t = make_target();
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLRenderCommandEncoder> enc =
        [cb renderCommandEncoderWithDescriptor:clear_pass(t, MTLClearColorMake(0, 0, 0, 1))];
    [enc setRenderPipelineState:ps];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
    bgra_t px[kSize * kSize];
    if (!finish(cb, t, px)) return NO;
    return compare("triangle", px,
                   ^bgra_t(int x, int y) { return in_tri(x, y) ? kRed : kBlack; },
                   ^BOOL(int x, int y) { return on_tri_edge(x, y); });
}

// Interleaved vertex buffer at buffer index 0: float2 position @0 and a
// uchar4-normalized color @8, stride 12 — goes through the serialized vertex
// descriptor on the host.
static NSString *const kTriangleVbufSrc =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "struct VIn { float2 pos [[attribute(0)]]; float4 color [[attribute(1)]]; };\n"
     "struct VOut { float4 pos [[position]]; float4 color; };\n"
     "vertex VOut vs(VIn in [[stage_in]]) {\n"
     "    VOut o; o.pos = float4(in.pos, 0, 1); o.color = in.color; return o;\n"
     "}\n"
     "fragment float4 fs(VOut in [[stage_in]]) { return in.color; }\n";

static BOOL case_triangle_vbuf(void) {
    MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
    vd.attributes[0].format = MTLVertexFormatFloat2;
    vd.attributes[0].offset = 0;
    vd.attributes[0].bufferIndex = 0;
    vd.attributes[1].format = MTLVertexFormatUChar4Normalized;
    vd.attributes[1].offset = 8;
    vd.attributes[1].bufferIndex = 0;
    vd.layouts[0].stride = 12;
    id<MTLRenderPipelineState> ps = make_pipeline(kTriangleVbufSrc, vd);
    if (!ps) return NO;

    struct __attribute__((packed)) { float x, y; uint8_t r, g, b, a; } verts[3] = {
        {-1, -1, 0, 255, 0, 255}, {1, -1, 0, 255, 0, 255}, {-1, 1, 0, 255, 0, 255},
    };
    id<MTLBuffer> vb = [gDevice newBufferWithBytes:verts length:sizeof verts
                                           options:MTLResourceStorageModeShared];
    id<MTLTexture> t = make_target();
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLRenderCommandEncoder> enc =
        [cb renderCommandEncoderWithDescriptor:clear_pass(t, MTLClearColorMake(0, 0, 0, 1))];
    [enc setRenderPipelineState:ps];
    [enc setVertexBuffer:vb offset:0 atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
    bgra_t px[kSize * kSize];
    if (!finish(cb, t, px)) return NO;
    return compare("triangle-vbuf", px,
                   ^bgra_t(int x, int y) { return in_tri(x, y) ? kGreen : kBlack; },
                   ^BOOL(int x, int y) { return on_tri_edge(x, y); });
}

// ------------------------------------------------------------------- main

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        gDevice = MTLCreateSystemDefaultDevice();
        if (!gDevice) { printf("no Metal device\n"); return 100; }
        gQueue = [gDevice newCommandQueue];
        printf("device: %s\n", gDevice.name.UTF8String);

        struct { const char *name; BOOL (*fn)(void); } cases[] = {
            {"clear", case_clear},
            {"triangle", case_triangle},
            {"triangle-vbuf", case_triangle_vbuf},
        };
        int failed = 0, ran = 0;
        for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
            if (argc > 1 && strcmp(argv[1], cases[i].name) != 0) continue;
            printf("[%s]\n", cases[i].name);
            BOOL ok = cases[i].fn();
            printf("  %s\n", ok ? "PASS" : "FAIL");
            failed += !ok;
            ran++;
        }
        printf("%d/%d passed\n", ran - failed, ran);
        return failed;
    }
}
