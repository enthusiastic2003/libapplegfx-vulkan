// kat.m — known-answer Metal tests for the paravirtual GPU.
//
// Runs INSIDE the macOS guest, off-screen (works over SSH, no GUI session).
// Each case renders (or computes) into a small resource, reads it back and
// compares it with an exactly computable expected result. Images are written
// to /tmp/kat-<case>.ppm; on failure the expected image is written next to it
// as /tmp/kat-<case>.want.ppm.
//
// Build in the guest (Command Line Tools):
//   clang -fobjc-arc -framework Foundation -framework Metal -framework IOSurface kat.m -o kat
// Run:
//   ./kat                 all cases, each in its own process (a crash or hang
//                         in one case doesn't stop the others)
//   ./kat a b ...         the named cases, each in its own process
//   ./kat -1 <case>       one case in this process (what the runner spawns)
//   ./kat -l              list cases
// Exit status: number of cases that did not pass.

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>

extern char **environ;

#define kSize ((NSUInteger)64)
static const int kTol = 4;                 // per-channel tolerance (out of 255)
static const int kCaseTimeoutSec = 20;     // per case, enforced by the runner
static const int kGpuTimeoutSec = 3;       // per command buffer

typedef struct { uint8_t b, g, r, a; } bgra_t;
typedef bgra_t (^expect_fn)(int x, int y);
typedef BOOL (^skip_fn)(int x, int y);

static id<MTLDevice> gDevice;
static id<MTLCommandQueue> gQueue;
static id<MTLLibrary> gLib;
static const char *gCase;

// Print a step only when it took a while, so a hang's location is visible
// (the runner kills the case after kCaseTimeoutSec) without cluttering passes.
static void step(const char *what) {
    static CFAbsoluteTime last;
    CFAbsoluteTime now = CFAbsoluteTimeGetCurrent();
    if (last != 0 && now - last > 0.5) printf("  (%.1f s until %s)\n", now - last, what);
    last = now;
}

static bgra_t px_rgba(int r, int g, int b, int a) {
    return (bgra_t){(uint8_t)b, (uint8_t)g, (uint8_t)r, (uint8_t)a};
}
static int unorm8(double v) {
    v = v < 0 ? 0 : v > 1 ? 1 : v;
    return (int)lround(v * 255.0);
}

static const bgra_t kBlack = {0, 0, 0, 255}, kBlue = {255, 0, 0, 255},
                    kRed = {0, 0, 255, 255}, kGreen = {0, 255, 0, 255};

// ------------------------------------------------------------- shaders

// One library for every case. Full-screen quads come from vs_quad (vertex_id
// only, uv origin top-left, matching texture coordinates).
static NSString *const kShaderSrc =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "struct QOut { float4 pos [[position]]; float2 uv; };\n"
     "vertex QOut vs_quad(uint vid [[vertex_id]]) {\n"
     "    const float2 p[6] = { float2(-1,-1), float2(1,-1), float2(-1,1),\n"
     "                          float2(-1,1),  float2(1,-1), float2(1,1) };\n"
     "    QOut o; o.pos = float4(p[vid], 0, 1);\n"
     "    o.uv = float2((p[vid].x + 1) * 0.5, (1 - p[vid].y) * 0.5); return o;\n"
     "}\n"
     "vertex float4 vs_tri(uint vid [[vertex_id]]) {\n"
     "    const float2 p[3] = { float2(-1, -1), float2(1, -1), float2(-1, 1) };\n"
     "    return float4(p[vid], 0, 1);\n"
     "}\n"
     "fragment float4 fs_red() { return float4(1, 0, 0, 1); }\n"
     // vertex buffer with a serialized vertex descriptor
     "struct VIn { float2 pos [[attribute(0)]]; float4 color [[attribute(1)]]; };\n"
     "struct VOut { float4 pos [[position]]; float4 color; };\n"
     "vertex VOut vs_vbuf(VIn in [[stage_in]]) {\n"
     "    VOut o; o.pos = float4(in.pos, 0, 1); o.color = in.color; return o;\n"
     "}\n"
     "fragment float4 fs_color(VOut in [[stage_in]]) { return in.color; }\n"
     // a constant color from a bound buffer / setFragmentBytes
     "fragment float4 fs_const(constant float4 &c [[buffer(0)]]) { return c; }\n"
     // texture sampling: constexpr sampler, and a sampler state object
     "fragment float4 fs_tex_nearest(QOut in [[stage_in]], texture2d<float> t [[texture(0)]]) {\n"
     "    constexpr sampler s(filter::nearest, address::clamp_to_edge);\n"
     "    return t.sample(s, in.uv);\n"
     "}\n"
     "fragment float4 fs_tex_sampler(QOut in [[stage_in]], texture2d<float> t [[texture(0)]],\n"
     "                               sampler s [[sampler(0)]]) {\n"
     "    return t.sample(s, in.uv);\n"
     "}\n"
     // sampler probe: one constexpr sampler per function (see case_sampler_probe)
     "fragment float4 fs_sp0(QOut in [[stage_in]], texture2d<float> t [[texture(0)]]) {\n"
     "    constexpr sampler s(filter::nearest); return t.sample(s, in.uv);\n"
     "}\n"
     "fragment float4 fs_sp1(QOut in [[stage_in]], texture2d<float> t [[texture(0)]]) {\n"
     "    constexpr sampler s(filter::linear); return t.sample(s, in.uv);\n"
     "}\n"
     "fragment float4 fs_sp2(QOut in [[stage_in]], texture2d<float> t [[texture(0)]]) {\n"
     "    constexpr sampler s(mag_filter::linear, min_filter::nearest); return t.sample(s, in.uv);\n"
     "}\n"
     "fragment float4 fs_sp3(QOut in [[stage_in]], texture2d<float> t [[texture(0)]]) {\n"
     "    constexpr sampler s(address::repeat); return t.sample(s, in.uv);\n"
     "}\n"
     "fragment float4 fs_sp4(QOut in [[stage_in]], texture2d<float> t [[texture(0)]]) {\n"
     "    constexpr sampler s(address::mirrored_repeat); return t.sample(s, in.uv);\n"
     "}\n"
     "fragment float4 fs_sp5(QOut in [[stage_in]], texture2d<float> t [[texture(0)]]) {\n"
     "    constexpr sampler s(s_address::repeat, t_address::clamp_to_edge); return t.sample(s, in.uv);\n"
     "}\n"
     "fragment float4 fs_sp6(QOut in [[stage_in]], texture2d<float> t [[texture(0)]]) {\n"
     "    constexpr sampler s(coord::pixel); return t.sample(s, in.uv);\n"
     "}\n"
     "fragment float4 fs_sp7(QOut in [[stage_in]], texture2d<float> t [[texture(0)]]) {\n"
     "    constexpr sampler s(address::clamp_to_zero); return t.sample(s, in.uv);\n"
     "}\n"
     "fragment float4 fs_sp8(QOut in [[stage_in]], texture2d<float> t [[texture(0)]]) {\n"
     "    constexpr sampler s(mip_filter::linear); return t.sample(s, in.uv);\n"
     "}\n"
     "fragment float4 fs_sp9(QOut in [[stage_in]], texture2d<float> t [[texture(0)]]) {\n"
     "    constexpr sampler s(max_anisotropy(4)); return t.sample(s, in.uv);\n"
     "}\n"
     "fragment float4 fs_sp10(QOut in [[stage_in]], texture2d<float> t [[texture(0)]]) {\n"
     "    constexpr sampler s(filter::linear, address::repeat); return t.sample(s, in.uv);\n"
     "}\n"
     "fragment float4 fs_sp11(QOut in [[stage_in]], texture2d<float> t [[texture(0)]]) {\n"
     "    constexpr sampler s(address::clamp_to_border, border_color::opaque_white); return t.sample(s, in.uv);\n"
     "}\n"
     // compute
     "kernel void k_fill(device uint *out [[buffer(0)]], constant uint &mul [[buffer(1)]],\n"
     "                   uint i [[thread_position_in_grid]]) { out[i] = i * mul + 1u; }\n";

// ------------------------------------------------------------- helpers

static id<MTLTexture> make_texture(NSUInteger w, NSUInteger h, MTLTextureUsage usage) {
    MTLTextureDescriptor *td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                     width:w height:h mipmapped:NO];
    td.usage = usage;
    td.storageMode = MTLStorageModeManaged;
    return [gDevice newTextureWithDescriptor:td];
}

static id<MTLTexture> make_target(void) {
    return make_texture(kSize, kSize, MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead);
}

static MTLRenderPassDescriptor *pass(id<MTLTexture> t, MTLLoadAction load, MTLClearColor c) {
    MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = t;
    rpd.colorAttachments[0].loadAction = load;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    rpd.colorAttachments[0].clearColor = c;
    return rpd;
}
static MTLRenderPassDescriptor *clear_pass(id<MTLTexture> t, MTLClearColor c) {
    return pass(t, MTLLoadActionClear, c);
}

// Commit, wait (bounded), report GPU errors.
static BOOL run_cb(id<MTLCommandBuffer> cb) {
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    [cb addCompletedHandler:^(id<MTLCommandBuffer> c) { dispatch_semaphore_signal(sem); }];
    step("commit");
    [cb commit];
    if (dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW,
                                                   (int64_t)kGpuTimeoutSec * NSEC_PER_SEC))) {
        printf("  command buffer timed out\n");
        return NO;
    }
    step("completion");
    if (cb.status != MTLCommandBufferStatusCompleted) {
        printf("  command buffer status %ld: %s\n", (long)cb.status,
               cb.error ? cb.error.localizedDescription.UTF8String : "(no error)");
        return NO;
    }
    return YES;
}

// Sync the managed texture back to the CPU, commit, wait, read back.
static BOOL finish(id<MTLCommandBuffer> cb, id<MTLTexture> t, bgra_t *px) {
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit synchronizeResource:t];
    [blit endEncoding];
    if (!run_cb(cb)) return NO;
    [t getBytes:px bytesPerRow:kSize * 4 fromRegion:MTLRegionMake2D(0, 0, kSize, kSize)
    mipmapLevel:0];
    return YES;
}

static void write_ppm(const char *name, const char *suffix, const bgra_t *px) {
    char path[256];
    snprintf(path, sizeof path, "/tmp/kat-%s%s.ppm", name, suffix);
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

// Compare a kSize x kSize image against expect(x, y). Pixels where `skip`
// returns YES (e.g. on a triangle edge, where coverage depends on
// rasterization rules) are not checked. A uniformly wrong image is reported as
// such, which usually means the draw didn't happen at all.
static BOOL compare(const bgra_t *px, expect_fn expect, skip_fn skip) {
    static bgra_t want[kSize * kSize];
    int bad = 0, checked = 0;
    for (int y = 0; y < (int)kSize; y++)
        for (int x = 0; x < (int)kSize; x++) {
            bgra_t e = expect(x, y);
            want[y * kSize + x] = e;
            if (skip && skip(x, y)) continue;
            checked++;
            bgra_t p = px[y * kSize + x];
            if (!close_to(p, e)) {
                if (bad < 4)
                    printf("  (%d,%d): got rgba %d %d %d %d, want %d %d %d %d\n", x, y,
                           p.r, p.g, p.b, p.a, e.r, e.g, e.b, e.a);
                bad++;
            }
        }
    BOOL uniform = YES;
    for (NSUInteger i = 1; i < kSize * kSize && uniform; i++)
        uniform = memcmp(&px[i], &px[0], sizeof px[0]) == 0;
    write_ppm(gCase, "", px);
    if (bad) write_ppm(gCase, ".want", want);
    printf("  %d/%d pixels wrong%s\n", bad, checked,
           bad && uniform ? " (image is a single color)" : "");
    return bad == 0;
}

typedef void (^blend_cfg)(MTLRenderPipelineColorAttachmentDescriptor *);

static id<MTLRenderPipelineState> make_pipeline(NSString *vs, NSString *fs,
                                                MTLVertexDescriptor *vd, blend_cfg blend) {
    NSError *err = nil;
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = [gLib newFunctionWithName:vs];
    pd.fragmentFunction = [gLib newFunctionWithName:fs];
    pd.vertexDescriptor = vd;
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    if (blend) blend(pd.colorAttachments[0]);
    id<MTLRenderPipelineState> ps = [gDevice newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!ps) printf("  pipeline: %s\n", err.localizedDescription.UTF8String);
    return ps;
}

// Lower-left half triangle in NDC: (-1,-1) (1,-1) (-1,1). In texture
// coordinates (origin top-left) it covers pixels with x < y.
static BOOL in_tri(int x, int y) { return x < y; }
static skip_fn tri_edge = ^BOOL(int x, int y) { return abs(x - y) <= 1; };

// Draw a full-screen quad with `ps` over a target cleared to `bg`; `bind`
// sets whatever the fragment shader needs.
static BOOL draw_quad(id<MTLRenderPipelineState> ps, MTLClearColor bg,
                      void (^bind)(id<MTLRenderCommandEncoder>), bgra_t *px) {
    if (!ps) return NO;
    id<MTLTexture> t = make_target();
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:clear_pass(t, bg)];
    [enc setRenderPipelineState:ps];
    if (bind) bind(enc);
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    [enc endEncoding];
    return finish(cb, t, px);
}

// 8x8 texel pattern, every texel distinct and the image asymmetric so flips
// and transposes show up.
#define kPat 8
static bgra_t pattern(int i, int j) {
    return px_rgba(i * 36, j * 36, (j * kPat + i) * 4, 255);
}
static void fill_pattern(uint8_t *base, size_t bpr) {
    for (int j = 0; j < kPat; j++)
        for (int i = 0; i < kPat; i++)
            *(bgra_t *)(base + j * bpr + i * 4) = pattern(i, j);
}
// Nearest-sampling an 8x8 texture over 64x64 pixels: 8x8 blocks.
static expect_fn pattern_blocks = ^bgra_t(int x, int y) {
    return pattern(x / (int)(kSize / kPat), y / (int)(kSize / kPat));
};

// ------------------------------------------------------------------ cases

static BOOL case_clear(void) {
    id<MTLTexture> t = make_target();
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    [[cb renderCommandEncoderWithDescriptor:clear_pass(t, MTLClearColorMake(0, 0, 1, 1))]
        endEncoding];
    static bgra_t px[kSize * kSize];
    if (!finish(cb, t, px)) return NO;
    return compare(px, ^bgra_t(int x, int y) { return kBlue; }, nil);
}

static BOOL case_triangle(void) {
    id<MTLRenderPipelineState> ps = make_pipeline(@"vs_tri", @"fs_red", nil, nil);
    if (!ps) return NO;
    id<MTLTexture> t = make_target();
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLRenderCommandEncoder> enc =
        [cb renderCommandEncoderWithDescriptor:clear_pass(t, MTLClearColorMake(0, 0, 0, 1))];
    [enc setRenderPipelineState:ps];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
    static bgra_t px[kSize * kSize];
    if (!finish(cb, t, px)) return NO;
    return compare(px, ^bgra_t(int x, int y) { return in_tri(x, y) ? kRed : kBlack; }, tri_edge);
}

// Interleaved vertex buffer at buffer index 0: float2 position @0 and a
// uchar4-normalized color @8, stride 12 — goes through the serialized vertex
// descriptor on the host.
static BOOL case_triangle_vbuf(void) {
    MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
    vd.attributes[0].format = MTLVertexFormatFloat2;
    vd.attributes[0].offset = 0;
    vd.attributes[0].bufferIndex = 0;
    vd.attributes[1].format = MTLVertexFormatUChar4Normalized;
    vd.attributes[1].offset = 8;
    vd.attributes[1].bufferIndex = 0;
    vd.layouts[0].stride = 12;
    id<MTLRenderPipelineState> ps = make_pipeline(@"vs_vbuf", @"fs_color", vd, nil);
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
    static bgra_t px[kSize * kSize];
    if (!finish(cb, t, px)) return NO;
    return compare(px, ^bgra_t(int x, int y) { return in_tri(x, y) ? kGreen : kBlack; },
                   tri_edge);
}

// Fragment constant from setFragmentBytes (inline data in the command stream).
static BOOL case_frag_bytes(void) {
    static bgra_t px[kSize * kSize];
    static const float c[4] = {1.0f, 0.5f, 0.0f, 1.0f};
    if (!draw_quad(make_pipeline(@"vs_quad", @"fs_const", nil, nil), MTLClearColorMake(0, 0, 0, 1),
                   ^(id<MTLRenderCommandEncoder> e) { [e setFragmentBytes:c length:sizeof c atIndex:0]; },
                   px))
        return NO;
    return compare(px, ^bgra_t(int x, int y) { return px_rgba(255, 128, 0, 255); }, nil);
}

// Fragment constant from a shared MTLBuffer bound at a non-zero offset.
static BOOL case_frag_buffer(void) {
    float data[16] = {0};
    data[8] = 0.2f; data[9] = 0.4f; data[10] = 0.6f; data[11] = 1.0f;   // at offset 32
    id<MTLBuffer> b = [gDevice newBufferWithBytes:data length:sizeof data
                                          options:MTLResourceStorageModeShared];
    static bgra_t px[kSize * kSize];
    if (!draw_quad(make_pipeline(@"vs_quad", @"fs_const", nil, nil), MTLClearColorMake(0, 0, 0, 1),
                   ^(id<MTLRenderCommandEncoder> e) { [e setFragmentBuffer:b offset:32 atIndex:0]; },
                   px))
        return NO;
    return compare(px, ^bgra_t(int x, int y) { return px_rgba(51, 102, 153, 255); }, nil);
}

// Sample a CPU-uploaded (replaceRegion) managed texture with a constexpr
// nearest sampler.
static BOOL case_tex_nearest(void) {
    id<MTLTexture> src = make_texture(kPat, kPat, MTLTextureUsageShaderRead);
    static uint8_t texels[kPat * kPat * 4];
    fill_pattern(texels, kPat * 4);
    [src replaceRegion:MTLRegionMake2D(0, 0, kPat, kPat) mipmapLevel:0 withBytes:texels
           bytesPerRow:kPat * 4];
    static bgra_t px[kSize * kSize];
    if (!draw_quad(make_pipeline(@"vs_quad", @"fs_tex_nearest", nil, nil),
                   MTLClearColorMake(0, 0, 0, 1),
                   ^(id<MTLRenderCommandEncoder> e) { [e setFragmentTexture:src atIndex:0]; }, px))
        return NO;
    return compare(px, pattern_blocks, nil);
}

// Bilinear filtering through a sampler state object: a 2x2 texture (black,
// red / green, blue) stretched over the target, clamp-to-edge.
static BOOL case_tex_linear(void) {
    id<MTLTexture> src = make_texture(2, 2, MTLTextureUsageShaderRead);
    const bgra_t c[4] = {kBlack, kRed, kGreen, kBlue};
    [src replaceRegion:MTLRegionMake2D(0, 0, 2, 2) mipmapLevel:0 withBytes:c bytesPerRow:8];
    MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
    sd.minFilter = MTLSamplerMinMagFilterLinear;
    sd.magFilter = MTLSamplerMinMagFilterLinear;
    sd.sAddressMode = MTLSamplerAddressModeClampToEdge;
    sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
    id<MTLSamplerState> ss = [gDevice newSamplerStateWithDescriptor:sd];
    static bgra_t px[kSize * kSize];
    if (!draw_quad(make_pipeline(@"vs_quad", @"fs_tex_sampler", nil, nil),
                   MTLClearColorMake(0, 0, 0, 1),
                   ^(id<MTLRenderCommandEncoder> e) {
                       [e setFragmentTexture:src atIndex:0];
                       [e setFragmentSamplerState:ss atIndex:0];
                   },
                   px))
        return NO;
    return compare(px, ^bgra_t(int x, int y) {
        double fx = ((x + 0.5) / kSize) * 2 - 0.5, fy = ((y + 0.5) / kSize) * 2 - 0.5;
        fx = fx < 0 ? 0 : fx > 1 ? 1 : fx;
        fy = fy < 0 ? 0 : fy > 1 ? 1 : fy;
        // texel colors as rgb: (0,0)=black (1,0)=red (0,1)=green (1,1)=blue
        double r = fx * (1 - fy), g = (1 - fx) * fy, b = fx * fy;
        return px_rgba(unorm8(r), unorm8(g), unorm8(b), 255);
    }, nil);
}

// Classic (non-premultiplied) alpha blending: 25% red over blue.
static BOOL case_blend_alpha(void) {
    static const float c[4] = {1.0f, 0.0f, 0.0f, 0.25f};
    id<MTLRenderPipelineState> ps = make_pipeline(@"vs_quad", @"fs_const", nil,
        ^(MTLRenderPipelineColorAttachmentDescriptor *a) {
            a.blendingEnabled = YES;
            a.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
            a.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            a.sourceAlphaBlendFactor = MTLBlendFactorOne;
            a.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        });
    static bgra_t px[kSize * kSize];
    if (!draw_quad(ps, MTLClearColorMake(0, 0, 1, 1),
                   ^(id<MTLRenderCommandEncoder> e) { [e setFragmentBytes:c length:sizeof c atIndex:0]; },
                   px))
        return NO;
    return compare(px, ^bgra_t(int x, int y) { return px_rgba(64, 0, 191, 255); }, nil);
}

// Premultiplied source-over (what the compositor uses): 50% red over green.
static BOOL case_blend_premul(void) {
    static const float c[4] = {0.5f, 0.0f, 0.0f, 0.5f};
    id<MTLRenderPipelineState> ps = make_pipeline(@"vs_quad", @"fs_const", nil,
        ^(MTLRenderPipelineColorAttachmentDescriptor *a) {
            a.blendingEnabled = YES;
            a.sourceRGBBlendFactor = MTLBlendFactorOne;
            a.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            a.sourceAlphaBlendFactor = MTLBlendFactorOne;
            a.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        });
    static bgra_t px[kSize * kSize];
    if (!draw_quad(ps, MTLClearColorMake(0, 1, 0, 1),
                   ^(id<MTLRenderCommandEncoder> e) { [e setFragmentBytes:c length:sizeof c atIndex:0]; },
                   px))
        return NO;
    return compare(px, ^bgra_t(int x, int y) { return px_rgba(128, 128, 0, 255); }, nil);
}

// Contents survive across command buffers: clear in one, Load + draw in the next.
static BOOL case_load(void) {
    id<MTLRenderPipelineState> ps = make_pipeline(@"vs_tri", @"fs_red", nil, nil);
    if (!ps) return NO;
    id<MTLTexture> t = make_target();
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    [[cb renderCommandEncoderWithDescriptor:clear_pass(t, MTLClearColorMake(0, 0, 1, 1))]
        endEncoding];
    if (!run_cb(cb)) return NO;
    cb = [gQueue commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:
        pass(t, MTLLoadActionLoad, MTLClearColorMake(0, 0, 0, 1))];
    [enc setRenderPipelineState:ps];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
    static bgra_t px[kSize * kSize];
    if (!finish(cb, t, px)) return NO;
    return compare(px, ^bgra_t(int x, int y) { return in_tri(x, y) ? kRed : kBlue; }, tri_edge);
}

// Blit encoder: copy a shared buffer into a texture.
static BOOL case_blit_copy(void) {
    static bgra_t img[kSize * kSize];
    expect_fn grad = ^bgra_t(int x, int y) { return px_rgba(x * 4, y * 4, (x ^ y) * 4, 255); };
    for (int y = 0; y < (int)kSize; y++)
        for (int x = 0; x < (int)kSize; x++) img[y * kSize + x] = grad(x, y);
    id<MTLBuffer> b = [gDevice newBufferWithBytes:img length:sizeof img
                                          options:MTLResourceStorageModeShared];
    id<MTLTexture> t = make_target();
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:b sourceOffset:0 sourceBytesPerRow:kSize * 4
         sourceBytesPerImage:sizeof img sourceSize:MTLSizeMake(kSize, kSize, 1)
                   toTexture:t destinationSlice:0 destinationLevel:0
           destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
    static bgra_t px[kSize * kSize];
    if (!finish(cb, t, px)) return NO;
    return compare(px, grad, nil);
}

static IOSurfaceRef make_surface(int w, int h) {
    NSDictionary *props = @{
        (id)kIOSurfaceWidth: @(w), (id)kIOSurfaceHeight: @(h),
        (id)kIOSurfaceBytesPerElement: @4, (id)kIOSurfacePixelFormat: @((uint32_t)'BGRA'),
    };
    return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

static id<MTLTexture> surface_texture(IOSurfaceRef s, int w, int h, MTLTextureUsage usage) {
    MTLTextureDescriptor *td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:w height:h mipmapped:NO];
    td.usage = usage;
    return [gDevice newTextureWithDescriptor:td iosurface:s plane:0];
}

// Sample an IOSurface the CPU filled (how window contents reach the compositor).
static BOOL case_iosurface_sample(void) {
    IOSurfaceRef s = make_surface(kPat, kPat);
    if (!s) { printf("  IOSurfaceCreate failed\n"); return NO; }
    IOSurfaceLock(s, 0, NULL);
    fill_pattern(IOSurfaceGetBaseAddress(s), IOSurfaceGetBytesPerRow(s));
    IOSurfaceUnlock(s, 0, NULL);
    id<MTLTexture> src = surface_texture(s, kPat, kPat, MTLTextureUsageShaderRead);
    static bgra_t px[kSize * kSize];
    BOOL ok = draw_quad(make_pipeline(@"vs_quad", @"fs_tex_nearest", nil, nil),
                        MTLClearColorMake(0, 0, 0, 1),
                        ^(id<MTLRenderCommandEncoder> e) { [e setFragmentTexture:src atIndex:0]; },
                        px)
              && compare(px, pattern_blocks, nil);
    CFRelease(s);
    return ok;
}

// Render into an IOSurface and read it on the CPU through the surface (how the
// compositor's output reaches the display).
static BOOL case_iosurface_target(void) {
    id<MTLRenderPipelineState> ps = make_pipeline(@"vs_tri", @"fs_red", nil, nil);
    if (!ps) return NO;
    IOSurfaceRef s = make_surface(kSize, kSize);
    if (!s) { printf("  IOSurfaceCreate failed\n"); return NO; }
    id<MTLTexture> t = surface_texture(s, kSize, kSize,
                                       MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead);
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLRenderCommandEncoder> enc =
        [cb renderCommandEncoderWithDescriptor:clear_pass(t, MTLClearColorMake(0, 0, 1, 1))];
    [enc setRenderPipelineState:ps];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
    BOOL ok = run_cb(cb);
    if (ok) {
        static bgra_t px[kSize * kSize];
        IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
        const uint8_t *base = IOSurfaceGetBaseAddress(s);
        size_t bpr = IOSurfaceGetBytesPerRow(s);
        for (NSUInteger y = 0; y < kSize; y++) memcpy(&px[y * kSize], base + y * bpr, kSize * 4);
        IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
        ok = compare(px, ^bgra_t(int x, int y) { return in_tri(x, y) ? kRed : kBlue; }, tri_edge);
    }
    CFRelease(s);
    return ok;
}

// Compute: out[i] = i * 3 + 1 into a shared buffer.
static BOOL case_compute(void) {
    const NSUInteger n = 4096;
    NSError *err = nil;
    id<MTLComputePipelineState> ps =
        [gDevice newComputePipelineStateWithFunction:[gLib newFunctionWithName:@"k_fill"] error:&err];
    if (!ps) { printf("  pipeline: %s\n", err.localizedDescription.UTF8String); return NO; }
    id<MTLBuffer> out = [gDevice newBufferWithLength:n * 4 options:MTLResourceStorageModeShared];
    memset(out.contents, 0xee, n * 4);
    const uint32_t mul = 3;
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:out offset:0 atIndex:0];
    [enc setBytes:&mul length:sizeof mul atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake(n / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding];
    if (!run_cb(cb)) return NO;
    const uint32_t *v = out.contents;
    int bad = 0;
    for (NSUInteger i = 0; i < n; i++)
        if (v[i] != i * mul + 1) {
            if (bad < 4) printf("  out[%lu] = 0x%x, want 0x%lx\n", (unsigned long)i, v[i],
                                (unsigned long)(i * mul + 1));
            bad++;
        }
    printf("  %d/%lu values wrong\n", bad, (unsigned long)n);
    return bad == 0;
}

// Not a pass/fail case: draws with known samplers so the host-side dumps
// (LAGFX_DUMP_AIR_GLOBALS=1 for constexpr samplers, LAGFX_DUMP_OBJ=1 for
// sampler state objects) can be matched to the settings.
static const char *const kProbeSamplers[] = {
    "filter::nearest",
    "filter::linear",
    "mag_filter::linear, min_filter::nearest",
    "address::repeat",
    "address::mirrored_repeat",
    "s_address::repeat, t_address::clamp_to_edge",
    "coord::pixel",
    "address::clamp_to_zero",
    "mip_filter::linear",
    "max_anisotropy(4)",
    "filter::linear, address::repeat",
    "address::clamp_to_border, border_color::opaque_white",
};

static BOOL case_sampler_probe(void) {
    id<MTLTexture> src = make_texture(2, 2, MTLTextureUsageShaderRead);
    const bgra_t c[4] = {kBlack, kRed, kGreen, kBlue};
    [src replaceRegion:MTLRegionMake2D(0, 0, 2, 2) mipmapLevel:0 withBytes:c bytesPerRow:8];
    static bgra_t px[kSize * kSize];
    for (size_t i = 0; i < sizeof kProbeSamplers / sizeof kProbeSamplers[0]; i++) {
        printf("  constexpr %zu: %s\n", i, kProbeSamplers[i]);
        draw_quad(make_pipeline(@"vs_quad", [NSString stringWithFormat:@"fs_sp%zu", i], nil, nil),
                  MTLClearColorMake(0, 0, 0, 1),
                  ^(id<MTLRenderCommandEncoder> e) { [e setFragmentTexture:src atIndex:0]; }, px);
    }
    struct { const char *what; void (^set)(MTLSamplerDescriptor *); } objs[] = {
        {"default", ^(MTLSamplerDescriptor *d) {}},
        {"min/mag linear", ^(MTLSamplerDescriptor *d) {
            d.minFilter = MTLSamplerMinMagFilterLinear; d.magFilter = MTLSamplerMinMagFilterLinear; }},
        {"mag linear only", ^(MTLSamplerDescriptor *d) { d.magFilter = MTLSamplerMinMagFilterLinear; }},
        {"s repeat", ^(MTLSamplerDescriptor *d) { d.sAddressMode = MTLSamplerAddressModeRepeat; }},
        {"t mirror", ^(MTLSamplerDescriptor *d) { d.tAddressMode = MTLSamplerAddressModeMirrorRepeat; }},
        {"unnormalized", ^(MTLSamplerDescriptor *d) { d.normalizedCoordinates = NO; }},
        {"mip linear", ^(MTLSamplerDescriptor *d) { d.mipFilter = MTLSamplerMipFilterLinear; }},
        {"aniso 4", ^(MTLSamplerDescriptor *d) { d.maxAnisotropy = 4; }},
        {"lod 1..5", ^(MTLSamplerDescriptor *d) { d.lodMinClamp = 1; d.lodMaxClamp = 5; }},
        {"border white", ^(MTLSamplerDescriptor *d) {
            d.sAddressMode = MTLSamplerAddressModeClampToBorderColor;
            d.borderColor = MTLSamplerBorderColorOpaqueWhite; }},
    };
    for (size_t i = 0; i < sizeof objs / sizeof objs[0]; i++) {
        MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
        objs[i].set(sd);
        id<MTLSamplerState> ss = [gDevice newSamplerStateWithDescriptor:sd];
        printf("  object %zu: %s\n", i, objs[i].what);
        draw_quad(make_pipeline(@"vs_quad", @"fs_tex_sampler", nil, nil), MTLClearColorMake(0, 0, 0, 1),
                  ^(id<MTLRenderCommandEncoder> e) {
                      [e setFragmentTexture:src atIndex:0];
                      [e setFragmentSamplerState:ss atIndex:0];
                  }, px);
    }
    return YES;
}

// Not a pass/fail case: clears textures with distinct, known properties so
// the host-side object dump (LAGFX_DUMP_OBJ=1) can be matched to fields.
static BOOL case_probe(void) {
    struct { NSUInteger w, h; MTLPixelFormat fmt; MTLStorageMode sm; MTLTextureUsage use; } t[] = {
        {48, 32, MTLPixelFormatRGBA8Unorm, MTLStorageModePrivate, MTLTextureUsageRenderTarget},
        {100, 20, MTLPixelFormatRGBA16Float, MTLStorageModeManaged,
         MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite},
        {256, 7, MTLPixelFormatBGRA8Unorm, MTLStorageModePrivate,
         MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead},
    };
    for (size_t i = 0; i < sizeof t / sizeof t[0]; i++) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:t[i].fmt width:t[i].w height:t[i].h mipmapped:NO];
        td.storageMode = t[i].sm;
        td.usage = t[i].use;
        id<MTLTexture> tex = [gDevice newTextureWithDescriptor:td];
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        [[cb renderCommandEncoderWithDescriptor:clear_pass(tex, MTLClearColorMake(0.25, 0.5, 0.75, 1))]
            endEncoding];
        run_cb(cb);
        printf("  probe %zu: %lux%lu fmt=%lu storage=%lu usage=%lu\n", i, (unsigned long)t[i].w,
               (unsigned long)t[i].h, (unsigned long)t[i].fmt, (unsigned long)t[i].sm,
               (unsigned long)t[i].use);
    }
    return YES;
}

// ------------------------------------------------------------------- main

static const struct { const char *name; BOOL (*fn)(void); BOOL in_all; } kCases[] = {
    {"clear", case_clear, YES},
    {"triangle", case_triangle, YES},
    {"triangle-vbuf", case_triangle_vbuf, YES},
    {"frag-bytes", case_frag_bytes, YES},
    {"frag-buffer", case_frag_buffer, YES},
    {"load", case_load, YES},
    {"blit-copy", case_blit_copy, YES},
    {"tex-nearest", case_tex_nearest, YES},
    {"tex-linear", case_tex_linear, YES},
    {"blend-alpha", case_blend_alpha, YES},
    {"blend-premul", case_blend_premul, YES},
    {"iosurface-sample", case_iosurface_sample, YES},
    {"iosurface-target", case_iosurface_target, YES},
    {"compute", case_compute, YES},
    {"probe", case_probe, NO},
    {"sampler-probe", case_sampler_probe, NO},
};
#define kNumCases (sizeof kCases / sizeof kCases[0])

static int find_case(const char *name) {
    for (size_t i = 0; i < kNumCases; i++)
        if (strcmp(kCases[i].name, name) == 0) return (int)i;
    return -1;
}

// Run one case in this process. Exit 0 = pass, 1 = fail, 2 = setup error.
static int run_one(const char *name) {
    int i = find_case(name);
    if (i < 0) { printf("unknown case %s\n", name); return 2; }
    step("start");
    gDevice = MTLCreateSystemDefaultDevice();
    if (!gDevice) { printf("  no Metal device\n"); return 2; }
    step("device");
    gQueue = [gDevice newCommandQueue];
    step("queue");
    NSError *err = nil;
    gLib = [gDevice newLibraryWithSource:kShaderSrc options:nil error:&err];
    if (!gLib) { printf("  shader compile: %s\n", err.localizedDescription.UTF8String); return 2; }
    step("library");
    gCase = name;
    return kCases[i].fn() ? 0 : 1;
}

// Spawn `self -1 name`, enforce the timeout, classify the outcome.
static const char *spawn_case(const char *self, const char *name) {
    pid_t pid;
    char *argv[] = {(char *)self, "-1", (char *)name, NULL};
    if (posix_spawn(&pid, self, NULL, NULL, argv, environ) != 0) return "SPAWN-ERROR";
    int status = 0;
    for (int t = 0; t < kCaseTimeoutSec * 10; t++) {
        pid_t r = waitpid(pid, &status, WNOHANG);
        if (r == pid) {
            if (WIFSIGNALED(status)) {
                printf("  killed by signal %d\n", WTERMSIG(status));
                return "CRASH";
            }
            switch (WEXITSTATUS(status)) {
            case 0: return "PASS";
            case 1: return "FAIL";
            default: return "ERROR";
            }
        }
        usleep(100000);
    }
    kill(pid, SIGKILL);
    waitpid(pid, &status, 0);
    printf("  no result after %d s, killed\n", kCaseTimeoutSec);
    return "TIMEOUT";
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        setvbuf(stdout, NULL, _IOLBF, 0);
        if (argc == 3 && strcmp(argv[1], "-1") == 0) return run_one(argv[2]);
        if (argc == 2 && strcmp(argv[1], "-l") == 0) {
            for (size_t i = 0; i < kNumCases; i++)
                printf("%s%s\n", kCases[i].name, kCases[i].in_all ? "" : " (only by name)");
            return 0;
        }

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        printf("device: %s\n", dev ? dev.name.UTF8String : "(none)");
        // Warm up: right after boot the first shader compile (compiler service
        // start-up) can take longer than a case's timeout. Compile the library
        // once here, untimed, so the cases measure the GPU, not the cold start.
        if (dev) {
            NSDate *t0 = [NSDate date];
            NSError *err = nil;
            id<MTLLibrary> lib = [dev newLibraryWithSource:kShaderSrc options:nil error:&err];
            MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
            pd.vertexFunction = [lib newFunctionWithName:@"vs_quad"];
            pd.fragmentFunction = [lib newFunctionWithName:@"fs_red"];
            pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
            if (lib) [dev newRenderPipelineStateWithDescriptor:pd error:&err];
            printf("warm-up: %.1f s%s\n", -t0.timeIntervalSinceNow, lib ? "" : " (shader compile failed)");
        }

        const char *self = argv[0];
        char resolved[PATH_MAX];
        if (realpath(argv[0], resolved)) self = resolved;

        const char *names[kNumCases];
        int n = 0;
        if (argc > 1) {
            for (int a = 1; a < argc; a++) {
                if (find_case(argv[a]) < 0) { printf("unknown case %s (try -l)\n", argv[a]); return 100; }
                if (n < (int)kNumCases) names[n++] = argv[a];
            }
        } else {
            for (size_t i = 0; i < kNumCases; i++)
                if (kCases[i].in_all) names[n++] = kCases[i].name;
        }

        const char *results[kNumCases];
        int passed = 0;
        for (int i = 0; i < n; i++) {
            printf("[%s]\n", names[i]);
            results[i] = spawn_case(self, names[i]);
            printf("  %s\n", results[i]);
            passed += strcmp(results[i], "PASS") == 0;
        }
        printf("--- summary\n");
        for (int i = 0; i < n; i++) printf("%-18s %s\n", names[i], results[i]);
        printf("%d/%d passed\n", passed, n);
        return n - passed;
    }
}
