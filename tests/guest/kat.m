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
     "[[patch(quad, 4)]] vertex float4 vs_patch(float2 uv [[position_in_patch]]) {\n"
     "    return float4(uv * 2 - 1, 0, 1);\n"
     "}\n"
     // positions indexed by vertex id and a matrix, both from buffers (as Core Animation draws)
     "vertex float4 vs_vid_mat(uint vid [[vertex_id]], const device float2 *pos [[buffer(1)]],\n"
     "                         constant float4x4 &m [[buffer(2)]]) {\n"
     "    return m * float4(pos[vid], 0, 1);\n"
     "}\n"
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
     // pixel coordinates through a sampler object (as WindowServer's final pass)
     "fragment float4 fs_tex_pixel(QOut in [[stage_in]], texture2d<float> t [[texture(0)]],\n"
     "                             sampler s [[sampler(0)]]) {\n"
     "    return t.sample(s, in.uv * float2(t.get_width(), t.get_height()));\n"
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
     // texel fetch at pixel position + offset: a 64x64 window into a large texture
     "fragment float4 fs_tex_read(float4 pos [[position]], texture2d<float> t [[texture(0)]],\n"
     "                            constant uint2 &off [[buffer(0)]]) {\n"
     "    return t.read(uint2(pos.xy) + off);\n"
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
// Vertex positions (pixels) and an orthographic matrix at large offsets in one
// shared buffer, bound at indices 1 and 2: the pattern of WindowServer's draws.
static BOOL case_vid_matrix(void) {
    id<MTLRenderPipelineState> ps = make_pipeline(@"vs_vid_mat", @"fs_red", nil, nil);
    if (!ps) return NO;
    enum { kPosOff = 0x18000, kMatOff = 0x19a00 };
    id<MTLBuffer> b = [gDevice newBufferWithLength:0x40000 options:MTLResourceStorageModeShared];
    uint8_t *base = b.contents;
    memset(base, 0xee, 0x40000);
    const float k = (float)kSize;
    float pos[6] = { 0, 0, k, 0, 0, k };
    memcpy(base + kPosOff, pos, sizeof pos);
    float m[16] = { 2 / k, 0, 0, 0,   0, 2 / k, 0, 0,   0, 0, 1, 0,   -1, -1, 0, 1 };  // columns
    memcpy(base + kMatOff, m, sizeof m);
    id<MTLTexture> t = make_target();
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLRenderCommandEncoder> enc =
        [cb renderCommandEncoderWithDescriptor:clear_pass(t, MTLClearColorMake(0, 0, 1, 1))];
    [enc setRenderPipelineState:ps];
    [enc setVertexBuffer:b offset:kPosOff atIndex:1];
    [enc setVertexBuffer:b offset:kMatOff atIndex:2];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
    static bgra_t px[kSize * kSize];
    if (!finish(cb, t, px)) return NO;
    return compare(px, ^bgra_t(int x, int y) { return in_tri(x, y) ? kRed : kBlue; }, tri_edge);
}

// Per-instance vertex data: positions per vertex (buffer 0), colours per
// instance (buffer 1: red, green). Two instances of the same triangle; the
// second (green) is drawn last.
static id<MTLRenderPipelineState> step_pipeline(MTLVertexStepFunction fn, NSUInteger rate,
                                                NSUInteger stride1) {
    MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
    vd.attributes[0].format = MTLVertexFormatFloat2;
    vd.attributes[0].offset = 0;
    vd.attributes[0].bufferIndex = 0;
    vd.attributes[1].format = MTLVertexFormatUChar4Normalized;
    vd.attributes[1].offset = 0;
    vd.attributes[1].bufferIndex = 1;
    vd.layouts[0].stride = 8;
    vd.layouts[1].stride = stride1;
    vd.layouts[1].stepFunction = fn;
    vd.layouts[1].stepRate = rate;
    return make_pipeline(@"vs_vbuf", @"fs_color", vd, nil);
}

static BOOL case_vbuf_instance(void) {
    id<MTLRenderPipelineState> ps = step_pipeline(MTLVertexStepFunctionPerInstance, 1, 4);
    if (!ps) return NO;
    const float pos[6] = { -1, -1, 1, -1, -1, 1 };
    const uint8_t col[8] = { 0, 0, 255, 255,   0, 255, 0, 255 };   // RGBA red, green
    id<MTLBuffer> vb = [gDevice newBufferWithBytes:pos length:sizeof pos
                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> cbuf = [gDevice newBufferWithBytes:col length:sizeof col
                                             options:MTLResourceStorageModeShared];
    id<MTLTexture> t = make_target();
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLRenderCommandEncoder> enc =
        [cb renderCommandEncoderWithDescriptor:clear_pass(t, MTLClearColorMake(0, 0, 0, 1))];
    [enc setRenderPipelineState:ps];
    [enc setVertexBuffer:vb offset:0 atIndex:0];
    [enc setVertexBuffer:cbuf offset:0 atIndex:1];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3 instanceCount:2];
    [enc endEncoding];
    static bgra_t px[kSize * kSize];
    if (!finish(cb, t, px)) return NO;
    return compare(px, ^bgra_t(int x, int y) { return in_tri(x, y) ? kGreen : kBlack; },
                   tri_edge);
}

// Name-only probe: layouts with step rate 3 (stride 0x24) and the constant
// step function (stride 0x2c), to decode the layout records (LAGFX_VTXLOG=1).
static BOOL case_vtxstep_probe(void) {
    id<MTLRenderPipelineState> a = step_pipeline(MTLVertexStepFunctionPerInstance, 3, 0x24);
    id<MTLRenderPipelineState> b = step_pipeline(MTLVertexStepFunctionConstant, 0, 0x2c);
    if (!a || !b) return NO;
    id<MTLBuffer> vb = [gDevice newBufferWithLength:0x400 options:MTLResourceStorageModeShared];
    id<MTLTexture> t = make_target();
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLRenderCommandEncoder> enc =
        [cb renderCommandEncoderWithDescriptor:clear_pass(t, MTLClearColorMake(0, 0, 0, 1))];
    [enc setVertexBuffer:vb offset:0 atIndex:0];
    [enc setVertexBuffer:vb offset:0 atIndex:1];
    [enc setRenderPipelineState:a];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3 instanceCount:2];
    [enc setRenderPipelineState:b];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3 instanceCount:2];
    [enc endEncoding];
    return run_cb(cb);
}

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

// Controlled wire-format probe for DrawIndexedPrimitives16.  The distinctive
// byte offset (0x220), six UInt16 indices and 0x1000-byte allocation let the
// host correlate every field without searching guest memory.
static BOOL case_indexed_probe(void) {
    id<MTLRenderPipelineState> ps = make_pipeline(@"vs_vid_mat", @"fs_red", nil, nil);
    if (!ps) return NO;
    const float k = (float)kSize;
    const float pos[8] = { 0, 0, k, 0, 0, k, k, k };
    const float m[16] = { 2 / k, 0, 0, 0,   0, 2 / k, 0, 0,
                          0, 0, 1, 0,       -1, -1, 0, 1 };
    id<MTLBuffer> vb = [gDevice newBufferWithBytes:pos length:sizeof pos
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> mb = [gDevice newBufferWithBytes:m length:sizeof m
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> ib = [gDevice newBufferWithLength:0x1000
                                           options:MTLResourceStorageModeShared];
    const uint16_t idx[6] = {0, 1, 2, 2, 1, 3};
    memcpy((uint8_t *)ib.contents + 0x220, idx, sizeof idx);
    id<MTLTexture> t = make_target();
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLRenderCommandEncoder> enc =
        [cb renderCommandEncoderWithDescriptor:clear_pass(t, MTLClearColorMake(0, 0, 0, 1))];
    [enc setRenderPipelineState:ps];
    [enc setVertexBuffer:vb offset:0 atIndex:1];
    [enc setVertexBuffer:mb offset:0 atIndex:2];
    [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:6
                    indexType:MTLIndexTypeUInt16 indexBuffer:ib indexBufferOffset:0x220];
    [enc endEncoding];
    static bgra_t px[kSize * kSize];
    if (!finish(cb, t, px)) return NO;
    return compare(px, ^bgra_t(int x, int y) { return kRed; }, nil);
}

// Name-only probe: an indexed draw that can't use the compact 16-bit form
// (UInt32 indices, offset above 64 KiB), with distinct field values to decode
// it (LAGFX_DUMP_DRAW=1): triangle strip (4), UInt32 (1), 5 indices, offset
// 0x12344. The fifth index repeats the fourth (a degenerate triangle).
static BOOL case_indexed32_probe(void) {
    id<MTLRenderPipelineState> ps = make_pipeline(@"vs_vid_mat", @"fs_red", nil, nil);
    if (!ps) return NO;
    const float k = (float)kSize;
    const float pos[8] = { 0, 0, k, 0, 0, k, k, k };
    const float m[16] = { 2 / k, 0, 0, 0,   0, 2 / k, 0, 0,
                          0, 0, 1, 0,       -1, -1, 0, 1 };
    id<MTLBuffer> vb = [gDevice newBufferWithBytes:pos length:sizeof pos
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> mb = [gDevice newBufferWithBytes:m length:sizeof m
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> ib = [gDevice newBufferWithLength:0x20000
                                           options:MTLResourceStorageModeShared];
    const uint32_t idx[5] = {0, 1, 2, 3, 3};
    memcpy((uint8_t *)ib.contents + 0x12344, idx, sizeof idx);
    id<MTLTexture> t = make_target();
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLRenderCommandEncoder> enc =
        [cb renderCommandEncoderWithDescriptor:clear_pass(t, MTLClearColorMake(0, 0, 0, 1))];
    [enc setRenderPipelineState:ps];
    [enc setVertexBuffer:vb offset:0 atIndex:1];
    [enc setVertexBuffer:mb offset:0 atIndex:2];
    [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangleStrip indexCount:5
                    indexType:MTLIndexTypeUInt32 indexBuffer:ib indexBufferOffset:0x12344];
    [enc endEncoding];
    static bgra_t px[kSize * kSize];
    if (!finish(cb, t, px)) return NO;
    return compare(px, ^bgra_t(int x, int y) { return kRed; }, nil);
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

// Name-only probe: instanced draws with distinct field values, to decode
// their commands (LAGFX_OPTASK=<kat task> on the host):
//   A: TriangleStrip, vertexStart 5, vertexCount 7, instanceCount 3
//   B: Triangle, vertexStart 6, vertexCount 9, instanceCount 4, baseInstance 2
//   C: Line, vertexStart 0x105, vertexCount 0x207, instanceCount 0x30a
//   D-F: see below
static BOOL case_draw_probe(void) {
    id<MTLRenderPipelineState> ps = make_pipeline(@"vs_quad", @"fs_red", nil, nil);
    if (!ps) return NO;
    id<MTLTexture> t = make_target();
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLRenderCommandEncoder> enc =
        [cb renderCommandEncoderWithDescriptor:clear_pass(t, MTLClearColorMake(0, 0, 0, 1))];
    [enc setRenderPipelineState:ps];
    [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:5 vertexCount:7 instanceCount:3];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:6 vertexCount:9 instanceCount:4
           baseInstance:2];
    [enc drawPrimitives:MTLPrimitiveTypeLine vertexStart:0x105 vertexCount:0x207
          instanceCount:0x30a];
    // D: start past 16 bits; E: count and base instance past 16 bits
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0x12345 vertexCount:6 instanceCount:3];
    [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:2 vertexCount:0x10004
          instanceCount:5 baseInstance:0x10007];
    // F: indexed with instances, base vertex and base instance
    id<MTLBuffer> ib = [gDevice newBufferWithLength:0x1000 options:MTLResourceStorageModeShared];
    [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:6 indexType:MTLIndexTypeUInt16
                   indexBuffer:ib indexBufferOffset:0x40 instanceCount:3 baseVertex:5
                  baseInstance:7];
    // I: plain (no instance count); J: instance count 1
    [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0x13 vertexCount:0x3f];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:7 vertexCount:5 instanceCount:1];
    // G: indirect (arguments in a buffer at 0x30); H: indexed indirect (args at 0x50)
    id<MTLBuffer> args = [gDevice newBufferWithLength:0x100 options:MTLResourceStorageModeShared];
    [enc drawPrimitives:MTLPrimitiveTypeLineStrip indirectBuffer:args indirectBufferOffset:0x30];
    [enc drawIndexedPrimitives:MTLPrimitiveTypePoint indexType:MTLIndexTypeUInt32 indexBuffer:ib
             indexBufferOffset:0x44 indirectBuffer:args indirectBufferOffset:0x50];
    [enc endEncoding];
    return run_cb(cb);
}

// Tessellation: a quad-patch pipeline whose vertex function maps the patch's
// domain onto the whole target, and a factor buffer of 1.0s (half floats).
static id<MTLRenderPipelineState> patch_pipeline(void) {
    NSError *err = nil;
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = [gLib newFunctionWithName:@"vs_patch"];
    pd.fragmentFunction = [gLib newFunctionWithName:@"fs_red"];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    pd.maxTessellationFactor = 16;
    pd.tessellationFactorStepFunction = MTLTessellationFactorStepFunctionConstant;
    pd.tessellationPartitionMode = MTLTessellationPartitionModeInteger;
    pd.tessellationOutputWindingOrder = MTLWindingCounterClockwise;
    id<MTLRenderPipelineState> ps = [gDevice newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!ps) printf("  pipeline: %s\n", err.localizedDescription.UTF8String);
    return ps;
}

static id<MTLBuffer> patch_factors(NSUInteger patches) {
    id<MTLBuffer> f = [gDevice newBufferWithLength:12 * patches + 16
                                           options:MTLResourceStorageModeShared];
    uint16_t *h = (uint16_t *)f.contents;
    for (NSUInteger i = 0; i < 6 * patches; i++) h[i] = 0x3c00;   // 1.0 (edge x4, inside x2)
    return f;
}

// One quad patch with tessellation factors of 1 covers the target.
static BOOL case_patch(void) {
    id<MTLRenderPipelineState> ps = patch_pipeline();
    if (!ps) return NO;
    id<MTLTexture> t = make_target();
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLRenderCommandEncoder> enc =
        [cb renderCommandEncoderWithDescriptor:clear_pass(t, MTLClearColorMake(0, 0, 0, 1))];
    [enc setRenderPipelineState:ps];
    [enc setTessellationFactorBuffer:patch_factors(1) offset:0 instanceStride:0];
    [enc drawPatches:4 patchStart:0 patchCount:1 patchIndexBuffer:nil patchIndexBufferOffset:0
       instanceCount:1 baseInstance:0];
    [enc endEncoding];
    static bgra_t px[kSize * kSize];
    if (!finish(cb, t, px)) return NO;
    return compare(px, ^bgra_t(int x, int y) { return kRed; }, nil);
}

// Name-only probe: patch draws with distinct values (LAGFX_OPTASK=* on the host).
//   P: 4 control points, patchStart 5, patchCount 7, instances 3, baseInstance 2
//   Q: patch index buffer at offset 0x40, patchStart 0x11, patchCount 0x22
//   R: patchStart 0x10005 (past 16 bits), patchCount 9, instances 0x10003
static BOOL case_patch_probe(void) {
    id<MTLRenderPipelineState> ps = patch_pipeline();
    if (!ps) return NO;
    id<MTLTexture> t = make_target();
    id<MTLBuffer> pib = [gDevice newBufferWithLength:0x1000 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLRenderCommandEncoder> enc =
        [cb renderCommandEncoderWithDescriptor:clear_pass(t, MTLClearColorMake(0, 0, 0, 1))];
    [enc setRenderPipelineState:ps];
    [enc setTessellationFactorBuffer:patch_factors(0x10010) offset:0x60 instanceStride:0x24];
    [enc drawPatches:4 patchStart:5 patchCount:7 patchIndexBuffer:nil patchIndexBufferOffset:0
       instanceCount:3 baseInstance:2];
    [enc drawPatches:4 patchStart:0x11 patchCount:0x22 patchIndexBuffer:pib
        patchIndexBufferOffset:0x40 instanceCount:1 baseInstance:0];
    [enc drawPatches:4 patchStart:0x10005 patchCount:9 patchIndexBuffer:nil
        patchIndexBufferOffset:0 instanceCount:0x10003 baseInstance:0];
    [enc endEncoding];
    return run_cb(cb);
}

// Name-only probe: draws through a Metal 4 encoder with distinct values
// (LAGFX_OPTASK=* on the host):
//   M: TriangleStrip, vertexStart 5, vertexCount 7, instanceCount 3, baseInstance 2
//   N: Triangle, vertexStart 0x11, vertexCount 0x3f, instanceCount 1
//   O: TriangleStrip, vertexStart 0, vertexCount 4, instanceCount 1
static BOOL case_mtl4_probe(void) {
    if (@available(macOS 26.0, *)) {
        id<MTL4CommandQueue> q = [gDevice newMTL4CommandQueue];
        id<MTL4CommandAllocator> alloc = [gDevice newCommandAllocator];
        id<MTL4CommandBuffer> cb = [gDevice newCommandBuffer];
        if (!q || !alloc || !cb) { printf("  no Metal 4 on this device\n"); return NO; }
        id<MTLRenderPipelineState> ps = make_pipeline(@"vs_quad", @"fs_red", nil, nil);
        id<MTLTexture> t = make_target();
        MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
        NSError *err = nil;
        id<MTLResidencySet> rs = [gDevice newResidencySetWithDescriptor:rd error:&err];
        [rs addAllocation:t];
        [rs commit];
        [q addResidencySet:rs];
        [cb beginCommandBufferWithAllocator:alloc];
        MTL4RenderPassDescriptor *rp = [MTL4RenderPassDescriptor new];
        rp.colorAttachments[0].texture = t;
        rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
        id<MTL4RenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
        [enc setRenderPipelineState:ps];
        [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:5 vertexCount:7
              instanceCount:3 baseInstance:2];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0x11 vertexCount:0x3f
              instanceCount:1];
        [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4
              instanceCount:1];
        [enc endEncoding];
        [cb endCommandBuffer];
        id<MTLSharedEvent> ev = [gDevice newSharedEvent];
        const id<MTL4CommandBuffer> cbs[1] = { cb };
        [q commit:cbs count:1];
        [q signalEvent:ev value:1];
        BOOL done = [ev waitUntilSignaledValue:1 timeoutMS:10000];
        if (!done) printf("  Metal 4 command buffer did not complete\n");
        return done;
    }
    return NO;
}

// Stencil-then-cover (how the lock screen clock fills its paths): a triangle
// with colour writes off marks the stencil (Always / Replace, reference 1),
// then a full quad tested Equal 1 colours only the marked pixels.
static id<MTLRenderPipelineState> stencil_pipeline(NSString *vs, BOOL color) {
    NSError *err = nil;
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = [gLib newFunctionWithName:vs];
    pd.fragmentFunction = [gLib newFunctionWithName:@"fs_red"];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    if (!color) pd.colorAttachments[0].writeMask = MTLColorWriteMaskNone;
    pd.stencilAttachmentPixelFormat = MTLPixelFormatStencil8;
    id<MTLRenderPipelineState> ps = [gDevice newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!ps) printf("  pipeline: %s\n", err.localizedDescription.UTF8String);
    return ps;
}

static BOOL case_stencil(void) {
    id<MTLRenderPipelineState> mark = stencil_pipeline(@"vs_tri", NO);
    id<MTLRenderPipelineState> cover = stencil_pipeline(@"vs_quad", YES);
    if (!mark || !cover) return NO;
    MTLTextureDescriptor *sd = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatStencil8 width:kSize height:kSize
                                 mipmapped:NO];
    sd.usage = MTLTextureUsageRenderTarget;
    sd.storageMode = MTLStorageModePrivate;
    id<MTLTexture> st = [gDevice newTextureWithDescriptor:sd];
    MTLStencilDescriptor *w = [MTLStencilDescriptor new];
    w.stencilCompareFunction = MTLCompareFunctionAlways;
    w.depthStencilPassOperation = MTLStencilOperationReplace;
    MTLDepthStencilDescriptor *wd = [MTLDepthStencilDescriptor new];
    wd.frontFaceStencil = w; wd.backFaceStencil = w;
    MTLStencilDescriptor *e = [MTLStencilDescriptor new];
    e.stencilCompareFunction = MTLCompareFunctionEqual;
    MTLDepthStencilDescriptor *ed = [MTLDepthStencilDescriptor new];
    ed.frontFaceStencil = e; ed.backFaceStencil = e;
    id<MTLTexture> t = make_target();
    MTLRenderPassDescriptor *rp = clear_pass(t, MTLClearColorMake(0, 0, 0, 1));
    rp.stencilAttachment.texture = st;
    rp.stencilAttachment.loadAction = MTLLoadActionClear;
    rp.stencilAttachment.storeAction = MTLStoreActionDontCare;
    rp.stencilAttachment.clearStencil = 0;
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
    [enc setStencilReferenceValue:1];
    [enc setRenderPipelineState:mark];
    [enc setDepthStencilState:[gDevice newDepthStencilStateWithDescriptor:wd]];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc setRenderPipelineState:cover];
    [enc setDepthStencilState:[gDevice newDepthStencilStateWithDescriptor:ed]];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    [enc endEncoding];
    static bgra_t px[kSize * kSize];
    if (!finish(cb, t, px)) return NO;
    return compare(px, ^bgra_t(int x, int y) { return in_tri(x, y) ? kRed : kBlack; }, tri_edge);
}

// Name-only probe: a render pass with a stencil attachment and a depth-stencil
// state with distinct settings (LAGFX_OPTASK=* and LAGFX_DUMP_OBJ=1 on the host):
//   stencil texture 64x64 Stencil8, clear stencil 0x5a, load Clear / store Store
//   front: compare Equal(3), fail Replace(2), depthFail Invert(5),
//          pass IncrementWrap(6), read mask 0x3c, write mask 0xc3
//   back: compare Greater(4), fail Zero(1), depthFail DecrementClamp(4),
//         pass Keep(0), read mask 0x0f, write mask 0xf0
//   references: front 0x12, back 0x34
static BOOL case_stencil_probe(void) {
    id<MTLRenderPipelineState> ps = nil;
    {
        NSError *err = nil;
        MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
        pd.vertexFunction = [gLib newFunctionWithName:@"vs_quad"];
        pd.fragmentFunction = [gLib newFunctionWithName:@"fs_red"];
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        pd.stencilAttachmentPixelFormat = MTLPixelFormatStencil8;
        ps = [gDevice newRenderPipelineStateWithDescriptor:pd error:&err];
        if (!ps) { printf("  pipeline: %s\n", err.localizedDescription.UTF8String); return NO; }
    }
    MTLTextureDescriptor *sd = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatStencil8 width:kSize height:kSize
                                 mipmapped:NO];
    sd.usage = MTLTextureUsageRenderTarget;
    sd.storageMode = MTLStorageModePrivate;
    id<MTLTexture> st = [gDevice newTextureWithDescriptor:sd];
    MTLStencilDescriptor *f = [MTLStencilDescriptor new];
    f.stencilCompareFunction = MTLCompareFunctionEqual;
    f.stencilFailureOperation = MTLStencilOperationReplace;
    f.depthFailureOperation = MTLStencilOperationInvert;
    f.depthStencilPassOperation = MTLStencilOperationIncrementWrap;
    f.readMask = 0x3c; f.writeMask = 0xc3;
    MTLStencilDescriptor *b = [MTLStencilDescriptor new];
    b.stencilCompareFunction = MTLCompareFunctionGreater;
    b.stencilFailureOperation = MTLStencilOperationZero;
    b.depthFailureOperation = MTLStencilOperationDecrementClamp;
    b.depthStencilPassOperation = MTLStencilOperationKeep;
    b.readMask = 0x0f; b.writeMask = 0xf0;
    MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new];
    dd.frontFaceStencil = f; dd.backFaceStencil = b;
    id<MTLDepthStencilState> ds = [gDevice newDepthStencilStateWithDescriptor:dd];
    id<MTLTexture> t = make_target();
    MTLRenderPassDescriptor *rp = clear_pass(t, MTLClearColorMake(0, 0, 0, 1));
    rp.stencilAttachment.texture = st;
    rp.stencilAttachment.loadAction = MTLLoadActionClear;
    rp.stencilAttachment.storeAction = MTLStoreActionStore;
    rp.stencilAttachment.clearStencil = 0x5a;
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
    [enc setRenderPipelineState:ps];
    [enc setDepthStencilState:ds];
    [enc setStencilFrontReferenceValue:0x12 backReferenceValue:0x34];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    [enc endEncoding];
    return run_cb(cb);
}

// UInt32 indices at a small offset: {0,1,2,2,1,3} at byte 0x200 of the index
// buffer draw the full quad (the clock renderer's op 0x09 form).
static BOOL case_indexed32_small(void) {
    id<MTLRenderPipelineState> ps = make_pipeline(@"vs_vid_mat", @"fs_red", nil, nil);
    if (!ps) return NO;
    const float k = (float)kSize;
    const float pos[8] = { 0, 0, k, 0, 0, k, k, k };
    const float m[16] = { 2 / k, 0, 0, 0,   0, 2 / k, 0, 0,
                          0, 0, 1, 0,       -1, -1, 0, 1 };
    id<MTLBuffer> vb = [gDevice newBufferWithBytes:pos length:sizeof pos
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> mb = [gDevice newBufferWithBytes:m length:sizeof m
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> ib = [gDevice newBufferWithLength:0x1000 options:MTLResourceStorageModeShared];
    const uint32_t idx[6] = {0, 1, 2, 2, 1, 3};
    memcpy((uint8_t *)ib.contents + 0x200, idx, sizeof idx);
    id<MTLTexture> t = make_target();
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLRenderCommandEncoder> enc =
        [cb renderCommandEncoderWithDescriptor:clear_pass(t, MTLClearColorMake(0, 0, 0, 1))];
    [enc setRenderPipelineState:ps];
    [enc setVertexBuffer:vb offset:0 atIndex:1];
    [enc setVertexBuffer:mb offset:0 atIndex:2];
    [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:6
                    indexType:MTLIndexTypeUInt32 indexBuffer:ib indexBufferOffset:0x200];
    [enc endEncoding];
    static bgra_t px[kSize * kSize];
    if (!finish(cb, t, px)) return NO;
    return compare(px, ^bgra_t(int x, int y) { return kRed; }, nil);
}

// Name-only probe: one draw (TriangleStrip, start 5, count 7, instances 3,
// base instance 2) through pipelines with different properties, to find what
// makes the driver encode a draw as op 0x09 (LAGFX_OPTASK=* on the host):
//   1 plain, 2 supportIndirectCommandBuffers, 3 inputPrimitiveTopology Triangle,
//   4 rasterSampleCount 4 (into a 4x target), 5 maxVertexAmplificationCount 2
static BOOL case_pso_probe(void) {
    id<MTLTexture> t = make_target();
    MTLTextureDescriptor *md = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:kSize height:kSize
                                 mipmapped:NO];
    md.textureType = MTLTextureType2DMultisample;
    md.sampleCount = 4;
    md.usage = MTLTextureUsageRenderTarget;
    md.storageMode = MTLStorageModePrivate;
    id<MTLTexture> ms = [gDevice newTextureWithDescriptor:md];
    for (int k = 1; k <= 5; k++) {
        NSError *err = nil;
        MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
        pd.vertexFunction = [gLib newFunctionWithName:@"vs_quad"];
        pd.fragmentFunction = [gLib newFunctionWithName:@"fs_red"];
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        if (k == 2) pd.supportIndirectCommandBuffers = YES;
        if (k == 3) pd.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
        if (k == 4) pd.rasterSampleCount = 4;
        if (k == 5) pd.maxVertexAmplificationCount = 2;
        id<MTLRenderPipelineState> ps = [gDevice newRenderPipelineStateWithDescriptor:pd error:&err];
        if (!ps) { printf("  pipeline %d: %s\n", k, err.localizedDescription.UTF8String); continue; }
        MTLRenderPassDescriptor *rp = clear_pass(k == 4 ? ms : t, MTLClearColorMake(0, 0, 0, 1));
        if (k == 4) rp.colorAttachments[0].storeAction = MTLStoreActionDontCare;
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
        [enc setRenderPipelineState:ps];
        if (k == 5) {
            MTLVertexAmplificationViewMapping map[2] = { {0, 0}, {0, 0} };
            [enc setVertexAmplificationCount:2 viewMappings:map];
        }
        [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:5 vertexCount:7
              instanceCount:3 baseInstance:2];
        [enc endEncoding];
        if (!run_cb(cb)) printf("  pipeline %d: command buffer failed\n", k);
    }
    return YES;
}

// A texture made from a buffer at an offset that isn't page-aligned (a linear
// texture: newTextureWithDescriptor:offset:bytesPerRow:), with a row pitch
// wider than the image. Prints the offset and pitch for decoding the object.
static BOOL case_tex_buffer(void) {
    NSUInteger align = [gDevice minimumLinearTextureAlignmentForPixelFormat:MTLPixelFormatBGRA8Unorm];
    NSUInteger off = 3 * align, bpr = ((kPat * 4 + 2 * align - 1) / align) * align;
    printf("  alignment %lu, offset 0x%lx, bytes per row 0x%lx\n", (unsigned long)align,
           (unsigned long)off, (unsigned long)bpr);
    id<MTLBuffer> b = [gDevice newBufferWithLength:0x10000 options:MTLResourceStorageModeShared];
    memset(b.contents, 0x33, 0x10000);
    fill_pattern((uint8_t *)b.contents + off, bpr);
    MTLTextureDescriptor *td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:kPat height:kPat
                                 mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> src = [b newTextureWithDescriptor:td offset:off bytesPerRow:bpr];
    if (!src) return NO;
    static bgra_t px[kSize * kSize];
    if (!draw_quad(make_pipeline(@"vs_quad", @"fs_tex_nearest", nil, nil),
                   MTLClearColorMake(0, 0, 0, 1),
                   ^(id<MTLRenderCommandEncoder> e) { [e setFragmentTexture:src atIndex:0]; }, px))
        return NO;
    return compare(px, pattern_blocks, nil);
}

// Pixel coordinates (normalizedCoordinates = NO, ClampToZero) through a sampler
// state object, as WindowServer's final full-screen pass samples its composite.
static BOOL case_tex_pixel(void) {
    id<MTLTexture> src = make_texture(kPat, kPat, MTLTextureUsageShaderRead);
    static uint8_t texels[kPat * kPat * 4];
    fill_pattern(texels, kPat * 4);
    [src replaceRegion:MTLRegionMake2D(0, 0, kPat, kPat) mipmapLevel:0 withBytes:texels
           bytesPerRow:kPat * 4];
    MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
    sd.normalizedCoordinates = NO;
    sd.sAddressMode = MTLSamplerAddressModeClampToZero;
    sd.tAddressMode = MTLSamplerAddressModeClampToZero;
    id<MTLSamplerState> ss = [gDevice newSamplerStateWithDescriptor:sd];
    static bgra_t px[kSize * kSize];
    if (!draw_quad(make_pipeline(@"vs_quad", @"fs_tex_pixel", nil, nil),
                   MTLClearColorMake(0, 0, 0, 1),
                   ^(id<MTLRenderCommandEncoder> e) {
                       [e setFragmentTexture:src atIndex:0];
                       [e setFragmentSamplerState:ss atIndex:0];
                   },
                   px))
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

// Large, odd-sized textures (as window and text surfaces are), read through a
// 64x64 window at an offset. Every texel encodes its coordinates, so a wrong
// row pitch shows as a shear and a wrong start offset as a constant shift.
#define kBigW 298
#define kBigH 113
static const uint32_t kBigOff[2] = {201, 37};
static bgra_t big_texel(int x, int y) {
    return px_rgba(x & 0xff, y & 0xff, 0x80 | (x >> 8) | ((y >> 8) << 2), 255);
}
static void fill_big(uint8_t *base, size_t bpr) {
    for (int y = 0; y < kBigH; y++)
        for (int x = 0; x < kBigW; x++) *(bgra_t *)(base + y * bpr + x * 4) = big_texel(x, y);
}
static BOOL check_big(id<MTLTexture> src) {
    if (!src) return NO;
    static bgra_t px[kSize * kSize];
    if (!draw_quad(make_pipeline(@"vs_quad", @"fs_tex_read", nil, nil), MTLClearColorMake(0, 0, 0, 1),
                   ^(id<MTLRenderCommandEncoder> e) {
                       [e setFragmentTexture:src atIndex:0];
                       [e setFragmentBytes:kBigOff length:sizeof kBigOff atIndex:0];
                   },
                   px))
        return NO;
    // Report where the first row's samples actually came from.
    bgra_t p = px[0];
    printf("  pixel (0,0) holds texel (%d,%d), want (%u,%u)\n",
           p.r | ((p.b & 3) << 8), p.g | (((p.b >> 2) & 3) << 8), kBigOff[0], kBigOff[1]);
    return compare(px, ^bgra_t(int x, int y) {
        int tx = x + (int)kBigOff[0], ty = y + (int)kBigOff[1];
        return tx < kBigW && ty < kBigH ? big_texel(tx, ty) : px_rgba(0, 0, 0, 0);
    }, ^BOOL(int x, int y) { return x + (int)kBigOff[0] >= kBigW || y + (int)kBigOff[1] >= kBigH; });
}

static BOOL case_tex_big(void) {
    id<MTLTexture> src = make_texture(kBigW, kBigH, MTLTextureUsageShaderRead);
    static uint8_t texels[kBigW * kBigH * 4];
    fill_big(texels, kBigW * 4);
    [src replaceRegion:MTLRegionMake2D(0, 0, kBigW, kBigH) mipmapLevel:0 withBytes:texels
           bytesPerRow:kBigW * 4];
    return check_big(src);
}

static BOOL case_iosurface_big(void) {
    IOSurfaceRef s = make_surface(kBigW, kBigH);
    if (!s) { printf("  IOSurfaceCreate failed\n"); return NO; }
    printf("  bytes per row %zu (width x 4 = %d), alloc %zu\n", IOSurfaceGetBytesPerRow(s),
           kBigW * 4, IOSurfaceGetAllocSize(s));
    IOSurfaceLock(s, 0, NULL);
    fill_big(IOSurfaceGetBaseAddress(s), IOSurfaceGetBytesPerRow(s));
    IOSurfaceUnlock(s, 0, NULL);
    BOOL ok = check_big(surface_texture(s, kBigW, kBigH, MTLTextureUsageShaderRead));
    CFRelease(s);
    return ok;
}

static BOOL case_tex_bufbig(void) {
    NSUInteger align = [gDevice minimumLinearTextureAlignmentForPixelFormat:MTLPixelFormatBGRA8Unorm];
    NSUInteger off = 5 * align, bpr = ((kBigW * 4 + 3 * align - 1) / align) * align;
    printf("  offset 0x%lx, bytes per row 0x%lx\n", (unsigned long)off, (unsigned long)bpr);
    NSUInteger len = off + bpr * kBigH;
    BOOL ok = YES;
    const MTLStorageMode modes[2] = {MTLStorageModeShared, MTLStorageModeManaged};
    for (int m = 0; m < 2; m++) {
        id<MTLBuffer> b = [gDevice newBufferWithLength:len
                                               options:(MTLResourceOptions)modes[m]
                                                       << MTLResourceStorageModeShift];
        memset(b.contents, 0x33, len);
        fill_big((uint8_t *)b.contents + off, bpr);
        if (modes[m] == MTLStorageModeManaged) [b didModifyRange:NSMakeRange(0, len)];
        MTLTextureDescriptor *td = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:kBigW height:kBigH
                                     mipmapped:NO];
        td.usage = MTLTextureUsageShaderRead;
        td.storageMode = modes[m];
        printf("  storage mode %lu:\n", (unsigned long)modes[m]);
        ok &= check_big([b newTextureWithDescriptor:td offset:off bytesPerRow:bpr]);
    }
    return ok;
}

// A managed texture updated with replaceRegion after it was first drawn with
// (as Core Animation updates the password-dot mask): the second draw must
// see the new contents.
static BOOL case_tex_update(void) {
    id<MTLTexture> src = make_texture(kPat, kPat, MTLTextureUsageShaderRead);
    static uint8_t texels[kPat * kPat * 4];
    memset(texels, 0, sizeof texels);
    [src replaceRegion:MTLRegionMake2D(0, 0, kPat, kPat) mipmapLevel:0 withBytes:texels
           bytesPerRow:kPat * 4];
    id<MTLRenderPipelineState> ps = make_pipeline(@"vs_quad", @"fs_tex_nearest", nil, nil);
    static bgra_t px[kSize * kSize];
    if (!draw_quad(ps, MTLClearColorMake(0, 0, 0, 1),
                   ^(id<MTLRenderCommandEncoder> e) { [e setFragmentTexture:src atIndex:0]; }, px))
        return NO;
    printf("  first draw (zeros): pixel 0 = %02x%02x%02x%02x\n", px[0].r, px[0].g, px[0].b, px[0].a);
    fill_pattern(texels, kPat * 4);
    [src replaceRegion:MTLRegionMake2D(0, 0, kPat, kPat) mipmapLevel:0 withBytes:texels
           bytesPerRow:kPat * 4];
    if (!draw_quad(ps, MTLClearColorMake(0, 0, 0, 1),
                   ^(id<MTLRenderCommandEncoder> e) { [e setFragmentTexture:src atIndex:0]; }, px))
        return NO;
    return compare(px, pattern_blocks, nil);
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

// Not a pass/fail case: pipelines with known blend settings, so the
// host-side pipeline dump (LAGFX_DUMP_PSO=1) can be matched to them.
static BOOL case_blend_probe(void) {
    struct { const char *what; blend_cfg cfg; } p[] = {
        {"blending off", nil},
        {"enabled, defaults", ^(MTLRenderPipelineColorAttachmentDescriptor *a) { a.blendingEnabled = YES; }},
        {"rgb src SourceAlpha(4) dst OneMinusSourceAlpha(5)", ^(MTLRenderPipelineColorAttachmentDescriptor *a) {
            a.blendingEnabled = YES;
            a.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
            a.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha; }},
        {"alpha src DestinationAlpha(8) dst OneMinusDestinationColor(7)", ^(MTLRenderPipelineColorAttachmentDescriptor *a) {
            a.blendingEnabled = YES;
            a.sourceAlphaBlendFactor = MTLBlendFactorDestinationAlpha;
            a.destinationAlphaBlendFactor = MTLBlendFactorOneMinusDestinationColor; }},
        {"rgb op ReverseSubtract(2)", ^(MTLRenderPipelineColorAttachmentDescriptor *a) {
            a.blendingEnabled = YES; a.rgbBlendOperation = MTLBlendOperationReverseSubtract; }},
        {"alpha op Max(4)", ^(MTLRenderPipelineColorAttachmentDescriptor *a) {
            a.blendingEnabled = YES; a.alphaBlendOperation = MTLBlendOperationMax; }},
        {"write mask Red|Alpha (9)", ^(MTLRenderPipelineColorAttachmentDescriptor *a) {
            a.writeMask = MTLColorWriteMaskRed | MTLColorWriteMaskAlpha; }},
        {"rgb src BlendColor(11) dst OneMinusBlendAlpha(14)", ^(MTLRenderPipelineColorAttachmentDescriptor *a) {
            a.blendingEnabled = YES;
            a.sourceRGBBlendFactor = MTLBlendFactorBlendColor;
            a.destinationRGBBlendFactor = MTLBlendFactorOneMinusBlendAlpha; }},
    };
    static bgra_t px[kSize * kSize];
    static const float c[4] = {1, 0, 0, 0.5f};
    for (size_t i = 0; i < sizeof p / sizeof p[0]; i++) {
        printf("  pipeline %zu: %s\n", i, p[i].what);
        draw_quad(make_pipeline(@"vs_quad", @"fs_const", nil, p[i].cfg), MTLClearColorMake(0, 0, 1, 1),
                  ^(id<MTLRenderCommandEncoder> e) { [e setFragmentBytes:c length:sizeof c atIndex:0]; },
                  px);
    }
    return YES;
}

// Not a pass/fail case: clears textures with distinct, known properties so
// the host-side object dump (LAGFX_DUMP_OBJ=1) can be matched to fields.
// Name-only probe for decoding IOSurface textures on the host: surfaces of
// distinct sizes, each page starting with "LAGFXIOS" (run the host with
// LAGFX_FIND_SIG=LAGFXIOS LAGFX_DUMP_IOS=1 LAGFX_DUMP_OBJ=1), sampled once.
static BOOL case_iosurface_probe(void) {
    id<MTLRenderPipelineState> ps = make_pipeline(@"vs_quad", @"fs_tex_nearest", nil, nil);
    if (!ps) return NO;
    static const struct { int w, h, bpr; } kSurf[] = { {40, 24, 0}, {100, 20, 512}, {64, 64, 0} };
    for (size_t i = 0; i < sizeof kSurf / sizeof kSurf[0]; i++) {
        NSMutableDictionary *props = [@{
            (id)kIOSurfaceWidth: @(kSurf[i].w), (id)kIOSurfaceHeight: @(kSurf[i].h),
            (id)kIOSurfaceBytesPerElement: @4, (id)kIOSurfacePixelFormat: @((uint32_t)'BGRA'),
        } mutableCopy];
        if (kSurf[i].bpr) props[(id)kIOSurfaceBytesPerRow] = @(kSurf[i].bpr);
        IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)props);
        if (!s) { printf("  IOSurfaceCreate failed\n"); return NO; }
        IOSurfaceLock(s, 0, NULL);
        uint8_t *base = IOSurfaceGetBaseAddress(s);
        size_t bpr = IOSurfaceGetBytesPerRow(s), alloc = IOSurfaceGetAllocSize(s);
        for (size_t k = 0; k < alloc; k++) base[k] = (uint8_t)(0x10 * (i + 1) + (k >> 12));
        for (size_t pg = 0; pg < alloc; pg += 4096) memcpy(base + pg, "LAGFXIOS", 8);
        IOSurfaceUnlock(s, 0, NULL);
        printf("  surface %zu: id=%u %dx%d bpr=%zu alloc=0x%zx\n", i, IOSurfaceGetID(s),
               kSurf[i].w, kSurf[i].h, bpr, alloc);
        id<MTLTexture> t = surface_texture(s, kSurf[i].w, kSurf[i].h, MTLTextureUsageShaderRead);
        static bgra_t px[kSize * kSize];
        draw_quad(ps, MTLClearColorMake(0, 0, 0, 1),
                  ^(id<MTLRenderCommandEncoder> e) { [e setFragmentTexture:t atIndex:0]; }, px);
        CFRelease(s);
    }
    return YES;
}

// Name-only probe: one texture of each MTLTextureType (distinct sizes), each
// bound as fragment texture 0 of a draw, so the host can dump their objects
// (LAGFX_DUMP_OBJ=1) and locate the texture-type field.
static BOOL case_type_probe(void) {
    id<MTLRenderPipelineState> ps = make_pipeline(@"vs_tri", @"fs_red", nil, nil);
    if (!ps) return NO;
    static const struct { MTLTextureType type; NSUInteger w, h, d, layers; } kT[] = {
        { MTLTextureType1D, 33, 1, 1, 1 },        { MTLTextureType1DArray, 35, 1, 1, 3 },
        { MTLTextureType2D, 37, 5, 1, 1 },        { MTLTextureType2DArray, 39, 7, 1, 4 },
        { MTLTextureTypeCube, 16, 16, 1, 1 },     { MTLTextureType3D, 41, 9, 6, 1 },
    };
    NSMutableArray *keep = [NSMutableArray array];   /* distinct object ids */
    for (size_t i = 0; i < sizeof kT / sizeof kT[0]; i++) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor new];
        td.textureType = kT[i].type;
        td.pixelFormat = MTLPixelFormatRGBA8Unorm;
        td.width = kT[i].w; td.height = kT[i].h; td.depth = kT[i].d;
        td.arrayLength = kT[i].layers;
        td.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> t = [gDevice newTextureWithDescriptor:td];
        [keep addObject:t];
        static bgra_t px[kSize * kSize];
        draw_quad(ps, MTLClearColorMake(0, 0, 0, 1),
                  ^(id<MTLRenderCommandEncoder> e) { [e setFragmentTexture:t atIndex:0]; }, px);
        printf("  type %lu: %lux%lux%lu layers %lu\n", (unsigned long)kT[i].type,
               (unsigned long)kT[i].w, (unsigned long)kT[i].h, (unsigned long)kT[i].d,
               (unsigned long)kT[i].layers);
    }
    return YES;
}

// Name-only probe: blit copies with distinct non-zero fields, to decode the
// copy commands (LAGFX_DUMP_BLIT=1 on the host).
static BOOL case_blit_probe(void) {
    id<MTLBuffer> b = [gDevice newBufferWithLength:0x10000 options:MTLResourceStorageModeShared];
    memset(b.contents, 0x5a, 0x10000);
    MTLTextureDescriptor *td = [MTLTextureDescriptor new];
    td.textureType = MTLTextureType2DArray;
    td.pixelFormat = MTLPixelFormatRGBA8Unorm;
    td.width = 64; td.height = 32; td.arrayLength = 4;
    td.mipmapLevelCount = 3;
    td.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> t = [gDevice newTextureWithDescriptor:td];
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    // offset 16, 512 B/row, 0x1400 B/image, 20x10, slice 2, level 1, origin (3,5)
    [blit copyFromBuffer:b sourceOffset:16 sourceBytesPerRow:512 sourceBytesPerImage:0x1400
              sourceSize:MTLSizeMake(20, 10, 1) toTexture:t destinationSlice:2 destinationLevel:1
       destinationOrigin:MTLOriginMake(3, 5, 0)];
    [blit endEncoding];
    return run_cb(cb);
}

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
    {"indexed-probe", case_indexed_probe, NO},
    {"indexed32-probe", case_indexed32_probe, NO},
    {"vid-matrix", case_vid_matrix, YES},
    {"frag-bytes", case_frag_bytes, YES},
    {"frag-buffer", case_frag_buffer, YES},
    {"load", case_load, YES},
    {"blit-copy", case_blit_copy, YES},
    {"tex-nearest", case_tex_nearest, YES},
    {"tex-buffer", case_tex_buffer, YES},
    {"patch", case_patch, YES},
    {"stencil", case_stencil, YES},
    {"vbuf-instance", case_vbuf_instance, YES},
    {"indexed32-small", case_indexed32_small, YES},
    {"tex-linear", case_tex_linear, YES},
    {"tex-pixel", case_tex_pixel, YES},
    {"blend-alpha", case_blend_alpha, YES},
    {"blend-premul", case_blend_premul, YES},
    {"iosurface-sample", case_iosurface_sample, YES},
    {"iosurface-target", case_iosurface_target, YES},
    {"tex-big", case_tex_big, YES},
    {"iosurface-big", case_iosurface_big, YES},
    {"tex-bufbig", case_tex_bufbig, YES},
    {"tex-update", case_tex_update, YES},
    {"compute", case_compute, YES},
    {"probe", case_probe, NO},
    {"sampler-probe", case_sampler_probe, NO},
    {"blend-probe", case_blend_probe, NO},
    {"iosurface-probe", case_iosurface_probe, NO},
    {"type-probe", case_type_probe, NO},
    {"blit-probe", case_blit_probe, NO},
    {"draw-probe", case_draw_probe, NO},
    {"patch-probe", case_patch_probe, NO},
    {"mtl4-probe", case_mtl4_probe, NO},
    {"stencil-probe", case_stencil_probe, NO},
    {"vtxstep-probe", case_vtxstep_probe, NO},
    {"pso-probe", case_pso_probe, NO},
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

        // Right after boot the GPU driver may not be up yet: wait for it.
        id<MTLDevice> dev = nil;
        for (int t = 0; t < 60 && !(dev = MTLCreateSystemDefaultDevice()); t++) sleep(1);
        printf("device: %s\n", dev ? dev.name.UTF8String : "(none after 60 s)");
        if (!dev) return 100;
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
