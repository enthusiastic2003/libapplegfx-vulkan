/*
 * ce-probe: can a user-space channel make the GPU's copy engine move a frame
 * out of VRAM? (Stage A of the NVSCAN GPU-copy path.)
 *
 *   1. virtual  VRAM -> sysmem         (a pattern written by the CPU into VRAM)
 *   2. physical VRAM -> virtual sysmem (source by its framebuffer offset)
 *   3. virtual  VRAM -> physical sysmem at a given guest-physical address
 *      (the ivshmem BAR: the pattern should appear in /dev/shm/nvscanout)
 *
 * Build in the guest, against NVIDIA's open kernel source (~/ogkm610):
 *   clang -O2 ce-probe.c -framework IOKit -o ce-probe $(./ce-probe-cflags)
 * Run: ./ce-probe [ivshmem-guest-physical-address, e.g. 0xf0000000]
 *
 * Modelled on NullMoth's kexts/NVRM/push.inc (channel and doorbell) and the
 * user-client shim in nvk/nvk-macos.patch (connections, escapes, mapping).
 */
#include <IOKit/IOKitLib.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "nvtypes.h"
#include "nvos.h"
#include "nv-ioctl.h"
#include "nv_escape.h"
#include "nv-unix-nvos-params-wrappers.h"
#include "nvmisc.h"
#include "class/cl0000.h"
#include "class/cl0002.h"
#include "class/cl003e.h"
#include "class/cl0040.h"
#include "class/cl0080.h"
#include "class/cl2080.h"
#include "class/cl50a0.h"
#include "class/cl90f1.h"
#include "class/clb0b5sw.h"
#include "class/clc361.h"
#include "class/clc461.h"
#include "class/clc36f.h"
#include "class/clc7b5.h"
#include "alloc/alloc_channel.h"
#include "ctrl/ctrl0000/ctrl0000gpu.h"
#include "ctrl/ctrl0041.h"
#include "ctrl/ctrl0080/ctrl0080gpu.h"
#include "ctrl/ctrla06f/ctrla06fgpfifo.h"
#include "ctrl/ctrlc36f.h"

enum { SEL_ESC = 1, SEL_BINDFD = 3 };

/* ---- user-client connections (nvk-macos.patch: nvrm_xnu_open / _ioctl) ---- */
static io_connect_t gConn[4096];

static io_connect_t nv_open(uint32_t type, int *fdOut)
{
    io_service_t svc = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("NVAccel"));
    uint32_t open_type = 0x4E560000u | type;
    if (!svc) {
        svc = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("NVRM"));
        open_type = type;
    }
    if (!svc) { fprintf(stderr, "no NVAccel/NVRM service\n"); exit(1); }
    io_connect_t conn = 0;
    kern_return_t kr = IOServiceOpen(svc, mach_task_self(), open_type, &conn);
    IOObjectRelease(svc);
    if (kr != KERN_SUCCESS) { fprintf(stderr, "IOServiceOpen(0x%x) 0x%08x\n", open_type, kr); exit(1); }
    int fd = open("/dev/null", O_RDONLY | O_CLOEXEC);
    uint64_t in = (uint64_t)fd;
    IOConnectCallScalarMethod(conn, SEL_BINDFD, &in, 1, NULL, NULL);
    gConn[fd] = conn;
    if (fdOut) *fdOut = fd;
    return conn;
}

static int nv_esc(io_connect_t conn, uint32_t cmd, void *data, uint32_t size)
{
    uint64_t in = cmd; size_t out = size;
    kern_return_t kr = IOConnectCallMethod(conn, SEL_ESC, &in, 1, data, size, NULL, NULL, data, &out);
    if (kr != KERN_SUCCESS) { fprintf(stderr, "esc 0x%02x -> 0x%08x\n", cmd, kr); return -1; }
    return 0;
}

static io_connect_t ctl;
static int ctlFd;
static NvHandle hClient, hDevice, hSub, hVaSpace;
static NvHandle gNext = 0xcafe0100;

static NvHandle rm_alloc(NvHandle parent, NvHandle h, NvU32 cls, void *params, NvU32 size, const char *what)
{
    NVOS21_PARAMETERS a; memset(&a, 0, sizeof a);
    a.hRoot = hClient ? hClient : h; a.hObjectParent = parent; a.hObjectNew = h; a.hClass = cls;
    a.pAllocParms = (NvP64)(uintptr_t)params; a.paramsSize = size;
    if (nv_esc(ctl, NV_ESC_RM_ALLOC, &a, sizeof a) || a.status) {
        fprintf(stderr, "  alloc %s (class 0x%x) status 0x%x\n", what, cls, a.status);
        return 0;
    }
    return a.hObjectNew;
}

static NvU32 rm_control(NvHandle obj, NvU32 cmd, void *params, NvU32 size)
{
    NVOS54_PARAMETERS c; memset(&c, 0, sizeof c);
    c.hClient = hClient; c.hObject = obj; c.cmd = cmd; c.params = (NvP64)(uintptr_t)params; c.paramsSize = size;
    if (nv_esc(ctl, NV_ESC_RM_CONTROL, &c, sizeof c)) return 0xffffffffu;
    return c.status;
}

/* ---- GPU buffers (push.inc: gbuf_alloc), optionally physically contiguous ---- */
struct gbuf { NvHandle hPhys, hVirt; NvU64 gpuva, size, phys; volatile void *cpu; };

static int gbuf_alloc(int vidmem, NvU64 size, const char *what, struct gbuf *b)
{
    memset(b, 0, sizeof *b); b->size = size;
    NV_MEMORY_ALLOCATION_PARAMS mp; memset(&mp, 0, sizeof mp);
    mp.owner = hClient; mp.type = NVOS32_TYPE_IMAGE; mp.size = size;
    mp.attr = DRF_DEF(OS32, _ATTR, _PAGE_SIZE, _4KB);
    if (vidmem) {
        mp.attr |= DRF_DEF(OS32, _ATTR, _PHYSICALITY, _CONTIGUOUS);   /* addressable by one physical offset */
        mp.attr |= DRF_DEF(OS32, _ATTR, _LOCATION, _VIDMEM) | DRF_DEF(OS32, _ATTR, _COHERENCY, _UNCACHED);
        mp.flags |= NVOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM;
    } else {
        mp.attr |= DRF_DEF(OS32, _ATTR, _LOCATION, _PCI) | DRF_DEF(OS32, _ATTR, _COHERENCY, _CACHED);
    }
    b->hPhys = rm_alloc(hDevice, gNext++, vidmem ? NV01_MEMORY_LOCAL_USER : NV01_MEMORY_SYSTEM, &mp, sizeof mp, what);
    if (!b->hPhys) return 1;

    int mfd; io_connect_t mc = nv_open(vidmem ? 1 : 0, &mfd);
    nv_ioctl_nvos33_parameters_with_fd m; memset(&m, 0, sizeof m);
    m.params.hClient = hClient; m.params.hDevice = hSub; m.params.hMemory = b->hPhys; m.params.length = size; m.fd = mfd;
    if (nv_esc(ctl, NV_ESC_RM_MAP_MEMORY, &m, sizeof m) || m.params.status) {
        fprintf(stderr, "  %s: map_memory status 0x%x\n", what, m.params.status); return 1;
    }
    mach_vm_address_t addr = 0; mach_vm_size_t len = 0;
    if (IOConnectMapMemory64(mc, 0, mach_task_self(), &addr, &len, kIOMapAnywhere) != KERN_SUCCESS) {
        fprintf(stderr, "  %s: IOConnectMapMemory64 failed\n", what); return 1;
    }
    b->cpu = (volatile void *)(uintptr_t)addr;

    NV_MEMORY_ALLOCATION_PARAMS vp; memset(&vp, 0, sizeof vp);
    vp.owner = hClient; vp.type = NVOS32_TYPE_IMAGE; vp.flags = NVOS32_ALLOC_FLAGS_VIRTUAL; vp.size = size; vp.hVASpace = hVaSpace;
    b->hVirt = rm_alloc(hDevice, gNext++, NV50_MEMORY_VIRTUAL, &vp, sizeof vp, "virtual");
    if (!b->hVirt) return 1;
    b->gpuva = vp.offset;
    NVOS46_PARAMETERS d; memset(&d, 0, sizeof d);
    d.hClient = hClient; d.hDevice = hDevice; d.hDma = b->hVirt; d.hMemory = b->hPhys; d.length = size;
    d.flags = DRF_DEF(OS46, _FLAGS, _PAGE_KIND, _VIRTUAL) |
              (vidmem ? DRF_DEF(OS46, _FLAGS, _CACHE_SNOOP, _DISABLE) : DRF_DEF(OS46, _FLAGS, _CACHE_SNOOP, _ENABLE));
    if (nv_esc(ctl, NV_ESC_RM_MAP_MEMORY_DMA, &d, sizeof d) || d.status) {
        fprintf(stderr, "  %s: map_memory_dma status 0x%x\n", what, d.status); return 1;
    }

    NV0041_CTRL_GET_SURFACE_PHYS_ATTR_PARAMS pa; memset(&pa, 0, sizeof pa);
    NvU32 st = rm_control(b->hPhys, NV0041_CTRL_CMD_GET_SURFACE_PHYS_ATTR, &pa, sizeof pa);
    b->phys = st ? ~0ull : pa.memOffset;
    printf("  %-8s %s %6llu KB cpu %p gpuva 0x%llx phys 0x%llx (attr status 0x%x aperture %u)\n", what,
           vidmem ? "vidmem" : "sysmem", (unsigned long long)(size >> 10), (void *)b->cpu,
           (unsigned long long)b->gpuva, (unsigned long long)b->phys, st, st ? 0 : pa.memAperture);
    return 0;
}

/* ---- channel ---- */
static struct gbuf userd, gpfifo, cmdbuf, sem;
static NvHandle hChan, hCopy;
static NvU32 token, ceClass;
static volatile uint8_t *usermode;
static NvU32 gpPut, pbOff;

#define MTHD(subch, mthd, n) (0x20000000u | ((NvU32)(n) << 16) | ((NvU32)(subch) << 13) | ((NvU32)(mthd) >> 2))
#define SUBCH_CE 4

static volatile NvU32 *pb_begin(void) { return (volatile NvU32 *)((volatile uint8_t *)cmdbuf.cpu + pbOff); }

/* Submit `ndw` dwords written at pb_begin(), then wait for the semaphore to reach `value`. */
static int submit_and_wait(NvU32 ndw, NvU32 value, const char *what)
{
    NvU64 va = cmdbuf.gpuva + pbOff;
    volatile NvU32 *gp = (volatile NvU32 *)gpfifo.cpu + gpPut * 2;
    gp[0] = (NvU32)va & ~3u;
    gp[1] = ((NvU32)(va >> 32) & 0xffu) | (ndw << 10);
    gpPut++; pbOff += ndw * 4;
    __sync_synchronize();
    ((volatile Nvc36fControl *)userd.cpu)->GPPut = gpPut;
    __sync_synchronize();
    *(volatile NvU32 *)(usermode + NVC361_NOTIFY_CHANNEL_PENDING) = token;
    volatile NvU32 *s = (volatile NvU32 *)sem.cpu;
    for (int i = 0; i < 2000; i++) {
        if (*s == value) return 0;
        usleep(1000);
    }
    fprintf(stderr, "  %s: semaphore %u, wanted %u (GPGet %u GPPut %u)\n", what, *s, value,
            ((volatile Nvc36fControl *)userd.cpu)->GPGet, gpPut);
    return 1;
}

static NvU32 gSemValue;

/* One pitch-linear copy of `lines` x `bytes` with a semaphore release at the end. */
static int ce_copy(NvU64 src, int srcPhys, NvU32 srcTarget, NvU64 dst, int dstPhys, NvU32 dstTarget,
                   NvU32 bytes, NvU32 lines, NvU32 pitch, const char *what)
{
    volatile NvU32 *p = pb_begin(); NvU32 n = 0;
    if (srcPhys) { p[n++] = MTHD(SUBCH_CE, NVC7B5_SET_SRC_PHYS_MODE, 1); p[n++] = srcTarget; }
    if (dstPhys) { p[n++] = MTHD(SUBCH_CE, NVC7B5_SET_DST_PHYS_MODE, 1); p[n++] = dstTarget; }
    p[n++] = MTHD(SUBCH_CE, NVC7B5_OFFSET_IN_UPPER, 8);
    p[n++] = (NvU32)(src >> 32); p[n++] = (NvU32)src;
    p[n++] = (NvU32)(dst >> 32); p[n++] = (NvU32)dst;
    p[n++] = pitch; p[n++] = pitch; p[n++] = bytes; p[n++] = lines;
    gSemValue++;
    p[n++] = MTHD(SUBCH_CE, NVC7B5_SET_SEMAPHORE_A, 3);
    p[n++] = (NvU32)(sem.gpuva >> 32); p[n++] = (NvU32)sem.gpuva; p[n++] = gSemValue;
    p[n++] = MTHD(SUBCH_CE, NVC7B5_LAUNCH_DMA, 1);
    p[n++] = DRF_DEF(C7B5, _LAUNCH_DMA, _DATA_TRANSFER_TYPE, _NON_PIPELINED) |
             DRF_DEF(C7B5, _LAUNCH_DMA, _FLUSH_ENABLE, _TRUE) |
             DRF_DEF(C7B5, _LAUNCH_DMA, _SEMAPHORE_TYPE, _RELEASE_ONE_WORD_SEMAPHORE) |
             DRF_DEF(C7B5, _LAUNCH_DMA, _SRC_MEMORY_LAYOUT, _PITCH) |
             DRF_DEF(C7B5, _LAUNCH_DMA, _DST_MEMORY_LAYOUT, _PITCH) |
             DRF_DEF(C7B5, _LAUNCH_DMA, _MULTI_LINE_ENABLE, _TRUE) |
             (srcPhys ? DRF_DEF(C7B5, _LAUNCH_DMA, _SRC_TYPE, _PHYSICAL) : DRF_DEF(C7B5, _LAUNCH_DMA, _SRC_TYPE, _VIRTUAL)) |
             (dstPhys ? DRF_DEF(C7B5, _LAUNCH_DMA, _DST_TYPE, _PHYSICAL) : DRF_DEF(C7B5, _LAUNCH_DMA, _DST_TYPE, _VIRTUAL));
    return submit_and_wait(n, gSemValue, what);
}

static int setup(void)
{
    int devFd;
    ctl = nv_open(0, &ctlFd);
    io_connect_t dev = nv_open(1, &devFd);
    nv_ioctl_card_info_t ci; memset(&ci, 0, sizeof ci);
    if (nv_esc(ctl, NV_ESC_CARD_INFO, &ci, sizeof ci) || !ci.valid) { fprintf(stderr, "no card info\n"); return 1; }
    nv_ioctl_register_fd_t rf; rf.ctl_fd = ctlFd; nv_esc(dev, NV_ESC_REGISTER_FD, &rf, sizeof rf);

    hClient = rm_alloc(0, 0, NV01_ROOT_CLIENT, NULL, 0, "root client");
    if (!hClient) return 1;
    NV0000_CTRL_GPU_ATTACH_IDS_PARAMS at; memset(&at, 0, sizeof at);
    at.gpuIds[0] = ci.gpu_id; at.gpuIds[1] = NV0000_CTRL_GPU_INVALID_ID;
    rm_control(hClient, NV0000_CTRL_CMD_GPU_ATTACH_IDS, &at, sizeof at);
    NV0080_ALLOC_PARAMETERS dp; memset(&dp, 0, sizeof dp); dp.hClientShare = hClient;
    hDevice = rm_alloc(hClient, 0xcafe0001, NV01_DEVICE_0, &dp, sizeof dp, "device");
    NV2080_ALLOC_PARAMETERS sp; memset(&sp, 0, sizeof sp);
    hSub = rm_alloc(hDevice, 0xcafe0002, NV20_SUBDEVICE_0, &sp, sizeof sp, "subdevice");
    if (!hDevice || !hSub) return 1;

    NV0080_CTRL_GPU_GET_CLASSLIST_V2_PARAMS cl; memset(&cl, 0, sizeof cl);
    rm_control(hDevice, NV0080_CTRL_CMD_GPU_GET_CLASSLIST_V2, &cl, sizeof cl);
    NvU32 chanClass = 0;
    printf("classes:");
    for (NvU32 i = 0; i < cl.numClasses && i < NV0080_CTRL_GPU_CLASSLIST_MAX_SIZE; i++) {
        NvU32 c = cl.classList[i];
        printf(" %x", c);
        if ((c & 0xff) == 0x6f && c > chanClass) chanClass = c;
        if ((c & 0xff) == 0xb5 && c > ceClass) ceClass = c;
    }
    printf("\nchannel class 0x%x, copy class 0x%x\n", chanClass, ceClass);
    if (!chanClass || !ceClass) return 1;

    NvHandle hUsermode = rm_alloc(hSub, gNext++, TURING_USERMODE_A, NULL, 0, "usermode");
    int ufd; io_connect_t umc = nv_open(1, &ufd);
    nv_ioctl_nvos33_parameters_with_fd um; memset(&um, 0, sizeof um);
    um.params.hClient = hClient; um.params.hDevice = hSub; um.params.hMemory = hUsermode; um.params.length = 4096; um.fd = ufd;
    if (!hUsermode || nv_esc(ctl, NV_ESC_RM_MAP_MEMORY, &um, sizeof um) || um.params.status) return 1;
    mach_vm_address_t uaddr = 0; mach_vm_size_t ulen = 0;
    if (IOConnectMapMemory64(umc, 0, mach_task_self(), &uaddr, &ulen, kIOMapAnywhere) != KERN_SUCCESS) return 1;
    usermode = (volatile uint8_t *)(uintptr_t)uaddr;

    NV_VASPACE_ALLOCATION_PARAMETERS vap; memset(&vap, 0, sizeof vap);
    vap.flags = NV_VASPACE_ALLOCATION_FLAGS_RETRY_PTE_ALLOC_IN_SYS;
    hVaSpace = rm_alloc(hDevice, gNext++, FERMI_VASPACE_A, &vap, sizeof vap, "vaspace");
    if (!hVaSpace) return 1;

    struct gbuf notifier;
    if (gbuf_alloc(0, 0x1000, "notifier", &notifier) || gbuf_alloc(1, 0x80000, "userd", &userd) ||
        gbuf_alloc(0, 0x40000, "gpfifo", &gpfifo) || gbuf_alloc(0, 0x10000, "cmdbuf", &cmdbuf) ||
        gbuf_alloc(0, 0x1000, "sem", &sem))
        return 1;
    memset((void *)sem.cpu, 0, 0x1000); memset((void *)notifier.cpu, 0, 0x1000);

    NV_CONTEXT_DMA_ALLOCATION_PARAMS cd; memset(&cd, 0, sizeof cd);
    cd.flags = DRF_DEF(OS03, _FLAGS, _MAPPING, _KERNEL) | DRF_DEF(OS03, _FLAGS, _HASH_TABLE, _DISABLE);
    cd.hMemory = notifier.hPhys; cd.limit = notifier.size - 1;
    NvHandle hCtxDma = rm_alloc(hDevice, gNext++, NV01_CONTEXT_DMA, &cd, sizeof cd, "ctxdma");
    if (!hCtxDma) return 1;

    /* A graphics channel, as push.inc's (proven) test uses; the copy class runs
     * on its subchannel 4 through the graphics copy engine. */
    NV_CHANNEL_ALLOC_PARAMS cp; memset(&cp, 0, sizeof cp);
    cp.hObjectError = hCtxDma; cp.gpFifoOffset = gpfifo.gpuva; cp.gpFifoEntries = 0x8000; cp.hVASpace = hVaSpace;
    cp.hUserdMemory[0] = userd.hPhys; cp.engineType = NV2080_ENGINE_TYPE_GRAPHICS;
    hChan = rm_alloc(hDevice, gNext++, chanClass, &cp, sizeof cp, "channel");
    if (!hChan) return 1;
    NVB0B5_ALLOCATION_PARAMETERS ca; memset(&ca, 0, sizeof ca); ca.version = NVB0B5_ALLOCATION_PARAMETERS_VERSION_1;
    ca.engineType = NV2080_ENGINE_TYPE_GRAPHICS;
    hCopy = rm_alloc(hChan, gNext++, ceClass, &ca, sizeof ca, "copy object");
    if (!hCopy) return 1;

    NVA06F_CTRL_BIND_PARAMS bp; memset(&bp, 0, sizeof bp); bp.engineType = NV2080_ENGINE_TYPE_GRAPHICS;
    NvU32 st = rm_control(hChan, NVA06F_CTRL_CMD_BIND, &bp, sizeof bp);
    if (st) { fprintf(stderr, "bind 0x%x\n", st); return 1; }
    NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS sc; memset(&sc, 0, sizeof sc); sc.bEnable = NV_TRUE;
    if ((st = rm_control(hChan, NVA06F_CTRL_CMD_GPFIFO_SCHEDULE, &sc, sizeof sc))) { fprintf(stderr, "schedule 0x%x\n", st); return 1; }
    NVC36F_CTRL_GPFIFO_SET_WORK_SUBMIT_TOKEN_NOTIF_INDEX_PARAMS ni; memset(&ni, 0, sizeof ni);
    ni.index = NV_CHANNELGPFIFO_NOTIFICATION_TYPE_WORK_SUBMIT_TOKEN;
    rm_control(hChan, NVC36F_CTRL_CMD_GPFIFO_SET_WORK_SUBMIT_TOKEN_NOTIF_INDEX, &ni, sizeof ni);
    NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS tk; memset(&tk, 0, sizeof tk);
    if ((st = rm_control(hChan, NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN, &tk, sizeof tk))) { fprintf(stderr, "token 0x%x\n", st); return 1; }
    token = tk.workSubmitToken;

    /* Bind the copy class to subchannel 4. */
    volatile NvU32 *p = pb_begin();
    p[0] = MTHD(SUBCH_CE, 0, 1); p[1] = ceClass;
    gSemValue++;
    p[2] = MTHD(SUBCH_CE, NVC7B5_SET_SEMAPHORE_A, 3);
    p[3] = (NvU32)(sem.gpuva >> 32); p[4] = (NvU32)sem.gpuva; p[5] = gSemValue;
    p[6] = MTHD(SUBCH_CE, NVC7B5_LAUNCH_DMA, 1);
    p[7] = DRF_DEF(C7B5, _LAUNCH_DMA, _DATA_TRANSFER_TYPE, _NONE) |
           DRF_DEF(C7B5, _LAUNCH_DMA, _SEMAPHORE_TYPE, _RELEASE_ONE_WORD_SEMAPHORE);
    if (submit_and_wait(8, gSemValue, "bind copy object")) return 1;
    printf("copy engine bound on subchannel %d: semaphore OK\n", SUBCH_CE);
    return 0;
}

static int check(volatile const NvU32 *d, NvU32 words, NvU32 seed, const char *what)
{
    for (NvU32 i = 0; i < words; i++)
        if (d[i] != (i * 2654435761u ^ seed)) {
            printf("%s: MISMATCH at word %u: 0x%08x, wanted 0x%08x\n", what, i, d[i], i * 2654435761u ^ seed);
            return 1;
        }
    printf("%s: OK (%u KB)\n", what, words >> 8);
    return 0;
}

int main(int argc, char **argv)
{
    NvU64 ivshmemPhys = argc > 1 ? strtoull(argv[1], NULL, 0) : 0;
    if (setup()) { printf("SETUP FAILED\n"); return 1; }

    const NvU32 W = 1920, H = 1080, pitch = W * 4, bytes = pitch * H;
    struct gbuf src, dst;
    if (gbuf_alloc(1, bytes, "src", &src) || gbuf_alloc(0, bytes, "dst", &dst)) return 1;
    NvU32 words = bytes / 4;
    volatile NvU32 *s = (volatile NvU32 *)src.cpu;
    for (NvU32 i = 0; i < words; i++) s[i] = i * 2654435761u ^ 0x1234u;   /* slow: CPU writes into VRAM */

    int fail = 0;
    memset((void *)dst.cpu, 0, bytes);
    if (!ce_copy(src.gpuva, 0, 0, dst.gpuva, 0, 0, pitch, H, pitch, "1 virtual"))
        fail |= check((volatile NvU32 *)dst.cpu, words, 0x1234u, "1 virtual  VRAM -> sysmem");
    else fail = 1;

    if (src.phys != ~0ull) {
        memset((void *)dst.cpu, 0, bytes);
        if (!ce_copy(src.phys, 1, NVC7B5_SET_SRC_PHYS_MODE_TARGET_LOCAL_FB, dst.gpuva, 0, 0, pitch, H, pitch, "2 physical src"))
            fail |= check((volatile NvU32 *)dst.cpu, words, 0x1234u, "2 physical VRAM -> sysmem");
        else fail = 1;
    }

    if (ivshmemPhys) {
        /* Into the ivshmem BAR, past the header and both 32 MiB slots: the host checks it. */
        NvU64 at = ivshmemPhys + 4096 + 2ull * 32 * 1024 * 1024;
        if (!ce_copy(src.gpuva, 0, 0, at, 1, NVC7B5_SET_DST_PHYS_MODE_TARGET_COHERENT_SYSMEM, pitch, H, pitch, "3 physical dst"))
            printf("3 virtual VRAM -> physical 0x%llx: copy done; check /dev/shm/nvscanout at offset 0x%llx on the host\n",
                   (unsigned long long)at, (unsigned long long)(at - ivshmemPhys));
        else fail = 1;
    }

    /* Timing: ten full-frame copies VRAM -> sysmem. */
    uint64_t t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    for (int i = 0; i < 10; i++) ce_copy(src.gpuva, 0, 0, dst.gpuva, 0, 0, pitch, H, pitch, "timing");
    uint64_t t1 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    printf("10 frame copies: %.2f ms each\n", (t1 - t0) / 10.0 / 1e6);
    printf(fail ? "SOME STEPS FAILED\n" : "ALL STEPS OK\n");
    return fail;
}
