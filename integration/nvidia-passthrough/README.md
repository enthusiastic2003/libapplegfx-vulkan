# NVIDIA passthrough on macOS 15

This directory captures the working RTX 5050 Laptop GPU passthrough setup used
with the NullMoth macOS driver.  It is intentionally separate from the normal
paravirtual Apple graphics path.

## What the QEMU patch fixes

macOS 15's PCI allocator rejects or erases resources larger than its 1 GiB BAR
limit and did not enumerate the GPU behind QEMU's generic root port because the
slot appeared empty.  The QEMU 11 patch adds generic, opt-in properties:

- `x-guest-bar1-size` exposes a smaller prefix of a larger host BAR1.
- `x-force-slot-present` keeps PCIe Presence Detect and Data Link Layer Active
  asserted across reset and guest configuration writes.
- `x-preserve-fw-resources` can preserve firmware BAR/window assignments for
  guests that erase otherwise valid resources.  The proven launcher does not
  need this property with the 512 MiB BAR1 compatibility window.

No NVIDIA vendor or device IDs are hardcoded in QEMU.

## Proven configuration

- Host GPU: NVIDIA RTX 5050 Laptop GPU (`10de:2d98`), audio `10de:22ec`
- Guest: macOS Sequoia 15.7.9
- Driver: NullMoth `nvidia-macos-driver` 1.14.7
- QEMU: 11.0.0
- Guest BAR1 compatibility window: 512 MiB
- Root port: hotplug enabled, power controller disabled, slot presence forced
- Output: NVIDIA HDMI through a USB capture card

The successful boot loaded `NVRM`, `NVRMFB`, `NVAccel`, and `NVRMAGDC` and
reported Metal 3 with 8 GB VRAM.

## Files

- `qemu-11.0.0-nvidia-passthrough.patch`: generic QEMU VFIO/PCIe changes.
- `qemu-pvg/build-qemu.sh`: applies the patch and builds QEMU plus apple-gfx.
- `qemu-pvg/gpu-vfio.sh`: binds/unbinds the GPU functions to VFIO.
- `qemu-pvg/gpu-reset.sh`: resets the laptop GPU and verifies BAR1/VFIO state.
- `run-macos.sh`: launcher snapshot containing `NVPT`, `NVPT_ROOT`, and `NV`.
- `efi/`: OpenCore configuration/image-generation snapshot for NullMoth.
- `guest-tools/nvidia-main.c`: makes NVIDIA HDMI primary for the current login
  session and removes broken hardware mirroring.
- `guest-tools/nvidia-mirror.c`: makes NVIDIA HDMI primary and mirrors the VGA
  display from it (hardware mirror); keeps both at a shared mode, so the
  NVIDIA display cannot switch to a 5K mode whose frames don't fit a slot.
- `nullmoth-nvscan.patch`: the NullMoth driver change (NVRMFB) that publishes
  the scanout through ivshmem and has the GPU copy the frames; the same commit
  is on the local `nvscan` branch of `downloads/src/nvidia-macos-driver`.
- `qemu-nvscan-display.c`: QEMU's `nvscan-display` device (committed in
  mos-qemu as `hw/display/nvscan-display.c`; `build-qemu.sh` copies it from
  there into the QEMU tree).
- `guest-tools/install-nvrmfb.sh`, `guest-tools/nvscan-install-bg.sh`: install
  a rebuilt NVRMFB into the auxiliary kernel collection (with a backup and
  rollback), in the background so the build survives SSH drops.
- `guest-tools/ce-probe.c`: user-space copy-engine probe (VRAM -> sysmem,
  physical addressing, VRAM -> ivshmem by guest-physical address).

## Boot sequence

The host must bind both NVIDIA functions to `vfio-pci`.  QEMU also needs an
unlimited memlock limit because VFIO maps guest RAM.

```bash
sudo ./qemu-pvg/gpu-reset.sh 12
NVPT=1 NVPT_ROOT=1 PVG=0 HEADLESS=0 MEM=6G INSTALLER=0 ./run-macos.sh
```

The launcher uses a generic root port with `x-force-slot-present=on` and the GPU
with `x-guest-bar1-size=512M`.  VGA remains present because macOS needs the
firmware framebuffer during boot.

After login, compile and run the guest helper to move the desktop to HDMI
without using NullMoth's partial hardware-mirror path:

```bash
clang nvidia-main.c -framework ApplicationServices -o nvidia-main
./nvidia-main
```

The helper uses `kCGConfigureForSession`; the boot display arrangement returns
after logout or reboot.

## The NVIDIA output in QEMU's window (NVSCAN)

With `NVSCAN=1` the launcher adds a 128 MiB ivshmem device backed by
`/dev/shm/nvscanout`, and QEMU's `nvscan-display` device on the same memory,
placed before the VGA device so it is console 0. NVRMFB (with the boot arg
`nvscan=1`: `cd efi && NV=2 NVSCAN=1 ./build-opencore.sh`) writes every flipped
frame into it; nvscan-display shows the newest one, and mirrors the VGA
console until frames arrive or after a guest reset. The raw VGA console stays
on the second tab.

The GPU copies the frames: NVRMFB hands the flipped surface to NVRM's
copy-engine page-off (`nvSurfPageoff`, NVIDIA's CeUtils), which writes it
straight to the slot's guest-physical pages. Measured at 1920x1080:

| | per frame |
|---|---|
| CPU copy (reads VRAM through BAR1, which KVM maps uncached: ~72 MB/s) | ~115 ms |
| GPU copy | ~0.82 ms (about 10 GB/s) |

The CPU copy ran inside the flip and held the whole desktop at ~8 fps; with
the GPU copy frames reach QEMU at the display's rate (54 fps measured while
dragging windows).

Boot:

```bash
sudo ./qemu-pvg/gpu-reset.sh 12            # before every VM start
NVPT=1 NVSCAN=1 PVG=0 HEADLESS=0 MEM=6G INSTALLER=0 ./run-macos.sh
ssh -p 2222 <user>@localhost ~/tools/nvidia-mirror   # after each login
```

Notes:

- NullMoth turns on its flip path about 60 s after the display is armed;
  until then QEMU shows the VGA mirror.
- The GPU drives a head only with an HDMI sink attached (here a USB capture
  card; a dummy plug would do). If the boot logs
  `NVRM-fb: dpy 0/1 0x100: connected 0`, the sink was not seen: restart the
  VM (after `gpu-reset.sh`).
- In this setup the driver placed BAR1 at 4 GiB itself
  (`bar1: PLACED — BAR1 4096 MB`), started the card in 500 ms and armed the
  display before its 60 s headless check.
