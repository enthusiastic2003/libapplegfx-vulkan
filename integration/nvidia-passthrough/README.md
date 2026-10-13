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
