#!/bin/bash
# Launch the macOS Sequoia guest under QEMU/KVM.
#   ./run-macos.sh              GTK window + VNC on 127.0.0.1:5901
#   HEADLESS=1 ./run-macos.sh   VNC only
#   INSTALLER=0 ./run-macos.sh  detach the recovery/installer disk
#   PVG=1 ./run-macos.sh        use our QEMU build (~/vm/prefix) and add the
#                               paravirtual GPU device; library log in images/lagfx.log
# Control sockets: images/qmp.sock (QMP), images/monitor.sock (HMP).
# OpenCore and kernel logs go to images/serial.log. SSH: ssh -p 2222 user@localhost
set -euo pipefail

VM=/home/sirjanh/vm
IMG=$VM/images
MEM=${MEM:-8G}
SMP=${SMP:-8}

[ -f "$IMG/OVMF_VARS.fd" ] || cp /usr/share/edk2/ovmf/OVMF_VARS.fd "$IMG/OVMF_VARS.fd"
[ -f "$IMG/macos.qcow2" ] || qemu-img create -f qcow2 "$IMG/macos.qcow2" 80G

# GL=1: GTK draws the frame with OpenGL (no CPU scaling/compositing in pixman)
if [ "${HEADLESS:-0}" = 1 ]; then DISPLAY_ARGS=(-display none)
elif [ "${GL:-0}" = 1 ]; then DISPLAY_ARGS=(-display gtk,gl=on)
else DISPLAY_ARGS=(-display gtk); fi
# QEMU's standard VGA: the firmware/OpenCore screen, and macOS needs that boot
# framebuffer to bring up the PVG display (without it the PVG driver never
# starts). macOS sees it as a second display: keep the two mirrored.
# VGA=0 drops it (the guest then has no working display).
VGA_ARGS=(-device VGA,id=vga,vgamem_mb=64)
[ "${VGA:-1}" = 0 ] && VGA_ARGS=(-vga none)
VNC_ARGS=(-vnc 127.0.0.1:1)
[ "${GL:-0}" = 1 ] && VNC_ARGS=()   # QEMU's VNC server can't share a GL display

QEMU=qemu-system-x86_64
PVG_ARGS=()
if [ "${PVG:-0}" = 1 ]; then
    QEMU=$VM/prefix/bin/qemu-system-x86_64
    PVG_ARGS=(-device apple-gfx-pci,id=pvg,gpu_cores=${PVG_CORES:-4})
    export LAGFX_LOG_LEVEL=${LAGFX_LOG_LEVEL:-info} LAGFX_LOG_FILE=$IMG/lagfx.log
fi
# NVIDIA passthrough uses the local QEMU build for the opt-in firmware-resource
# preservation properties even when the PVG device itself is disabled.
[ "${NVPT:-0}" = 1 ] && QEMU=$VM/prefix/bin/qemu-system-x86_64

INSTALLER_ARGS=()
if [ "${INSTALLER:-1}" = 1 ]; then
    INSTALLER_ARGS=(-drive id=installer,if=none,format=raw,file="$IMG/BaseSystem.img"
                    -device ide-hd,bus=sata.3,drive=installer)
fi

# Guest RAM in a shareable memfd: the PVG library maps texture memory from it
# and the GPU copies to and from guest RAM directly. MEMFD=0: anonymous RAM.
MEM_ARGS=(-m "$MEM")
MACHINE=q35,accel=kvm
if [ "${MEMFD:-1}" = 1 ]; then
    MEM_ARGS+=(-object memory-backend-memfd,id=ram,size="$MEM",share=on)
    MACHINE+=,memory-backend=ram
fi

# NV=1: OpenCore set up for the NullMoth NVIDIA driver (efi/: NV=1|2
# ./build-opencore.sh). NVPT=1: also pass the laptop's RTX 5050 (01:00.0)
# and its HDMI audio (01:00.1) through; bind them to vfio-pci first
# (qemu-pvg/gpu-vfio.sh). The NVIDIA GPU then drives the HDMI port. PVG stays
# (the desktop until the NVIDIA driver runs) unless PVG=0.
OCIMG=opencore.img
[ "${NV:-0}" = 1 ] || [ "${NVPT:-0}" = 1 ] && OCIMG=opencore-nv.img
NVPT_ARGS=()
CPU_EXTRA=""
if [ "${NVPT:-0}" = 1 ]; then
    # the guest's address width must be the host's (39 bits here): the
    # firmware places the GPU's 8 GiB BAR by it, and KVM refuses anything
    # the host can't address
    CPU_EXTRA=",host-phys-bits=on"
    # Put the GPU behind a real root port: NullMoth requires one before it will
    # place BAR1 and take the display immediately. Hide native hotplug so the
    # guest PCI configurator has no reason to tear down OVMF's working bridge
    # window. Do not use pref64-reserve here: that is extra hotplug capacity on
    # top of the GPU's BARs, and a 4G reserve makes OVMF leave this port empty.
    # NVPT_ROOT=0 retains the old root-bus layout for comparison.
    if [ "${NVPT_ROOT:-1}" = 1 ]; then
        NVPT_ARGS=(-device pcie-root-port,id=nvroot,bus=pcie.0,addr=0x6.0,chassis=7,slot=7,hotplug=on,power_controller_present=off,x-do-not-expose-native-hotplug-cap=on,x-force-slot-present=on
                   -device vfio-pci,host=0000:01:00.0,bus=nvroot,addr=0x0.0,multifunction=on,x-guest-bar1-size=512M
                   -device vfio-pci,host=0000:01:00.1,bus=nvroot,addr=0x0.1
                   -fw_cfg name=opt/ovmf/X-PciMmio64Mb,string=65536)
    else
        NVPT_ARGS=(-device vfio-pci,host=0000:01:00.0,bus=pcie.0,addr=0x10.0,multifunction=on,x-preserve-fw-resources=on
                   -device vfio-pci,host=0000:01:00.1,bus=pcie.0,addr=0x10.1,x-preserve-fw-resources=on
                   -fw_cfg name=opt/ovmf/X-PciMmio64Mb,string=65536)
    fi
fi

exec "$QEMU" \
    -name macos-sequoia \
    -machine "$MACHINE" \
    -cpu Skylake-Client-noTSX-IBRS,vendor=GenuineIntel,+invtsc,vmware-cpuid-freq=on,enforce"$CPU_EXTRA" \
    -smp "$SMP",sockets=1,cores=$((SMP / 2)),threads=2 \
    "${MEM_ARGS[@]}" \
    -drive if=pflash,format=raw,readonly=on,file=/usr/share/edk2/ovmf/OVMF_CODE.fd \
    -drive if=pflash,format=raw,file="$IMG/OVMF_VARS.fd" \
    -device ich9-ahci,id=sata \
    -drive id=opencore,if=none,format=raw,file="$IMG/$OCIMG" \
    -device ide-hd,bus=sata.2,drive=opencore,bootindex=0 \
    -drive id=macos,if=none,format=qcow2,file="$IMG/macos.qcow2",discard=unmap \
    -device ide-hd,bus=sata.4,drive=macos \
    "${INSTALLER_ARGS[@]}" \
    -netdev user,id=net0,hostfwd=tcp:127.0.0.1:2222-:22 \
    -device vmxnet3,netdev=net0,mac=52:54:00:AB:CD:EF \
    -device qemu-xhci,id=xhci \
    -device usb-kbd,bus=xhci.0 \
    -device usb-tablet,bus=xhci.0 \
    "${VGA_ARGS[@]}" \
    "${PVG_ARGS[@]}" \
    "${NVPT_ARGS[@]}" \
    "${DISPLAY_ARGS[@]}" \
    "${VNC_ARGS[@]}" \
    -chardev file,id=serial0,path="$IMG/serial.log" \
    -serial chardev:serial0 \
    -qmp unix:"$IMG/qmp.sock",server=on,wait=off \
    -monitor unix:"$IMG/monitor.sock",server=on,wait=off
