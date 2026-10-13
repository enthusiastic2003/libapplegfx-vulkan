#!/bin/bash
# Build QEMU 11.0.0 (x86_64 only) with the apple-gfx-pci-linux device from mos-qemu,
# linked against libapplegfx-vulkan in ~/vm/prefix. Installs into ~/vm/prefix.
# Runs in a memory-capped cgroup so the build can't starve the running VM.
#   JOBS=3 MEMMAX=2G ./build-qemu.sh
set -euo pipefail

VM=/home/sirjanh/vm
PREFIX=$VM/prefix
SRC=$VM/downloads/qemu/qemu-11.0.0
MOS=$VM/downloads/src/mos-qemu
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
NV_PATCH=$SCRIPT_DIR/../qemu-11.0.0-nvidia-passthrough.patch
JOBS=${JOBS:-3}
MEMMAX=${MEMMAX:-2G}

cd "$SRC"

# Apply the generic VFIO/PCIe compatibility changes once.  They are opt-in and
# affect no devices unless the launcher requests their x-* properties.
if patch --dry-run -p1 < "$NV_PATCH" >/dev/null 2>&1; then
    patch -p1 < "$NV_PATCH"
elif ! patch --dry-run -R -p1 < "$NV_PATCH" >/dev/null 2>&1; then
    echo "QEMU NVIDIA passthrough patch neither applies nor is already present" >&2
    exit 1
fi

# Device sources (idempotent).
cp "$MOS"/hw/display/apple-gfx-pci-linux.c "$MOS"/hw/display/apple-gfx-common-linux.c \
   "$MOS"/hw/display/apple-gfx-linux.h hw/display/

# No default option ROM: upstream mos ships Apple's extracted EFI ROM, which we don't use.
# Boot display comes from the std VGA device instead.
sed -i 's|^\(\s*\)pci->romfile = PG_PCI_DEFAULT_ROMFILE;|\1/* no default option ROM */|' \
    hw/display/apple-gfx-pci-linux.c

if ! grep -q APPLE_GFX_PCI_LINUX hw/display/Kconfig; then
    cat >> hw/display/Kconfig <<'EOF'

config APPLE_GFX_PCI_LINUX
    bool
    default y if PCI_DEVICES
    depends on PCI
EOF
fi

if ! grep -q APPLE_GFX_PCI_LINUX hw/display/meson.build; then
    cat >> hw/display/meson.build <<'EOF'

# Apple paravirtual graphics PCI device, Linux host (libapplegfx-vulkan backend).
if config_all_devices.has_key('CONFIG_APPLE_GFX_PCI_LINUX')
  libapplegfx_vulkan = dependency('libapplegfx-vulkan', required: true)
  system_ss.add(when: 'CONFIG_APPLE_GFX_PCI_LINUX', if_true: [
    files('apple-gfx-pci-linux.c', 'apple-gfx-common-linux.c'),
    libapplegfx_vulkan,
  ])
endif
EOF
fi

# Trace points: swap upstream's apple-gfx.m block (macOS-only, not built here) for the
# Linux port's block, which has different argument types and extra events.
if ! grep -q apple_gfx_vblank_tick hw/display/trace-events; then
    awk -v mos="$MOS/hw/display/trace-events" '
        /^# apple-gfx\.m$/ {
            while ((getline l < mos) > 0) {
                if (l ~ /^# apple-gfx-pci \(Linux port\)/) f = 1
                if (f && l == "") break
                if (f) print l
            }
            skip = 1; next
        }
        skip && /^$/ { skip = 0 }
        !skip { print }
    ' hw/display/trace-events > hw/display/trace-events.new
    mv hw/display/trace-events.new hw/display/trace-events
fi

export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig
run() { systemd-run --user --scope -q -p MemoryMax="$MEMMAX" -p MemorySwapMax=1G -- "$@"; }

if [ ! -f build/build.ninja ]; then
    mkdir -p build
    (cd build && run ../configure \
        --prefix="$PREFIX" \
        --target-list=x86_64-softmmu \
        --enable-kvm --enable-gtk --enable-vnc --enable-slirp --enable-opengl \
        --disable-docs --disable-werror \
        --extra-ldflags="-Wl,-rpath,$PREFIX/lib")
fi
run ninja -C build -j"$JOBS"
run ninja -C build install >/dev/null
echo "Installed $PREFIX/bin/qemu-system-x86_64"
