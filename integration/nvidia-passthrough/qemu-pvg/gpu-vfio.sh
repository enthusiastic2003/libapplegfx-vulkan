#!/bin/bash
# gpu-vfio.sh [host] : hand the laptop's RTX 5050 (01:00.0) and its HDMI audio
# (01:00.1) to vfio-pci for passthrough (run-macos.sh NVPT=1), or with "host"
# give them back to the host's nvidia / snd_hda_intel drivers. Run with sudo.
set -euo pipefail

GPU=0000:01:00.0
AUD=0000:01:00.1
USER_NAME=${SUDO_USER:-sirjanh}

[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }

unbind() {   # unbind device $1 from whatever driver has it
    local d=/sys/bus/pci/devices/$1
    [ -e "$d/driver" ] && echo "$1" > "$d/driver/unbind"
    return 0
}

if [ "${1:-}" = host ]; then
    for dev in $GPU $AUD; do
        unbind $dev
        echo "" > /sys/bus/pci/devices/$dev/driver_override
    done
    modprobe nvidia_drm 2>/dev/null || true
    echo $GPU > /sys/bus/pci/drivers_probe
    echo $AUD > /sys/bus/pci/drivers_probe
    echo "GPU: $(basename "$(readlink /sys/bus/pci/devices/$GPU/driver)")," \
         "audio: $(basename "$(readlink /sys/bus/pci/devices/$AUD/driver)")"
    exit 0
fi

# Nothing on the host may hold the GPU: the desktop runs on the Intel iGPU.
if command -v fuser >/dev/null && fuser -s /dev/nvidia* 2>/dev/null; then
    echo "processes still use the NVIDIA GPU:" >&2
    fuser -v /dev/nvidia* 2>&1 | sed 's/^/  /' >&2
    echo "close them (or: nvidia-smi --query-compute-apps=pid,name --format=csv)" >&2
    exit 1
fi
systemctl stop nvidia-persistenced 2>/dev/null || true

modprobe vfio-pci
for dev in $GPU $AUD; do
    echo vfio-pci > /sys/bus/pci/devices/$dev/driver_override
    unbind $dev
    echo $dev > /sys/bus/pci/drivers_probe
done

GROUP=$(basename "$(readlink /sys/bus/pci/devices/$GPU/iommu_group)")
chown "$USER_NAME" /dev/vfio/$GROUP   # QEMU runs as the user
echo "GPU: $(basename "$(readlink /sys/bus/pci/devices/$GPU/driver)")," \
     "audio: $(basename "$(readlink /sys/bus/pci/devices/$AUD/driver)"), /dev/vfio/$GROUP -> $USER_NAME"
echo "Start the VM with a memlock limit (VFIO pins all guest RAM), e.g.:"
echo "  sudo prlimit --memlock=unlimited:unlimited -- runuser -u $USER_NAME -- \\"
echo "      env NVPT=1 MEM=6G INSTALLER=0 HEADLESS=0 ~/vm/run-macos.sh"
