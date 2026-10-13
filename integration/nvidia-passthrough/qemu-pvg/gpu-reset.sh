#!/bin/bash
# gpu-reset.sh [size_log2_MiB] : put the passed-through RTX 5050 back in a clean
# state between VM runs. Unbinds it from vfio-pci, resets the bus behind its bridge
# (an FLR alone left the GSP firmware unable to boot again), resizes BAR1
# (default 11 = 2 GiB; 8 = 256 MiB; the full 8 GiB crashes QEMU under macOS),
# rebinds vfio-pci and gives the user the VFIO group. Run with sudo, VM stopped.
set -euo pipefail

GPU=0000:01:00.0
AUD=0000:01:00.1
SIZE=${1:-11}
USER_NAME=${SUDO_USER:-sirjanh}
D=/sys/bus/pci/devices/$GPU

[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }
pgrep -x qemu-system-x86 >/dev/null && { echo "stop the VM first" >&2; exit 1; }

for dev in $GPU $AUD; do
    if [ -e /sys/bus/pci/devices/$dev/driver ]; then
        echo $dev > /sys/bus/pci/devices/$dev/driver/unbind
    fi
done
# reset the whole bus behind the parent bridge (GPU and audio together): the
# per-device "bus" method is refused because the bus has two functions
BRIDGE=$(basename "$(readlink -f $D/..)")
echo 1 > /sys/bus/pci/devices/$BRIDGE/reset_subordinate
sleep 1
echo $SIZE > $D/resource1_resize
for dev in $GPU $AUD; do
    echo vfio-pci > /sys/bus/pci/devices/$dev/driver_override
    echo $dev > /sys/bus/pci/drivers_probe
done
GROUP=$(basename "$(readlink $D/iommu_group)")
chown "$USER_NAME" /dev/vfio/$GROUP

python3 -I -c "
l = open('$D/resource').read().split('\n')[1].split()
print('BAR1: %d MiB' % ((int(l[1], 16) - int(l[0], 16) + 1) >> 20))"
echo "GPU: $(basename "$(readlink $D/driver)"), audio: $(basename "$(readlink /sys/bus/pci/devices/$AUD/driver)"), /dev/vfio/$GROUP -> $USER_NAME"
