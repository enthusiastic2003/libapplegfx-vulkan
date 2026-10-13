#!/bin/bash
# install-nvrmfb.sh <NVRMFB binary>: replace the installed NullMoth NVRMFB
# kext's binary and rebuild the auxiliary kernel collection the way NullMoth's
# install.sh does. Keeps a backup and restores it if the build fails.
# Run with sudo; reboot afterwards.
set -euo pipefail

NEW=${1:?usage: sudo $0 /tmp/nvscan-fb/NVRMFB.kext/Contents/MacOS/NVRMFB}
EXT=/Library/Extensions
BIN=$EXT/NVRMFB.kext/Contents/MacOS/NVRMFB
KC=/Library/KernelCollections/AuxiliaryKernelExtensions.kc
KB=/System/Library/KernelCollections/BootKernelExtensions.kc
KS=/System/Library/KernelCollections/SystemKernelExtensions.kc
KARG=(--allow-missing-kdk); [ -f /System/Library/Kernels/kernel ] && KARG+=(--kernel /System/Library/Kernels/kernel)

[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }
[ -s "$NEW" ] || { echo "no binary at $NEW" >&2; exit 1; }

# BACKUP=<dir> reuses the backup of an interrupted run (which may already
# have replaced the binary, so a fresh backup would hold the new one)
if [ -n "${BACKUP:-}" ]; then
    BK=$BACKUP
    [ -s "$BK/NVRMFB.kext/Contents/MacOS/NVRMFB" ] && [ -s "$BK/AuxiliaryKernelExtensions.kc" ] \
        || { echo "incomplete backup in $BK" >&2; exit 1; }
else
    BK=/Library/NullMoth/nvrmfb-backup-$(date +%Y%m%d-%H%M%S)
    mkdir -p "$BK"
    ditto "$EXT/NVRMFB.kext" "$BK/NVRMFB.kext"
    cp "$KC" "$BK/AuxiliaryKernelExtensions.kc"
fi
echo "backup: $BK"

restore() {
    echo "FAILED; restoring the previous NVRMFB" >&2
    rm -rf "$EXT/NVRMFB.kext" && ditto "$BK/NVRMFB.kext" "$EXT/NVRMFB.kext"
    cp "$BK/AuxiliaryKernelExtensions.kc" "$KC"
    exit 1
}
trap restore ERR

cp "$NEW" "$BIN"
chown -R root:wheel "$EXT/NVRMFB.kext"
chmod -R go-w "$EXT/NVRMFB.kext"

echo "building the kernel collection: takes several minutes, prints nothing meanwhile"
NEWKC=$KC.nvscan-new
rm -f "$NEWKC"
kmutil create -n aux --volume-root / "${KARG[@]}" -B "$KB" -S "$KS" --repository "$EXT" \
    -A "$NEWKC" -z > /var/tmp/nvscan-kmutil.log 2>&1 || { tail -20 /var/tmp/nvscan-kmutil.log >&2; false; }
INS=$(kmutil inspect -a x86_64 -A "$NEWKC")
for k in NVRM NVAccel NVRMFB NVRMAGDC; do
    printf '%s\n' "$INS" | grep -oE 'com\.nullmoth\.[A-Za-z0-9]+' | grep -Fxq "com.nullmoth.$k" \
        || { echo "com.nullmoth.$k missing from the new collection" >&2; false; }
done
mv -f "$NEWKC" "$KC"
trap - ERR
echo "installed: NVRMFB $(md5 -q "$BIN")"
echo "Reboot now: sudo shutdown -r now"
echo "Undo: sudo rm -rf $EXT/NVRMFB.kext && sudo ditto $BK/NVRMFB.kext $EXT/NVRMFB.kext && sudo cp $BK/AuxiliaryKernelExtensions.kc $KC"
