#!/bin/bash
# Build images/opencore.img: a GPT disk with one FAT32 ESP holding OpenCore.
set -euo pipefail

VM=/home/sirjanh/vm
OC=$VM/OpenCore-1.0.8-RELEASE
HERE=$VM/efi
STAGE=$HERE/stage
# NV=1|2: the NVIDIA-driver variant (see gen-config.py), as opencore-nv.img
IMG=$VM/images/opencore.img
[ "${NV:-0}" != 0 ] && IMG=$VM/images/opencore-nv.img
export NV=${NV:-0}
export NVSCAN=${NVSCAN:-0}

rm -rf "$STAGE"
mkdir -p "$STAGE/EFI/BOOT" "$STAGE/EFI/OC"/{ACPI,Drivers,Kexts,Tools}

cp "$OC/X64/EFI/BOOT/BOOTx64.efi" "$STAGE/EFI/BOOT/"
# Hide the OpenCore disk itself from the picker (selecting it just re-launches OpenCore).
printf Disabled > "$STAGE/EFI/BOOT/.contentVisibility"
cp "$OC/X64/EFI/OC/OpenCore.efi" "$STAGE/EFI/OC/"
for d in OpenHfsPlus OpenRuntime ResetNvramEntry; do
    cp "$OC/X64/EFI/OC/Drivers/$d.efi" "$STAGE/EFI/OC/Drivers/"
done
cp -r "$VM/downloads/Lilu-1.7.2-RELEASE/Lilu.kext" "$STAGE/EFI/OC/Kexts/"
cp -r "$VM/downloads/VirtualSMC-1.3.8-RELEASE/Kexts/VirtualSMC.kext" "$STAGE/EFI/OC/Kexts/"

iasl -p "$STAGE/EFI/OC/ACPI/SSDT-VM" "$HERE/SSDT-VM.dsl" >/dev/null
python3 "$HERE/gen-config.py" "$OC/Docs/Sample.plist" "$STAGE/EFI/OC/config.plist"
"$OC/Utilities/ocvalidate/ocvalidate.linux" "$STAGE/EFI/OC/config.plist"

rm -f "$IMG"
truncate -s 256M "$IMG"
sgdisk -n 1:2048:0 -t 1:ef00 -c 1:EFI "$IMG" >/dev/null
SECTORS=$(( $(sgdisk -i 1 "$IMG" | sed -n 's/^Last sector: \([0-9]*\).*/\1/p') - 2048 + 1 ))
mformat -i "$IMG@@1M" -F -T "$SECTORS" -h 64 -s 32 -v OPENCORE ::
mcopy -i "$IMG@@1M" -s "$STAGE/EFI" ::/
echo "Built $IMG"
