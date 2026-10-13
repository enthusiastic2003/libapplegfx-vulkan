#!/usr/bin/env python3
"""Generate an OpenCore config.plist for a macOS Tahoe QEMU/KVM guest.

Usage: gen-config.py <Sample.plist> <out config.plist>
Starts from OpenCore's Sample.plist and overrides only what the VM needs.
"""
import plistlib
import sys

# SMBIOS identity (generated with macserial -m iMac20,1). Keep stable across rebuilds.
SMBIOS = {
    "SystemProductName": "iMac20,1",
    "SystemSerialNumber": "C02D15ZTPN5T",
    "MLB": "C02027501GUPHC1UE",
    "SystemUUID": "180279D8-9C0E-42AC-851E-DC364F039FB5",
    # Must match the guest NIC MAC in run-macos.sh.
    "ROM": bytes.fromhex("525400ABCDEF"),
}

BOOT_ARGS = "-v keepsyms=1 debug=0x100 serial=3"

# NV=1: settings for the NullMoth NVIDIA driver (nullmoth/nvidia-macos-driver),
# for a passed-through NVIDIA GPU: SIP 0x0A43 and its boot args. NV=2 also
# excludes IONDRVSupport (the firmware framebuffer would take display 0).
import os
NV = int(os.environ.get("NV", "0") or 0)
# no npci=0x2000: it turns off macOS's 64-bit prefetchable BAR handling, and
# the passed-through GPU's BAR1 above 4 GB then vanished from the registry
NV_BOOT_ARGS = ("nvfb=1 nvaccel=1 nvfbheads=4 -nvkmsnosmooth amfi_get_out_of_my_way=0x1 "
                "amfi=0x80 ipc_control_port_options=0")
if NV:
    BOOT_ARGS = BOOT_ARGS + " " + NV_BOOT_ARGS
    if os.environ.get("NVSCAN") == "1":
        BOOT_ARGS += " nvscan=1"
    if os.environ.get("NVNOGO") == "1":   # start the card by hand: nvrmctl go 2
        BOOT_ARGS += " -nvrmnogo"

NVRAM_GUID = "7C436110-AB2A-4BBB-A880-FE41995C9F82"


def kext(bundle, exe, comment):
    return {
        "Arch": "x86_64", "BundlePath": bundle, "Comment": comment, "Enabled": True,
        "ExecutablePath": exe, "MaxKernel": "", "MinKernel": "", "PlistPath": "Contents/Info.plist",
    }


def driver(path):
    return {"Arguments": "", "Comment": "", "Enabled": True, "HideVerbose": False,
            "LoadEarly": False, "Path": path}


def main(sample, out):
    with open(sample, "rb") as f:
        p = plistlib.load(f)
    for k in [k for k in p if k.startswith("#")]:
        del p[k]

    p["ACPI"]["Add"] = [{"Comment": "EC + USBX", "Enabled": True, "Path": "SSDT-VM.aml"}]
    p["ACPI"]["Delete"] = []
    p["ACPI"]["Patch"] = []

    p["Booter"]["MmioWhitelist"] = []
    p["Booter"]["Patch"] = []

    p["DeviceProperties"]["Add"] = {}
    p["DeviceProperties"]["Delete"] = {}

    k = p["Kernel"]
    k["Add"] = [
        kext("Lilu.kext", "Contents/MacOS/Lilu", "Patch engine"),
        kext("VirtualSMC.kext", "Contents/MacOS/VirtualSMC", "SMC emulator"),
    ]
    k["Block"] = []
    if NV >= 2:
        k["Block"].append({
            "Arch": "Any", "Comment": "firmware framebuffer (NVIDIA driver owns display 0)",
            "Enabled": True, "Identifier": "com.apple.iokit.IONDRVSupport",
            "MaxKernel": "", "MinKernel": "", "Strategy": "Exclude",
        })
    k["Force"] = []
    k["Patch"] = []
    k["Quirks"].update({
        "DisableLinkeditJettison": True,
        "PanicNoKextDump": True,
        "PowerTimeoutKernelPanic": True,
        "ProvideCurrentCpuInfo": True,
    })

    m = p["Misc"]
    m["BlessOverride"] = []
    m["Entries"] = []
    m["Tools"] = []
    m["Boot"].update({"Timeout": 5, "PickerMode": "Builtin", "HideAuxiliary": False})
    # Log to the serial port (captured by QEMU to a file) instead of on screen.
    m["Debug"].update({"Target": 0x09, "AppleDebug": True, "ApplePanic": True})
    m["Serial"]["Init"] = True
    m["Security"].update({
        "AllowSetDefault": True,
        "ScanPolicy": 0,
        "SecureBootModel": "Disabled",
        "Vault": "Optional",
    })

    nv = p["NVRAM"]
    nv["Add"][NVRAM_GUID].update({
        "boot-args": BOOT_ARGS,
        "csr-active-config": bytes.fromhex("430A0000") if NV else bytes(4),
        "prev-lang:kbd": b"en-US:0",
    })
    nv["Delete"].setdefault(NVRAM_GUID, [])
    for key in ("boot-args", "csr-active-config", "prev-lang:kbd"):
        if key not in nv["Delete"][NVRAM_GUID]:
            nv["Delete"][NVRAM_GUID].append(key)
    nv["WriteFlash"] = True

    pi = p["PlatformInfo"]
    pi["Generic"].update(SMBIOS)
    pi["Automatic"] = True
    pi["UpdateSMBIOSMode"] = "Create"

    u = p["UEFI"]
    u["Drivers"] = [driver("OpenHfsPlus.efi"), driver("OpenRuntime.efi"), driver("ResetNvramEntry.efi")]
    u["Output"]["Resolution"] = "1920x1080@32"

    with open(out, "wb") as f:
        plistlib.dump(p, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
