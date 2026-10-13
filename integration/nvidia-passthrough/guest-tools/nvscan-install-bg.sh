#!/bin/bash
# Runs install-nvrmfb.sh as root in the background, so Ctrl-C or a dropped SSH
# connection can't interrupt the kernel-collection build. Log: /tmp/nvscan-install.log
#   sudo -v && BACKUP=/Library/NullMoth/nvrmfb-backup-... ~/tools/nvscan-install-bg.sh
# BACKUP (optional) reuses the backup of an interrupted earlier run.
sudo -b env ${BACKUP:+BACKUP="$BACKUP"} "$HOME/tools/install-nvrmfb.sh" \
    "$HOME/nvscan-fb/NVRMFB.kext/Contents/MacOS/NVRMFB" > /tmp/nvscan-install.log 2>&1
echo "started; log: /tmp/nvscan-install.log"
