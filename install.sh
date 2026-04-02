#!/bin/bash
set -eu

export LD_LIBRARY_PATH=""
export LD_PRELOAD=""

env -i PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin" \
  HOME="$HOME" TERM="$TERM" \
  apt-get update -qq && \
env -i PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin" \
  HOME="$HOME" TERM="$TERM" \
  apt-get install -y -qq curl snapd || true

sudo systemctl start snapd.socket snapd.service || true
sudo snap wait system seed.loaded || true

sudo snap install snapcraft --classic 2>/dev/null || true

# Fix snap-confine permissions if needed
if [ -f /usr/lib/snapd/snap-confine ]; then
  sudo chmod 6755 /usr/lib/snapd/snap-confine
  sudo chown root:root /usr/lib/snapd/snap-confine
fi

snapcraft pack -v

sudo snap install --dangerous ./openvla-oft-finetuned-libero-10_*.snap

sudo snap install --dangerous ./openvla-oft-finetuned-libero-10+*.comp

sudo openvla-oft-finetuned-libero-10 use-engine nvidia-gpu-xvla

sudo snap start openvla-oft-finetuned-libero-10.server

sudo snap services openvla-oft-finetuned-libero-10
