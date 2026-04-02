#!/bin/bash
set -eu

apt-get update -qq && apt-get install -y -qq curl snapd
sudo snap install snapcraft --classic

snapcraft pack -v

sudo snap install --dangerous ./openvla-oft-finetuned-libero-10_*.snap

sudo snap install --dangerous ./openvla-oft-finetuned-libero-10+*.comp

sudo openvla-oft-finetuned-libero-10 use-engine nvidia-gpu-xvla

sudo snap start openvla-oft-finetuned-libero-10.server

sudo snap services openvla-oft-finetuned-libero-10
