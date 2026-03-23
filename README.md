# openvla-oft-finetuned-libero-10

Inference snap for **OpenVLA-OFT finetuned on LIBERO-10** with two serving interfaces:

- FastAPI multipart (`generic-cpu-fastapi`, `nvidia-gpu-amd64-fastapi`)
- XVLA JSON (`generic-cpu-xvla`, `nvidia-gpu-amd64-xvla`)

The snap package name in `snap/snapcraft.yaml` is:

- `openvla-oft-finetuned-libero-10`

## Build From Source

Clone the repository:

```shell
git clone https://github.com/canonical/openvla-7b-oft-finetuned-libero-10-snap.git
cd openvla-7b-oft-finetuned-libero-10-snap
```

```bash
snapcraft pack -v
```

Install the generated snap and components:

```bash
sudo snap install --dangerous ./openvla-oft-finetuned-libero-10_*.snap
sudo snap install --dangerous ./openvla-oft-finetuned-libero-10+*.comp
```

## Select Engine

Use the snap CLI app (`openvla-oft-finetuned-libero-10`) to inspect and select runtime engines:

```bash
sudo openvla-oft-finetuned-libero-10 list-engines
sudo openvla-oft-finetuned-libero-10 use-engine --auto
```

Or choose explicitly:

```bash
sudo openvla-oft-finetuned-libero-10 use-engine generic-cpu-xvla
sudo openvla-oft-finetuned-libero-10 use-engine nvidia-gpu-amd64-xvla
sudo openvla-oft-finetuned-libero-10 use-engine generic-cpu-fastapi
sudo openvla-oft-finetuned-libero-10 use-engine nvidia-gpu-amd64-fastapi
```

Then, check server status:

```shell
openvla-7b status
```

For instance, if the xvla server has been selected the status will look as follows:

```shell
engine: generic-cpu-xvla
services:
    server: active
endpoints:
    xvla-server-interface: http://localhost:9090/act
```

## Start And Check Server

```bash
sudo snap start openvla-oft-finetuned-libero-10.server
sudo snap services openvla-oft-finetuned-libero-10
sudo snap logs -f openvla-oft-finetuned-libero-10.server
```

Default serving port is managed by modelctl config (`http.port`, usually `9090`).

## Call The API

### XVLA JSON Endpoint

Use the included test client:

```bash
python3 test_xvla.py \
  --url http://localhost:9090/act \
  --image /path/to/image.png \
  --prompt "pick up the blue block"
```

Or send JSON directly (payload uses `instruction`, `image`).

### FastAPI Multipart Endpoint

When using a `*-fastapi` engine:

```bash
curl -X POST "http://localhost:9090/act" \
  -F "text=pick up the blue block" \
  -F "image=@/path/to/image.png"
```

## Resources

- [Inference Snaps documentation](https://documentation.ubuntu.com/inference-snaps/)
