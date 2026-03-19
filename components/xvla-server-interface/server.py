import argparse
import contextlib
import io
import logging
import os
import traceback

import json_numpy
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from PIL import Image

from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

json_numpy.patch()

parser = argparse.ArgumentParser(description="OpenVLA-OFT XVLA interface")
parser.add_argument("--model_path", type=str, default=os.environ.get("MODEL_PATH", ""))
parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
parser.add_argument("--device", type=str, default=os.environ.get("DEVICE", "cpu"))
parser.add_argument("--unnorm_key", type=str, default=os.environ.get("UNNORM_KEY", ""))
parser.add_argument("--num_images", type=int, default=int(os.environ.get("NUM_IMAGES", "1")))
parser.add_argument("--use_proprio", type=lambda v: v.lower() != "false", default=os.environ.get("USE_PROPRIO", "true").lower() != "false")
parser.add_argument("--proprio_dim", type=int, default=int(os.environ.get("PROPRIO_DIM", str(PROPRIO_DIM))))
args, _ = parser.parse_known_args()

model = None
processor = None
action_head = None
proprio_projector = None
cfg = None
model_ready = False
expected_proprio_dim = None


class SimpleConfig:
    def __init__(self, model_path: str, unnorm_key: str = "", num_images: int = 1, use_proprio: bool = True):
        self.pretrained_checkpoint = model_path
        self.use_l1_regression = True
        self.use_diffusion = False
        self.use_film = False
        self.load_in_8bit = False
        self.load_in_4bit = False
        self.num_images_in_input = num_images
        self.use_proprio = use_proprio
        self.center_crop = True
        self.num_open_loop_steps = NUM_ACTIONS_CHUNK
        self.unnorm_key = unnorm_key


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor, action_head, proprio_projector, cfg, model_ready, expected_proprio_dim

    cfg = SimpleConfig(
        args.model_path,
        unnorm_key=args.unnorm_key,
        num_images=args.num_images,
        use_proprio=args.use_proprio,
    )

    model = get_vla(cfg).to(args.device)

    if not cfg.unnorm_key:
        cfg.unnorm_key = next(iter(model.norm_stats.keys()))
    elif cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"

    if cfg.unnorm_key not in model.norm_stats:
        raise RuntimeError(f"Invalid unnorm_key '{cfg.unnorm_key}'. Available keys: {list(model.norm_stats.keys())}")

    processor = get_processor(cfg)
    action_head = get_action_head(cfg, model.llm_dim).to(args.device)

    # TODO: External proprio API is pending; for now we keep an internal
    # placeholder state/projector required by openvla_utils.
    if cfg.use_proprio:
        logging.warning("USE_PROPRIO input handling is TODO; using internal zero-state placeholder.")
    expected_proprio_dim = int(PROPRIO_DIM)
    proprio_projector = get_proprio_projector(cfg, llm_dim=model.llm_dim, proprio_dim=expected_proprio_dim).to(args.device)

    model.eval()
    model_ready = True
    print("✅ Model loaded and ready for actions.")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/ready")
def ready():
    if model_ready:
        return {"ready": True}
    return Response(status_code=503, content='{"ready": false}', media_type="application/json")


def deserialize_image_payload(image_payload):
    value = json_numpy.loads(image_payload) if isinstance(image_payload, str) else image_payload

    if isinstance(value, np.ndarray):
        if value.ndim == 1:
            try:
                return Image.open(io.BytesIO(value.astype(np.uint8).tobytes())).convert("RGB")
            except Exception as exc:
                raise ValueError(f"Unable to decode image bytes: {exc}") from exc
        image_array = value
    elif isinstance(value, list):
        image_array = np.asarray(value)
    else:
        raise ValueError("Image payload must deserialize to numpy array or list")

    if image_array.ndim not in (2, 3):
        raise ValueError("Image payload must be 2D or 3D")

    if image_array.dtype != np.uint8:
        if np.issubdtype(image_array.dtype, np.floating) and image_array.size > 0 and image_array.max() <= 1.0:
            image_array = image_array * 255.0
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    if image_array.ndim == 3 and image_array.shape[2] == 1:
        image_array = image_array[:, :, 0]
    if image_array.ndim == 3 and image_array.shape[2] > 3:
        image_array = image_array[:, :, :3]

    return Image.fromarray(image_array).convert("RGB")


def get_instruction(payload: dict) -> str:
    instruction = payload.get("instruction") or payload.get("language_instruction")
    if instruction is None:
        raise ValueError("Missing field: instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Instruction must be a non-empty string")
    return instruction


def get_primary_image(payload: dict):
    for key in ("image", "image0", "full_image"):
        if key in payload:
            return deserialize_image_payload(payload[key])
    raise ValueError("Missing field: image")


@app.post("/act")
def predict_action(payload: dict):
    try:
        instruction = get_instruction(payload)
        image = get_primary_image(payload)

        observation = {
            "full_image": np.array(image),
            # TODO: replace with user-provided/stateful proprio once API is defined.
            "state": np.zeros(expected_proprio_dim, dtype=np.float32),
        }

        with torch.inference_mode():
            actions = get_vla_action(
                cfg,
                model,
                processor,
                observation,
                instruction,
                action_head=action_head,
                proprio_projector=proprio_projector,
                use_film=cfg.use_film,
            )

        response = {
            "action_chunk": [a.tolist() for a in actions],
            "action": actions[0].tolist(),
        }

        return JSONResponse(response)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception:
        logging.error(traceback.format_exc())
        return JSONResponse({"error": "Internal server error"}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
