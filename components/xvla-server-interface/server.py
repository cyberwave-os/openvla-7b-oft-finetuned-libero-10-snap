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

    proprio_stats = model.norm_stats.get(cfg.unnorm_key, {}).get("proprio", {})
    for key in ("q01", "q99", "min", "max"):
        if key in proprio_stats:
            expected_proprio_dim = int(np.asarray(proprio_stats[key]).reshape(-1).shape[0])
            break
    if expected_proprio_dim is None:
        expected_proprio_dim = int(PROPRIO_DIM)

    processor = get_processor(cfg)
    action_head = get_action_head(cfg, model.llm_dim).to(args.device)
    proprio_projector = get_proprio_projector(
        cfg,
        llm_dim=model.llm_dim,
        proprio_dim=expected_proprio_dim,
    ).to(args.device)

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


@app.post("/act")
def predict_action(payload: dict):
    try:
        if "language_instruction" not in payload and "instruction" not in payload:
            return JSONResponse({"error": "Missing field: language_instruction"}, status_code=400)

        primary_key = "image0" if "image0" in payload else ("full_image" if "full_image" in payload else None)
        if primary_key is None:
            return JSONResponse({"error": "No image provided. Include at least image0."}, status_code=400)

        images = [
            deserialize_image_payload(payload[key])
            for key in ("image0", "image1", "image2")
            if key in payload
        ]
        if not images:
            images = [deserialize_image_payload(payload[primary_key])]

        # XVLA request uses image0/image1/image2. OpenVLA-OFT expects observation
        # keys full_image/wrist_image/wrist_image2, so we map to that layout.
        observation = {"full_image": np.array(images[0])}
        if len(images) > 1:
            observation["wrist_image"] = np.array(images[1])
        if len(images) > 2:
            observation["wrist_image2"] = np.array(images[2])

        # External API accepts XVLA naming (`proprio`). Internally, OpenVLA-OFT
        # inference expects observation["state"], so we normalize into that shape.
        proprio_payload = payload.get("proprio")
        if proprio_payload is not None:
            state = np.asarray(
                json_numpy.loads(proprio_payload) if isinstance(proprio_payload, str) else proprio_payload,
                dtype=np.float32,
            ).reshape(-1)
        elif "state" in payload:
            raw_state = payload["state"]
            state = np.asarray(json_numpy.loads(raw_state) if isinstance(raw_state, str) else raw_state, dtype=np.float32).reshape(-1)
        else:
            state = np.zeros(expected_proprio_dim, dtype=np.float32)

        if state.shape[0] != expected_proprio_dim:
            raise ValueError(
                f"Invalid proprio/state length {state.shape[0]}; expected {expected_proprio_dim} for unnorm_key '{cfg.unnorm_key}'"
            )

        observation["state"] = state
        instruction = payload.get("language_instruction") or payload.get("instruction")

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
