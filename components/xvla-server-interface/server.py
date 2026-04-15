import argparse
import contextlib
import io
import logging
import os
import sys
import traceback
from unittest.mock import MagicMock

import json_numpy
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from PIL import Image

# Create a dummy module for tensorflow_graphics
# This module is imported by openvla_utils but is not actually used for inference,
# so we can mock it to avoid adding it as a dependency.
mock_tf = MagicMock()
sys.modules["tensorflow_graphics"] = mock_tf
sys.modules["tensorflow_graphics.geometry"] = mock_tf
sys.modules["tensorflow_graphics.geometry.transformation"] = mock_tf

from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
)
from prismatic.vla.training_config import load_training_config
from prismatic.vla.constants import normalization_type_from_string
import experiments.robot.openvla_utils as _utils_mod

json_numpy.patch()

parser = argparse.ArgumentParser(description="OpenVLA-OFT XVLA interface")
parser.add_argument("--model_path", type=str, default=os.environ.get("MODEL_PATH", ""))
parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
parser.add_argument("--device", type=str, default=os.environ.get("DEVICE", "cpu"))
parser.add_argument("--unnorm_key", type=str, default=os.environ.get("UNNORM_KEY", ""))
parser.add_argument(
    "--num_images", type=int, default=int(os.environ.get("NUM_IMAGES", "1"))
)
parser.add_argument(
    "--use_proprio",
    type=lambda v: v.lower() != "false",
    default=os.environ.get("USE_PROPRIO", "true").lower() != "false",
)
parser.add_argument(
    "--proprio_dim", type=int, default=int(os.environ.get("PROPRIO_DIM", "6"))
)
args, _ = parser.parse_known_args()

model = None
processor = None
action_head = None
proprio_projector = None
cfg = None
model_ready = False
expected_proprio_dim = None
training_config = None


class SimpleConfig:
    def __init__(
        self,
        model_path: str,
        training_config: dict,
        unnorm_key: str = "",
        num_images: int = 1,
        use_proprio: bool = True,
    ):
        self.pretrained_checkpoint = model_path

        # Load all config from training_config.json
        self.use_l1_regression = training_config.get("use_l1_regression", True)
        self.use_diffusion = training_config.get("use_diffusion", False)
        self.use_film = training_config.get("use_film", False)
        self.load_in_8bit = False
        self.load_in_4bit = False
        self.num_images_in_input = training_config.get(
            "num_images_in_input", num_images
        )
        self.use_proprio = training_config.get("use_proprio", use_proprio)
        self.center_crop = True
        self.num_open_loop_steps = training_config.get("num_actions_chunk", 10)

        # Use provided unnorm_key or fallback to dataset_name from training_config
        self.unnorm_key = (
            unnorm_key if unnorm_key else training_config.get("dataset_name", "")
        )


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global \
        model, \
        processor, \
        action_head, \
        proprio_projector, \
        cfg, \
        model_ready, \
        expected_proprio_dim, \
        training_config

    # Load training_config.json from checkpoint
    tc = load_training_config(args.model_path)
    if tc is None:
        logging.warning(
            f"No training_config.json found in {args.model_path}, using defaults"
        )
        tc = {}

    training_config = tc
    logging.info(f"Loaded training config: {training_config}")

    # Override normalization_type global if specified in training_config
    if "normalization_type" in tc:
        norm_type = normalization_type_from_string(tc["normalization_type"])
        _utils_mod.ACTION_PROPRIO_NORMALIZATION_TYPE = norm_type
        logging.info(f"Set ACTION_PROPRIO_NORMALIZATION_TYPE to {norm_type}")

    # Extract dimensions from training_config
    action_dim = tc.get("action_dim", 6)
    proprio_dim = tc.get("proprio_dim", args.proprio_dim)
    expected_proprio_dim = proprio_dim
    logging.info(f"Using action_dim={action_dim}, proprio_dim={proprio_dim}")

    cfg = SimpleConfig(
        args.model_path,
        tc,
        unnorm_key=args.unnorm_key,
        num_images=args.num_images,
        use_proprio=args.use_proprio,
    )

    model = get_vla(cfg).to(args.device)

    if not cfg.unnorm_key:
        cfg.unnorm_key = next(iter(model.norm_stats.keys()))
    elif (
        cfg.unnorm_key not in model.norm_stats
        and f"{cfg.unnorm_key}_no_noops" in model.norm_stats
    ):
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"

    if cfg.unnorm_key not in model.norm_stats:
        raise RuntimeError(
            f"Invalid unnorm_key '{cfg.unnorm_key}'. Available keys: {list(model.norm_stats.keys())}"
        )

    processor = get_processor(cfg)

    # Pass action_dim explicitly to bypass global constant
    action_head = get_action_head(cfg, model.llm_dim, action_dim=action_dim).to(
        args.device
    )

    # TODO: External proprio API is pending; for now we keep an internal
    # placeholder state/projector required by openvla_utils.
    if cfg.use_proprio:
        logging.warning(
            "USE_PROPRIO input handling is TODO; using internal zero-state placeholder."
        )
        # Pass proprio_dim explicitly to bypass global constant
        proprio_projector = get_proprio_projector(
            cfg, llm_dim=model.llm_dim, proprio_dim=proprio_dim
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
    return Response(
        status_code=503, content='{"ready": false}', media_type="application/json"
    )


@app.get("/config")
async def get_config():
    """Expose the loaded training configuration for client discovery."""
    return training_config if training_config else {}


def deserialize_image_payload(image_payload):
    value = (
        json_numpy.loads(image_payload)
        if isinstance(image_payload, str)
        else image_payload
    )

    if isinstance(value, np.ndarray):
        if value.ndim == 1:
            try:
                return Image.open(io.BytesIO(value.astype(np.uint8).tobytes())).convert(
                    "RGB"
                )
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
        if (
            np.issubdtype(image_array.dtype, np.floating)
            and image_array.size > 0
            and image_array.max() <= 1.0
        ):
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


def get_additional_images(payload: dict, num_images: int):
    """Get additional camera images if num_images_in_input > 1."""
    additional = {}

    if num_images <= 1:
        return additional

    # Look for wrist/secondary camera images
    for key in payload:
        if key in (
            "image",
            "image0",
            "full_image",
            "instruction",
            "language_instruction",
            "state",
        ):
            continue

        # Check for image-related keys (wrist, image1, image2, secondary, etc.)
        if "image" in key.lower() or "wrist" in key.lower() or "camera" in key.lower():
            try:
                additional[key] = np.array(deserialize_image_payload(payload[key]))
            except Exception as e:
                logging.warning(f"Failed to deserialize additional image '{key}': {e}")

    return additional


@app.post("/act")
def predict_action(payload: dict):
    try:
        instruction = get_instruction(payload)
        image = get_primary_image(payload)

        # Get number of cameras from training config
        num_images = training_config.get("num_images_in_input", 1)

        observation = {
            "full_image": np.array(image),
            # TODO: replace with user-provided/stateful proprio once API is defined.
            "state": np.zeros(expected_proprio_dim, dtype=np.float32),
        }

        # Add additional camera images if configured
        if num_images > 1:
            additional_images = get_additional_images(payload, num_images)
            observation.update(additional_images)

            if len(additional_images) > 0:
                logging.info(
                    f"Using {len(additional_images) + 1} cameras: full_image + {list(additional_images.keys())}"
                )
            else:
                logging.warning(
                    f"training_config specifies {num_images} images but only primary image provided"
                )

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

        # Return the action chunk directly as a list
        response = [a.tolist() for a in actions]

        return JSONResponse(response)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception:
        logging.error(traceback.format_exc())
        return JSONResponse({"error": "Internal server error"}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
