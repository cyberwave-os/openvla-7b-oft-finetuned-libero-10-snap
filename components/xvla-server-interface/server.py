#!/usr/bin/env python
"""
server_xvla.py

Training-config-driven XVLA JSON interface server for OpenVLA-OFT inference.

This server uses JSON payloads with json_numpy encoding for image data,
and automatically configures itself from training_config.json in the checkpoint.

Usage:
    python vla-scripts/server_xvla.py --model_path /path/to/checkpoint [--port 8080] [--device cuda]
"""

import argparse
import contextlib
import io
import json
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
    ):
        self.pretrained_checkpoint = model_path

        # Load all config from training_config.json
        self.use_l1_regression = training_config.get("use_l1_regression", True)
        self.use_diffusion = training_config.get("use_diffusion", False)
        self.use_film = training_config.get("use_film", False)
        self.load_in_8bit = False
        self.load_in_4bit = False
        self.num_images_in_input = training_config.get("num_images_in_input", 1)
        self.use_proprio = training_config.get("use_proprio", True)
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

    logging.info("=" * 80)
    logging.info("OpenVLA-OFT XVLA Server Starting")
    logging.info("=" * 80)
    logging.info(f"Python version: {sys.version}")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    logging.info(f"Model checkpoint: {args.model_path}")
    logging.info(f"Device: {args.device}")
    logging.info(f"Port: {args.port}")

    # Load training_config.json from checkpoint
    logging.info(f"Loading training_config.json from {args.model_path}")
    tc = load_training_config(args.model_path)
    if tc is None:
        logging.warning(
            f"No training_config.json found in {args.model_path}, using defaults"
        )
        tc = {}
    else:
        logging.info(f"Training config loaded successfully with {len(tc)} parameters")

    training_config = tc
    # Config is already logged by load_training_config()

    # Override all global constants from training_config
    from prismatic.vla.constants import override_constants_from_training_config
    override_constants_from_training_config(args.model_path)
    
    # Extract dimensions for local use (already set globally above)
    action_dim = tc.get("action_dim", 6)
    proprio_dim = tc.get("proprio_dim", 6)
    expected_proprio_dim = proprio_dim
    num_images = tc.get("num_images_in_input", 1)
    logging.info(
        f"Model configuration: action_dim={action_dim}, proprio_dim={proprio_dim}, num_images={num_images}"
    )

    cfg = SimpleConfig(args.model_path, tc)

    logging.info("Loading base VLA model...")
    model = get_vla(cfg).to(args.device)
    logging.info(f"✓ Base VLA model loaded successfully")
    logging.info(f"  Model LLM dimension: {model.llm_dim}")
    logging.info(f"  Model type: {type(model).__name__}")
    
    # CRITICAL: Update model's internal num_open_loop_steps to match training config
    # The model may have a stale value saved in its checkpoint
    if hasattr(model, 'config') and hasattr(model.config, 'num_open_loop_steps'):
        old_value = model.config.num_open_loop_steps
        model.config.num_open_loop_steps = tc.get("num_actions_chunk", 10)
        logging.info(f"Updated model.config.num_open_loop_steps: {old_value} → {model.config.num_open_loop_steps}")
    if hasattr(model, 'num_open_loop_steps'):
        old_value = model.num_open_loop_steps
        model.num_open_loop_steps = tc.get("num_actions_chunk", 10)
        logging.info(f"Updated model.num_open_loop_steps: {old_value} → {model.num_open_loop_steps}")

    if not cfg.unnorm_key:
        cfg.unnorm_key = next(iter(model.norm_stats.keys()))
        logging.info(
            f"No unnorm_key specified, using first available: {cfg.unnorm_key}"
        )
    elif (
        cfg.unnorm_key not in model.norm_stats
        and f"{cfg.unnorm_key}_no_noops" in model.norm_stats
    ):
        old_key = cfg.unnorm_key
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        logging.info(f"Adjusted unnorm_key: {old_key} → {cfg.unnorm_key}")

    if cfg.unnorm_key not in model.norm_stats:
        available_keys = list(model.norm_stats.keys())
        logging.error(
            f"Invalid unnorm_key '{cfg.unnorm_key}'. Available keys: {available_keys}"
        )
        raise RuntimeError(
            f"Invalid unnorm_key '{cfg.unnorm_key}'. Available keys: {available_keys}"
        )

    logging.info(f"Using normalization key: {cfg.unnorm_key}")

    logging.info("Loading processor...")
    processor = get_processor(cfg)
    logging.info("✓ Processor loaded successfully")

    # Pass action_dim explicitly to bypass global constants
    num_actions_chunk = tc.get("num_actions_chunk", 10)
    logging.info(
        f"Loading action head with llm_dim={model.llm_dim}, action_dim={action_dim}..."
    )
    try:
        from experiments.robot.openvla_utils import find_checkpoint_file

        action_head_checkpoint = find_checkpoint_file(args.model_path, "action_head")
        logging.info(f"  Found action head checkpoint: {action_head_checkpoint}")
    except Exception as e:
        logging.warning(f"  Could not find action head checkpoint: {e}")

    action_head = get_action_head(cfg, model.llm_dim, action_dim=action_dim).to(args.device)
    
    # Configure action head's num_actions_chunk from training config
    if hasattr(action_head, 'num_actions_chunk'):
        old_val = action_head.num_actions_chunk
        action_head.num_actions_chunk = num_actions_chunk
        logging.info(f"  Updated action_head.num_actions_chunk: {old_val} → {num_actions_chunk}")
    
    logging.info("✓ Action head loaded successfully")

    # Pass proprio_dim explicitly to bypass global constant
    if cfg.use_proprio:
        logging.info(
            f"Loading proprio projector with llm_dim={model.llm_dim}, proprio_dim={proprio_dim}..."
        )
        try:
            from experiments.robot.openvla_utils import find_checkpoint_file

            proprio_checkpoint = find_checkpoint_file(
                args.model_path, "proprio_projector"
            )
            logging.info(f"  Found proprio projector checkpoint: {proprio_checkpoint}")
        except Exception as e:
            logging.warning(f"  Could not find proprio projector checkpoint: {e}")

        proprio_projector = get_proprio_projector(
            cfg, llm_dim=model.llm_dim, proprio_dim=proprio_dim
        ).to(args.device)
        logging.info("✓ Proprio projector loaded successfully")
    else:
        logging.info("Proprio disabled in training config, skipping proprio projector")

    model.eval()
    model_ready = True
    logging.info("=" * 80)
    logging.info("✅ Model loaded and ready for inference")
    logging.info("=" * 80)
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
                img = Image.open(io.BytesIO(value.astype(np.uint8).tobytes())).convert(
                    "RGB"
                )
                logging.debug(f"Deserialized 1D byte array to PIL Image: {img.size}")
                return img
            except Exception as exc:
                logging.error(f"Unable to decode image bytes: {exc}")
                raise ValueError(f"Unable to decode image bytes: {exc}") from exc
        image_array = value
        logging.debug(
            f"Using numpy array directly: shape={image_array.shape}, dtype={image_array.dtype}"
        )
    elif isinstance(value, Image.Image):
        logging.debug(f"Using PIL Image directly: {value.size}")
        return value
    else:
        logging.error(f"Unexpected image payload type: {type(value)}")
        raise ValueError(f"Unexpected image payload type: {type(value)}")

    if image_array.ndim not in (2, 3):
        logging.error(f"Invalid image dimensions: {image_array.ndim}")
        raise ValueError("Image payload must be 2D or 3D")

    if image_array.dtype != np.uint8:
        logging.debug(f"Converting image from {image_array.dtype} to uint8")
        if (
            np.issubdtype(image_array.dtype, np.floating)
            and image_array.size > 0
            and image_array.max() <= 1.0
        ):
            logging.debug("Scaling float image [0,1] to [0,255]")
            image_array = image_array * 255.0
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    if image_array.ndim == 3 and image_array.shape[2] == 1:
        logging.debug("Converting single-channel image to grayscale")
        image_array = image_array[:, :, 0]
    if image_array.ndim == 3 and image_array.shape[2] > 3:
        logging.debug(f"Truncating image from {image_array.shape[2]} channels to 3")
        image_array = image_array[:, :, :3]

    result = Image.fromarray(image_array).convert("RGB")
    logging.debug(f"Final PIL Image: {result.size}")
    return result


def get_instruction(payload: dict) -> str:
    instruction = payload.get("instruction") or payload.get("language_instruction")
    if instruction is None:
        raise ValueError("Missing field: instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Instruction must be a non-empty string")
    return instruction


def get_primary_image(payload: dict):
    for key in ("image1", "full_image", "image", "image0"):
        if key in payload:
            return deserialize_image_payload(payload[key])
    raise ValueError("Missing field: image1 or full_image")


def get_additional_images(payload: dict, num_images: int):
    """Get additional camera images if num_images_in_input > 1."""
    additional = {}

    if num_images <= 1:
        return additional

    # Look for standardized image2, image3 first, then fallback to other keys
    standard_keys = ["image2", "image3"]
    for key in standard_keys:
        if key in payload:
            try:
                additional[key] = np.array(deserialize_image_payload(payload[key]))
            except Exception as e:
                logging.warning(f"Failed to deserialize image '{key}': {e}")

    # Look for wrist/secondary camera images
    for key in payload:
        if (
            key
            in (
                "image1",
                "image",
                "image0",
                "full_image",
                "instruction",
                "language_instruction",
                "state",
            )
            or key in additional
        ):
            continue

        # Check for image-related keys (wrist, secondary, camera, etc.)
        if "image" in key.lower() or "wrist" in key.lower() or "camera" in key.lower():
            try:
                additional[key] = np.array(deserialize_image_payload(payload[key]))
            except Exception as e:
                logging.warning(f"Failed to deserialize additional image '{key}': {e}")

    return additional


@app.post("/act")
def predict_action(payload: dict):
    """
    Predict action from observation.

    Accepts JSON payload with "encoded" field containing json_numpy-encoded observation.
    The observation should contain:
    - instruction: Task instruction string
    - image1: Primary camera image (numpy array) - can also use full_image
    - image2: Secondary camera image (optional, for multi-camera setups)
    - image3: Third camera image (optional, for 3-camera setups)
    - state: Robot state (optional, auto-filled if missing)
    - Backward compatible with wrist_image, full_image, and other image key names

    Returns:
        List of action arrays (action chunk)
    """
    try:
        logging.debug(
            f"Received /act request with payload keys: {list(payload.keys())}"
        )

        # Handle double-encoded payload (json_numpy support)
        if "encoded" in payload:
            logging.debug("Decoding double-encoded payload")
            payload = json.loads(payload["encoded"])
            logging.debug(f"Decoded payload keys: {list(payload.keys())}")

        instruction = get_instruction(payload)
        logging.debug(
            f"Instruction: {instruction[:50]}..."
            if len(instruction) > 50
            else f"Instruction: {instruction}"
        )

        # Get number of cameras from training config
        num_images = training_config.get("num_images_in_input", 1)
        logging.debug(f"Expected number of images: {num_images}")

        # Build observation with standardized keys
        observation = {
            # Use provided state or create zero state
            "state": payload.get(
                "state", np.zeros(expected_proprio_dim, dtype=np.float32)
            ),
        }

        state_shape = (
            observation["state"].shape
            if hasattr(observation["state"], "shape")
            else len(observation["state"])
        )
        logging.debug(
            f"Robot state shape: {state_shape}, expected: {expected_proprio_dim}"
        )

        # Map image1/image2/image3 to internal keys: full_image, wrist_image, tertiary_image
        image_mapping = [
            (["image1", "full_image", "image", "image0"], "full_image"),
            (["image2", "wrist_image"], "wrist_image"),
            (["image3", "tertiary_image"], "tertiary_image"),
        ]

        images_found = []
        for input_keys, internal_key in image_mapping:
            for input_key in input_keys:
                if input_key in payload:
                    try:
                        img = deserialize_image_payload(payload[input_key])
                        img_array = np.array(img)
                        observation[internal_key] = img_array
                        images_found.append(f"{input_key} → {internal_key}")
                        logging.debug(
                            f"Mapped {input_key} → {internal_key}, shape: {img_array.shape}"
                        )
                        break
                    except Exception as e:
                        logging.warning(f"Failed to deserialize '{input_key}': {e}")

        # Validate we have the primary image
        if "full_image" not in observation:
            logging.error("Missing primary image in payload")
            raise ValueError("Missing primary image (expected image1 or full_image)")

        # Validate we have enough images
        actual_image_count = sum(
            1
            for k in observation
            if k in ("full_image", "wrist_image", "tertiary_image")
        )

        logging.debug(
            f"Image count: {actual_image_count}/{num_images} - {images_found}"
        )

        if actual_image_count < num_images:
            error_msg = f"Expected {num_images} images but received {actual_image_count}. Provide image1, image2, ... up to image{num_images}"
            logging.error(error_msg)
            return JSONResponse(
                {"error": error_msg},
                status_code=400,
            )

        logging.info(
            f"Processing inference: {actual_image_count} cameras, instruction='{instruction[:30]}...'"
        )

        with torch.inference_mode():
            logging.debug("Starting VLA inference...")
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
            logging.debug(f"Inference complete, generated {len(actions)} action steps")
            if len(actions) > 0:
                logging.debug(f"Action shape: {actions[0].shape}")

        # Return the action chunk directly as a list
        response = [a.tolist() for a in actions]
        logging.debug(f"Returning {len(response)} action steps to client")

        return JSONResponse(response)
    except ValueError as exc:
        logging.warning(f"ValueError in /act: {exc}")
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        logging.error(f"Exception in /act: {exc}")
        logging.error(traceback.format_exc())
        return JSONResponse({"error": "Internal server error"}, status_code=500)


if __name__ == "__main__":
    # Configure logging
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="OpenVLA-OFT XVLA interface")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    logging.info("Starting XVLA server...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
