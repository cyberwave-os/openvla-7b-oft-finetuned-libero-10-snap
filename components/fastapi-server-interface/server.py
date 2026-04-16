#!/usr/bin/env python
"""
server_fastapi.py

Training-config-driven FastAPI multipart server for OpenVLA-OFT inference.

This server handles multipart form data (images as files) and automatically 
configures itself from training_config.json in the checkpoint.

Usage:
    python vla-scripts/server_fastapi.py --model_path /path/to/checkpoint [--port 8080] [--device cuda]
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

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, Response, UploadFile
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


app = None
model = None
processor = None
action_head = None
proprio_projector = None
cfg = None
model_ready = False
expected_proprio_dim = None
training_config = None


class SimpleConfig:
    def __init__(self, model_path: str, training_config: dict, unnorm_key: str = ""):
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
    logging.info("OpenVLA-OFT FastAPI Server Starting")
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

    # Override normalization_type global if specified in training_config
    if "normalization_type" in tc:
        norm_type = normalization_type_from_string(tc["normalization_type"])
        _utils_mod.ACTION_PROPRIO_NORMALIZATION_TYPE = norm_type
        logging.info(f"Set ACTION_PROPRIO_NORMALIZATION_TYPE to {norm_type}")

    # Extract dimensions from training_config
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

    # Pass action_dim explicitly to bypass global constant
    logging.info(
        f"Loading action head with llm_dim={model.llm_dim}, action_dim={action_dim}..."
    )
    try:
        from experiments.robot.openvla_utils import find_checkpoint_file
        action_head_checkpoint = find_checkpoint_file(args.model_path, "action_head")
        logging.info(f"  Found action head checkpoint: {action_head_checkpoint}")
    except Exception as e:
        logging.warning(f"  Could not find action head checkpoint: {e}")
    
    action_head = get_action_head(cfg, model.llm_dim, action_dim=action_dim).to(
        args.device
    )
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


@app.post("/act")
async def predict_action(
    text: str = Form(...),
    state: str = Form(None),
    image1: UploadFile = File(...),
    image2: UploadFile = File(None),
    image3: UploadFile = File(None),
):
    """
    Predict action from image(s), instruction, and optional robot state.

    Args:
        text: Task instruction
        state: Robot state/proprioception as JSON array string (optional)
        image1: Primary camera image (required)
        image2: Secondary camera image (optional, for multi-camera setups)
        image3: Third camera image (optional, for 3-camera setups)

    Returns:
        List of action arrays (action chunk)
    """
    try:
        logging.debug(f"Received /act request: text='{text[:30]}...', state provided={state is not None}")
        
        # Get expected number of images from training config
        num_images_expected = training_config.get("num_images_in_input", 1)
        logging.debug(f"Expected number of images: {num_images_expected}")

        # Collect provided images
        images_provided = [image1]
        image_names = ["image1 (primary)"]

        if image2 is not None:
            images_provided.append(image2)
            image_names.append("image2 (secondary)")

        if image3 is not None:
            images_provided.append(image3)
            image_names.append("image3 (tertiary)")

        num_images_provided = len(images_provided)
        logging.debug(f"Received {num_images_provided} images: {', '.join(image_names)}")

        # Validate number of images
        if num_images_provided < num_images_expected:
            return JSONResponse(
                {
                    "error": f"Expected {num_images_expected} images but received {num_images_provided}. "
                    f"Training config requires {num_images_expected} cameras."
                },
                status_code=400,
            )

        if num_images_provided > num_images_expected:
            logging.warning(
                f"Received {num_images_provided} images but training config expects {num_images_expected}. "
                f"Only using first {num_images_expected} images."
            )
            images_provided = images_provided[:num_images_expected]
            image_names = image_names[:num_images_expected]

        logging.info(
            f"Processing {num_images_expected} images: {', '.join(image_names)}"
        )

        # Load images
        observation = {}
        image_keys = ["full_image", "wrist_image", "tertiary_image"]

        for idx, (img_file, img_key) in enumerate(
            zip(images_provided, image_keys[:num_images_expected])
        ):
            img_bytes = await img_file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            observation[img_key] = np.array(img)
            logging.debug(f"Loaded {img_key}: shape={observation[img_key].shape}")

        # Parse robot state if provided
        if state is not None:
            try:
                import json

                state_array = json.loads(state)
                if not isinstance(state_array, list):
                    raise ValueError("State must be a JSON array")
                state_np = np.array(state_array, dtype=np.float32)

                # Validate state dimension
                if len(state_np) != expected_proprio_dim:
                    logging.warning(
                        f"Received state with {len(state_np)} dimensions, "
                        f"expected {expected_proprio_dim}. Using provided state anyway."
                    )
                observation["state"] = state_np
                logging.debug(f"Using provided robot state with {len(state_np)} dimensions")
            except (json.JSONDecodeError, ValueError) as e:
                logging.warning(f"Failed to parse state: {e}. Using zero state.")
                observation["state"] = np.zeros(expected_proprio_dim, dtype=np.float32)
        else:
            # Use zero state as default
            observation["state"] = np.zeros(expected_proprio_dim, dtype=np.float32)
            logging.debug(f"Using zero robot state ({expected_proprio_dim} dimensions)")

        # Run inference
        logging.debug("Starting VLA inference...")
        logging.info(
            f"Processing inference: {num_images_expected} cameras, instruction='{text[:30]}...'"
        )
        with torch.inference_mode():
            actions = get_vla_action(
                cfg,
                model,
                processor,
                observation,
                text,
                action_head=action_head,
                proprio_projector=proprio_projector,
                use_film=cfg.use_film,
            )
        logging.debug(f"Inference complete, generated {len(actions)} action steps")
        if len(actions) > 0:
            logging.debug(f"Action shape: {actions[0].shape}")

        # Return action chunk as list of lists
        response = [a.tolist() for a in actions]
        logging.debug(f"Returning {len(response)} action steps to client")
        return JSONResponse(response)

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
    
    parser = argparse.ArgumentParser(description="OpenVLA-OFT FastAPI interface")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    logging.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
