import argparse
import contextlib
import io
import logging
import sys
from unittest.mock import MagicMock

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
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
        self.unnorm_key = unnorm_key if unnorm_key else training_config.get("dataset_name", "")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor, action_head, proprio_projector, cfg, training_config

    # Load training_config.json from checkpoint
    tc = load_training_config(args.model_path)
    if tc is None:
        logging.warning(f"No training_config.json found in {args.model_path}, using defaults")
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
    proprio_dim = tc.get("proprio_dim", 6)
    logging.info(f"Using action_dim={action_dim}, proprio_dim={proprio_dim}")

    cfg = SimpleConfig(args.model_path, tc, unnorm_key=args.unnorm_key)
    model = get_vla(cfg).to(args.device)

    if not cfg.unnorm_key:
        cfg.unnorm_key = next(iter(model.norm_stats.keys()))
    elif cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"

    if cfg.unnorm_key not in model.norm_stats:
        raise RuntimeError(f"Invalid unnorm_key '{cfg.unnorm_key}'. Available keys: {list(model.norm_stats.keys())}")

    processor = get_processor(cfg)
    
    # Pass action_dim explicitly to bypass global constant
    action_head = get_action_head(cfg, model.llm_dim, action_dim=action_dim).to(args.device)
    
    # Pass proprio_dim explicitly to bypass global constant
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, llm_dim=model.llm_dim, proprio_dim=proprio_dim).to(args.device)
    
    model.eval()
    logging.info("✅ Model loaded and ready for inference")
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/act")
async def predict_action(text: str = Form(...), image: UploadFile = File(...), wrist_image: UploadFile = File(None)):
    """
    Predict action from image(s) and instruction.
    
    Args:
        text: Task instruction
        image: Primary camera image (required)
        wrist_image: Wrist camera image (optional, for multi-camera setups)
    """
    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Use proprio_dim and num_images from training_config
    proprio_dim = training_config.get("proprio_dim", 6)
    num_images = training_config.get("num_images_in_input", 1)

    observation = {
        "full_image": np.array(img),
        # TODO: replace with user-provided/stateful proprio once API is defined.
        "state": np.zeros(proprio_dim, dtype=np.float32),
    }
    
    # Add wrist camera if provided and multi-camera is configured
    if wrist_image is not None and num_images > 1:
        wrist_bytes = await wrist_image.read()
        wrist_img = Image.open(io.BytesIO(wrist_bytes)).convert("RGB")
        observation["wrist_image"] = np.array(wrist_img)
        logging.info(f"Using {num_images} cameras: full_image + wrist_image")
    elif num_images > 1 and wrist_image is None:
        logging.warning(f"training_config specifies {num_images} images but only primary image provided")

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

    # Return the action chunk directly as a list
    return [a.tolist() for a in actions]


@app.get("/config")
async def get_config():
    """Expose the loaded training configuration for client discovery."""
    return training_config if training_config else {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenVLA-OFT FastAPI interface")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--unnorm_key", type=str, default="")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=args.port)
