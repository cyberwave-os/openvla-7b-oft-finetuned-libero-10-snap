import argparse
import contextlib
import io
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

parser = argparse.ArgumentParser(description="OpenVLA-OFT FastAPI interface")
parser.add_argument("--model_path", type=str, default=os.environ.get("MODEL_PATH", ""))
parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
parser.add_argument("--device", type=str, default=os.environ.get("DEVICE", "cpu"))
args, _ = parser.parse_known_args()

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
        self.unnorm_key = unnorm_key if unnorm_key else training_config.get("dataset_name", "")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor, action_head, proprio_projector, cfg, model_ready, expected_proprio_dim, training_config

    # Load training_config.json from checkpoint
    tc = load_training_config(args.model_path)
    if tc is None:
        logging.warning(f"No training_config.json found in {args.model_path}, using defaults")
        tc = {}
    
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
    logging.info(f"Using action_dim={action_dim}, proprio_dim={proprio_dim}")

    cfg = SimpleConfig(args.model_path, tc)
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


@app.post("/act")
async def predict_action(
    text: str = Form(...),
    state: str = Form(None),
    image: UploadFile = File(...),
    image1: UploadFile = File(None),
    image2: UploadFile = File(None),
):
    """
    Predict action from image(s), instruction, and optional robot state.
    
    Args:
        text: Task instruction
        state: Robot state/proprioception as JSON array string (optional)
        image: Primary camera image (required)
        image1: Secondary camera image (optional, for multi-camera setups)
        image2: Third camera image (optional, for 3-camera setups)
    
    Returns:
        List of action arrays (action chunk)
    """
    try:
        # Get expected number of images from training config
        num_images_expected = training_config.get("num_images_in_input", 1)
        
        # Collect provided images
        images_provided = [image]
        image_names = ["image (primary)"]
        
        if image1 is not None:
            images_provided.append(image1)
            image_names.append("image1 (secondary)")
        
        if image2 is not None:
            images_provided.append(image2)
            image_names.append("image2 (tertiary)")
        
        num_images_provided = len(images_provided)
        
        # Validate number of images
        if num_images_provided < num_images_expected:
            return JSONResponse(
                {
                    "error": f"Expected {num_images_expected} images but received {num_images_provided}. "
                             f"Training config requires {num_images_expected} cameras."
                },
                status_code=400
            )
        
        if num_images_provided > num_images_expected:
            logging.warning(
                f"Received {num_images_provided} images but training config expects {num_images_expected}. "
                f"Only using first {num_images_expected} images."
            )
            images_provided = images_provided[:num_images_expected]
            image_names = image_names[:num_images_expected]
        
        logging.info(f"Processing {num_images_expected} images: {', '.join(image_names)}")
        
        # Load images
        observation = {}
        image_keys = ["full_image", "wrist_image", "tertiary_image"]
        
        for idx, (img_file, img_key) in enumerate(zip(images_provided, image_keys[:num_images_expected])):
            img_bytes = await img_file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            observation[img_key] = np.array(img)
        
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
                logging.info(f"Using provided robot state: {state_array}")
            except (json.JSONDecodeError, ValueError) as e:
                logging.warning(f"Failed to parse state: {e}. Using zero state.")
                observation["state"] = np.zeros(expected_proprio_dim, dtype=np.float32)
        else:
            # Use zero state as default
            observation["state"] = np.zeros(expected_proprio_dim, dtype=np.float32)
        
        # Run inference
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
        
        # Return action chunk as list of lists
        return JSONResponse([a.tolist() for a in actions])
        
    except Exception:
        logging.error(traceback.format_exc())
        return JSONResponse({"error": "Internal server error"}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
