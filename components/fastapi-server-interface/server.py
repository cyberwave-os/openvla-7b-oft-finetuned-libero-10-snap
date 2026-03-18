import argparse
import io

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image

from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

app = FastAPI()
model = None
processor = None
action_head = None
proprio_projector = None
cfg = None


class SimpleConfig:
    def __init__(self, model_path: str, unnorm_key: str = ""):
        self.pretrained_checkpoint = model_path
        self.use_l1_regression = True
        self.use_diffusion = False
        self.use_film = False
        self.load_in_8bit = False
        self.load_in_4bit = False
        self.num_images_in_input = 1
        self.use_proprio = True
        self.center_crop = True
        self.num_open_loop_steps = NUM_ACTIONS_CHUNK
        self.unnorm_key = unnorm_key


@app.on_event("startup")
def load_vla():
    global model, processor, action_head, proprio_projector, cfg

    cfg = SimpleConfig(args.model_path, unnorm_key=args.unnorm_key)
    model = get_vla(cfg).to(args.device)

    if not cfg.unnorm_key:
        cfg.unnorm_key = next(iter(model.norm_stats.keys()))
    elif cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"

    if cfg.unnorm_key not in model.norm_stats:
        raise RuntimeError(f"Invalid unnorm_key '{cfg.unnorm_key}'. Available keys: {list(model.norm_stats.keys())}")

    processor = get_processor(cfg)
    action_head = get_action_head(cfg, model.llm_dim).to(args.device)
    proprio_projector = get_proprio_projector(cfg, llm_dim=model.llm_dim, proprio_dim=PROPRIO_DIM).to(args.device)
    model.eval()


@app.post("/act")
async def predict_action(text: str = Form(...), image: UploadFile = File(...)):
    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    observation = {
        "full_image": np.array(img),
        "state": np.zeros(PROPRIO_DIM, dtype=np.float32),
    }

    with torch.inference_mode():
        actions = get_vla_action(cfg, model, processor, observation, text, action_head, proprio_projector)

    action = actions[0]
    return {
        "action_chunk": [a.tolist() for a in actions],
        "next_action": action.tolist(),
        "gripper_pose": float(action[-1]),
        "interpreted_action": "GRAB" if action[-1] > 0.5 else "RELEASE",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenVLA-OFT FastAPI interface")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--unnorm_key", type=str, default="")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=args.port)
