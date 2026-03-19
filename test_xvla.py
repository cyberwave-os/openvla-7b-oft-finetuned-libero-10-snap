import argparse
import time

import json_numpy
import numpy as np
import requests
from PIL import Image


def build_payload(image_path: str, prompt: str) -> dict:
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image, dtype=np.uint8)

    payload = {
        "instruction": prompt,
        "image": json_numpy.dumps(image_np),
    }

    return payload


def extract_action(parsed_json):
    if not isinstance(parsed_json, dict):
        return None

    if "action" in parsed_json:
        return parsed_json["action"]
    if "full_action_vector" in parsed_json:
        return parsed_json["full_action_vector"]

    chunk = parsed_json.get("action_chunk")
    if isinstance(chunk, list) and len(chunk) > 0:
        return chunk[0]

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Test OpenVLA-OFT XVLA /act endpoint")
    parser.add_argument("--url", default="http://localhost:9090/act", help="XVLA endpoint URL")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--prompt", required=True, help="Language instruction")
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--read-timeout", type=float, default=300.0)
    args = parser.parse_args()

    payload = build_payload(args.image, args.prompt)

    print(f"Sending request to {args.url}")
    print(f"Timeouts => connect: {args.connect_timeout}s, read: {args.read_timeout}s")

    started = time.time()
    response = requests.post(
        args.url,
        json=payload,
        timeout=(args.connect_timeout, args.read_timeout),
    )
    elapsed = time.time() - started

    print(f"HTTP {response.status_code} in {elapsed:.2f}s")
    response.raise_for_status()

    parsed = response.json()
    action = extract_action(parsed)

    if action is None:
        print("Response did not contain action/full_action_vector/action_chunk")
        print(parsed)
        return

    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    print(f"Action length: {action_arr.shape[0]}")
    print(f"Action: {action_arr.tolist()}")


if __name__ == "__main__":
    main()
