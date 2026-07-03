"""Export a trained checkpoint to CPU inference artifacts.

Run after (or independently of) training:

    python -m ml.export --ckpt ml/weights/ckpt_latest.pt

Produces in ml/weights/:
    policy_ts.pt   TorchScript (torch CPU inference)
    policy.onnx    ONNX (onnxruntime -- what the Render server uses)
"""

from __future__ import annotations

import argparse
import os

import torch

from .encoding import NUM_CHANNELS
from .model import SnakeNet


def export_model(net: torch.nn.Module, board: int, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    net = net.cpu().eval()
    example = torch.zeros(1, NUM_CHANNELS, board, board)

    ts_path = os.path.join(out_dir, "policy_ts.pt")
    with torch.no_grad():
        ts = torch.jit.trace(net, example)
    ts.save(ts_path)
    print(f"[export] {ts_path}")

    onnx_path = os.path.join(out_dir, "policy.onnx")
    try:
        torch.onnx.export(
            net, example, onnx_path,
            input_names=["obs"], output_names=["logits", "value"],
            dynamic_axes={"obs": {0: "batch"},
                          "logits": {0: "batch"}, "value": {0: "batch"}},
            opset_version=17,
        )
        print(f"[export] {onnx_path}")
    except Exception as e:  # onnx is optional
        print(f"[export] ONNX export failed ({e}); TorchScript still works")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(
        os.path.dirname(__file__), "weights", "ckpt_latest.pt"))
    p.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "weights"))
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = ck.get("cfg", {"channels": 128, "blocks": 8, "board": 11})
    net = SnakeNet(cfg["channels"], cfg["blocks"], cfg["board"])
    net.load_state_dict(ck["model"])
    export_model(net, cfg["board"], args.out)


if __name__ == "__main__":
    main()
