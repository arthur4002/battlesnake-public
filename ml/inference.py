"""Server-side inference: Battlesnake JSON -> move.

Loads the exported network once (ONNX Runtime preferred: tiny install, fast
CPU inference; falls back to TorchScript if torch is installed). Every /move:

    1. encode the board exactly as during training,
    2. forward pass -> policy logits,
    3. apply the same certain-death mask used in training,
    4. play the argmax of the masked policy.

If both model files are missing, falls back to a safe heuristic move so the
server never crashes.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

from .encoding import encode_obs, safe_action_mask, state_from_json
from .rules import MOVE_NAMES

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
_backend = None  # ("onnx", session) | ("torch", module) | None


def _load():
    global _backend
    if _backend is not None:
        return _backend

    onnx_path = os.path.join(WEIGHTS_DIR, "policy.onnx")
    ts_path = os.path.join(WEIGHTS_DIR, "policy_ts.pt")

    if os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            sess = ort.InferenceSession(
                onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
            _backend = ("onnx", sess)
            return _backend
        except Exception:
            pass

    if os.path.exists(ts_path):
        try:
            import torch
            torch.set_num_threads(1)
            mod = torch.jit.load(ts_path, map_location="cpu").eval()
            _backend = ("torch", mod)
            return _backend
        except Exception:
            pass

    _backend = ("none", None)
    return _backend


def _policy_logits(obs: np.ndarray) -> np.ndarray:
    kind, m = _load()
    x = obs[None].astype(np.float32)
    if kind == "onnx":
        return m.run(["logits"], {"obs": x})[0][0]
    if kind == "torch":
        import torch
        with torch.no_grad():
            logits, _ = m(torch.from_numpy(x))
        return logits[0].numpy()
    return np.zeros(4, dtype=np.float32)  # no model -> uniform over safe moves


def get_info() -> Dict:
    return {
        "apiversion": "1",
        "author": "Alex2034",
        "color": "#7C3AED",
        "head": "smart-caterpillar",
        "tail": "coffee",
        "version": "ml-ppo-1.0",
    }


def choose_move(game_state: Dict) -> str:
    try:
        width, height, food, snakes, me = state_from_json(game_state)
        obs = encode_obs(width, height, food, snakes, me)
        mask = safe_action_mask(width, height, snakes, me)
        logits = _policy_logits(obs)
        logits = np.where(mask, logits, -1e9)
        return MOVE_NAMES[int(np.argmax(logits))]
    except Exception:
        # never 500 the game server
        return _fallback(game_state)


def _fallback(game_state: Dict) -> str:
    try:
        width, height, food, snakes, me = state_from_json(game_state)
        mask = safe_action_mask(width, height, snakes, me)
        for i in range(4):
            if mask[i]:
                return MOVE_NAMES[i]
    except Exception:
        pass
    return "up"


def warmup() -> None:
    """Load the model and run one dummy forward pass (call at server start)."""
    try:
        from .encoding import NUM_CHANNELS
        _policy_logits(np.zeros((NUM_CHANNELS, 11, 11), dtype=np.float32))
    except Exception:
        pass
