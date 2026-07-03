"""Quick sanity evaluation: seat 0 = trained policy, other seats = a
"safe-random" baseline (uniform over non-suicidal moves).

    python -m ml.eval --games 200

Reports win rate and mean survival length for the trained snake. A strong
model should be well above the ~25% symmetric baseline within minutes of
training and reach >90% vs safe-random.
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

from .encoding import encode_obs, safe_action_mask
from .env import _snake_dicts
from .model import SnakeNet, masked_logits
from .rules import Game

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(WEIGHTS_DIR, "ckpt_latest.pt"))
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--num-snakes", type=int, default=4)
    p.add_argument("--board", type=int, default=11)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    cfg = ck.get("cfg", {"channels": 128, "blocks": 8, "board": args.board})
    net = SnakeNet(cfg["channels"], cfg["blocks"], cfg["board"]).to(device).eval()
    net.load_state_dict(ck["model"])

    rng = random.Random(123)
    wins, turns = 0, []
    for gi in range(args.games):
        g = Game(args.board, args.board, args.num_snakes,
                 rng=random.Random(1000 + gi))
        while not g.is_over() and g.turn < 512:
            snakes = _snake_dicts(g)
            moves = []
            for i, s in enumerate(g.snakes):
                if not s.alive:
                    moves.append(0)
                    continue
                mask = safe_action_mask(g.width, g.height, snakes, i)
                if i == 0:
                    obs = encode_obs(g.width, g.height, g.food, snakes, i)
                    with torch.no_grad():
                        logits, _ = net(torch.from_numpy(obs[None]).to(device))
                    logits = masked_logits(
                        logits.float(), torch.from_numpy(mask[None]).to(device))
                    moves.append(int(logits.argmax()))
                else:
                    moves.append(rng.choice(np.flatnonzero(mask).tolist()))
            g.step(moves)
            if not g.snakes[0].alive:
                break
        turns.append(g.turn)
        if g.snakes[0].alive and g.alive_count() == 1:
            wins += 1
    print(f"win rate vs safe-random: {wins/args.games:.1%} "
          f"({wins}/{args.games}), mean game length {np.mean(turns):.0f}")


if __name__ == "__main__":
    main()
