"""Observation & action-mask encoding shared by training and inference.

The encoder works on a tiny canonical state so that both the fast self-play
env (rules.Game) and the Battlesnake JSON payload map onto the exact same
tensors.

Canonical state (all coordinates are Battlesnake-style, y grows up):
    width, height : int
    food          : iterable of (x, y)
    snakes        : list of dicts {"body": [(x,y), ...] (head first),
                                   "health": int, "alive": bool}
    me            : index into `snakes` of the point-of-view snake

Observation: float32 [C, H, W], indexed obs[c, y, x].

Channels:
    0  my head
    1  my body (all segments, incl. head)
    2  my tail cell
    3  enemy heads, snakes with len >= mine   (dangerous head-to-head)
    4  enemy heads, snakes with len <  mine   (winnable head-to-head)
    5  enemy bodies (all segments)
    6  time-to-vacate for every occupied cell, (steps till free)/20, clip 1
    7  food
    8  my health / 100                        (constant plane)
    9  (my len - longest enemy len) / 10, clip [-1, 1]   (constant plane)
    10 my len / 20, clip 1                    (constant plane)
    11 ones                                   (board mask)
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .rules import MOVE_DELTAS

NUM_CHANNELS = 12


def encode_obs(width: int, height: int, food, snakes: Sequence[Dict],
               me: int) -> np.ndarray:
    obs = np.zeros((NUM_CHANNELS, height, width), dtype=np.float32)
    my = snakes[me]
    my_len = len(my["body"])

    # snakes
    max_enemy = 0
    for idx, s in enumerate(snakes):
        if not s["alive"]:
            continue
        body = s["body"]
        L = len(body)
        hx, hy = body[0]
        if idx == me:
            obs[0, hy, hx] = 1.0
            body_ch, head_ch = 1, None
        else:
            max_enemy = max(max_enemy, L)
            body_ch = 5
            head_ch = 3 if L >= my_len else 4
            if 0 <= hx < width and 0 <= hy < height:
                obs[head_ch, hy, hx] = 1.0
        # stacked segments (after eating) share a cell; ttl = max steps
        for j, (x, y) in enumerate(body):
            if not (0 <= x < width and 0 <= y < height):
                continue
            obs[body_ch, y, x] = 1.0
            ttl = (L - j) / 20.0
            if ttl > obs[6, y, x]:
                obs[6, y, x] = min(ttl, 1.0)
        tx, ty = body[-1]
        if idx == me and 0 <= tx < width and 0 <= ty < height:
            obs[2, ty, tx] = 1.0

    for (x, y) in food:
        obs[7, y, x] = 1.0

    obs[8] = my["health"] / 100.0
    obs[9] = float(np.clip((my_len - max_enemy) / 10.0, -1.0, 1.0))
    obs[10] = min(my_len / 20.0, 1.0)
    obs[11] = 1.0
    return obs


def safe_action_mask(width: int, height: int, snakes: Sequence[Dict],
                     me: int) -> np.ndarray:
    """Boolean mask [4] of moves that are not *certainly* fatal:
    stays on the board and does not enter a cell that is guaranteed to still
    be occupied next turn (any body segment except a tail that will vacate).

    Head-to-head risks are NOT masked -- the policy learns those.
    If nothing is safe, all moves are allowed (death is inevitable anyway).
    """
    blocked = set()
    for s in snakes:
        if not s["alive"]:
            continue
        body = s["body"]
        # tail vacates next turn unless it is stacked (snake just ate)
        tail_vacates = len(body) < 2 or body[-1] != body[-2]
        last = len(body) - 1 if tail_vacates else len(body)
        for cell in body[:last]:
            blocked.add(cell)

    hx, hy = snakes[me]["body"][0]
    mask = np.zeros(4, dtype=bool)
    for a, (dx, dy) in enumerate(MOVE_DELTAS):
        nx, ny = hx + dx, hy + dy
        if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in blocked:
            mask[a] = True
    if not mask.any():
        mask[:] = True
    return mask


# --------------------------------------------------------------------------
# Battlesnake JSON -> canonical state
# --------------------------------------------------------------------------

def state_from_json(game_state: Dict) -> Tuple[int, int, List, List[Dict], int]:
    board = game_state["board"]
    width, height = board["width"], board["height"]
    food = [(f["x"], f["y"]) for f in board["food"]]
    my_id = game_state["you"]["id"]

    snakes: List[Dict] = []
    me = 0
    for s in board["snakes"]:
        body = [(p["x"], p["y"]) for p in s["body"]]
        if s["id"] == my_id:
            me = len(snakes)
        snakes.append({"body": body, "health": s["health"], "alive": True})
    return width, height, food, snakes, me
