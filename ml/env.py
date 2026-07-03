"""Vectorized self-play Battlesnake environment for PPO.

Agents = num_envs * num_snakes. Every agent is controlled by the (same)
policy; the trainer receives observations/rewards for every snake, so a
single network learns from all seats simultaneously (pure self-play).

Reward per snake:
    +1.0                       won the game (last snake alive)
    -1.0                       died
    +REW_FOOD                  ate food
    +REW_STEP                  survived the turn
    0 on truncation (max turns reached) -- treated as a draw.

Auto-reset: when a game ends every agent in that env comes back to life on
the next step. `valid` marks agents that were alive at the *beginning* of a
step (their transition should be trained on).
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np

from .encoding import NUM_CHANNELS, encode_obs, safe_action_mask
from .rules import Game

REW_WIN = 1.0
REW_DEATH = -1.0
REW_FOOD = 0.03
REW_STEP = 0.002
MAX_TURNS = 512


def _snake_dicts(game: Game) -> List[Dict]:
    return [{"body": s.body, "health": s.health, "alive": s.alive}
            for s in game.snakes]


class VecBattlesnake:
    """`num_envs` independent games stepped in lock-step."""

    def __init__(self, num_envs: int, num_snakes: int = 4, size: int = 11,
                 seed: int = 0):
        self.num_envs = num_envs
        self.num_snakes = num_snakes
        self.size = size
        self.games = [
            Game(size, size, num_snakes, rng=random.Random(seed * 100003 + i))
            for i in range(num_envs)
        ]
        self.num_agents = num_envs * num_snakes
        self.obs_shape = (NUM_CHANNELS, size, size)

    # ------------------------------------------------------------------ api
    def reset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        for g in self.games:
            g.reset()
        return self._observe()

    def step(self, actions: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                        np.ndarray, np.ndarray, np.ndarray]:
        """actions: int array [num_agents].

        Returns:
            obs      float32 [A, C, H, W]  (post-step / post-reset)
            masks    bool    [A, 4]
            alive    bool    [A]           snake alive after the step
            rewards  float32 [A]
            dones    bool    [A]           terminal for this agent this step
            valid    bool    [A]           agent was alive before the step
        """
        A = self.num_agents
        rewards = np.zeros(A, dtype=np.float32)
        dones = np.zeros(A, dtype=bool)
        valid = np.zeros(A, dtype=bool)

        acts = actions.reshape(self.num_envs, self.num_snakes)
        for e, g in enumerate(self.games):
            base = e * self.num_snakes
            pre_alive = [s.alive for s in g.snakes]
            for i, a in enumerate(pre_alive):
                valid[base + i] = a

            ate, died = g.step(list(acts[e]))

            game_over = g.is_over() or g.turn >= MAX_TURNS
            for i in range(self.num_snakes):
                if not pre_alive[i]:
                    continue
                r = 0.0
                if died[i]:
                    r += REW_DEATH
                    dones[base + i] = True
                else:
                    r += REW_STEP
                    if ate[i]:
                        r += REW_FOOD
                    if game_over:
                        dones[base + i] = True
                        if self.num_snakes > 1 and g.alive_count() == 1:
                            r += REW_WIN
                        # truncation / draw: no terminal bonus
                rewards[base + i] = r

            if game_over:
                g.reset()

        obs, masks, alive = self._observe()
        return obs, masks, alive, rewards, dones, valid

    # ------------------------------------------------------------- internals
    def _observe(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        A = self.num_agents
        obs = np.zeros((A,) + self.obs_shape, dtype=np.float32)
        masks = np.ones((A, 4), dtype=bool)
        alive = np.zeros(A, dtype=bool)
        for e, g in enumerate(self.games):
            snakes = _snake_dicts(g)
            base = e * self.num_snakes
            for i, s in enumerate(g.snakes):
                if not s.alive:
                    continue
                alive[base + i] = True
                obs[base + i] = encode_obs(g.width, g.height, g.food,
                                           snakes, i)
                masks[base + i] = safe_action_mask(g.width, g.height,
                                                   snakes, i)
        return obs, masks, alive


# --------------------------------------------------------------------------
# Multiprocessing wrapper: shards envs across worker processes.
# --------------------------------------------------------------------------

def _worker(remote, num_envs, num_snakes, size, seed):
    env = VecBattlesnake(num_envs, num_snakes, size, seed)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                remote.send(env.reset())
            elif cmd == "step":
                remote.send(env.step(data))
            elif cmd == "close":
                break
    finally:
        remote.close()


class SubprocVecBattlesnake:
    """Same interface as VecBattlesnake, envs sharded over processes."""

    def __init__(self, num_envs: int, num_snakes: int = 4, size: int = 11,
                 seed: int = 0, num_workers: int = 8):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        num_workers = min(num_workers, num_envs)
        shard = [num_envs // num_workers +
                 (1 if i < num_envs % num_workers else 0)
                 for i in range(num_workers)]
        self.shard_agents = [n * num_snakes for n in shard]
        self.num_envs = num_envs
        self.num_snakes = num_snakes
        self.num_agents = num_envs * num_snakes
        self.obs_shape = (NUM_CHANNELS, size, size)

        self.remotes, self.procs = [], []
        for i, n in enumerate(shard):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker,
                            args=(child, n, num_snakes, size, seed + i * 7919),
                            daemon=True)
            p.start()
            child.close()
            self.remotes.append(parent)
            self.procs.append(p)

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        return self._gather([r.recv() for r in self.remotes])

    def step(self, actions: np.ndarray):
        i = 0
        for r, n in zip(self.remotes, self.shard_agents):
            r.send(("step", actions[i:i + n]))
            i += n
        return self._gather([r.recv() for r in self.remotes])

    @staticmethod
    def _gather(parts):
        return tuple(np.concatenate([p[k] for p in parts])
                     for k in range(len(parts[0])))

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for p in self.procs:
            p.join(timeout=2)
