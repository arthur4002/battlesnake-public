"""Residual CNN policy + value network (AlphaZero-style trunk, PPO heads).

Small enough to run in a few milliseconds on a single CPU core at inference
time (Render free tier), big enough to saturate learning within ~1 hour of
self-play on an A100.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoding import NUM_CHANNELS

NUM_ACTIONS = 4


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.n1 = nn.GroupNorm(8, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.n2 = nn.GroupNorm(8, ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return self.act(x + h)


class SnakeNet(nn.Module):
    def __init__(self, channels: int = 128, blocks: int = 8,
                 board: int = 11, in_ch: int = NUM_CHANNELS):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 8, 1, bias=False),
            nn.GroupNorm(4, 8),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * board * board, NUM_ACTIONS),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 4, 1, bias=False),
            nn.GroupNorm(4, 4),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(4 * board * board, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor):
        """x: [B, C, H, W] -> (logits [B, 4], value [B])."""
        h = self.trunk(self.stem(x))
        return self.policy_head(h), self.value_head(h).squeeze(-1)


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set logits of masked-out actions to -inf-ish."""
    return logits.masked_fill(~mask, -1e9)
