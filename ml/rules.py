"""Battlesnake standard rules engine (single game).

Faithful to the official Go rules (BattlesnakeOfficial/rules, "standard"):
  1. Move snakes: prepend new head, drop last body segment. Health -1.
  2. Feed: if head is on food -> health = 100, duplicate the tail segment
     (so the snake grows by one and its tail is "stacked" for one turn).
  3. Spawn food: keep at least MIN_FOOD on the board, otherwise spawn one
     with probability FOOD_SPAWN_CHANCE on a random unoccupied cell.
  4. Eliminations (simultaneous, computed on post-move state):
       - health <= 0
       - head out of bounds
       - head on own body (any non-head segment)
       - head on another snake's non-head segment
       - head-to-head: the shorter snake dies, equal lengths -> both die.

Coordinates follow the Battlesnake API: (x, y), y grows UP.
Moves: 0=up(+y) 1=down(-y) 2=left(-x) 3=right(+x).
"""

from __future__ import annotations

import random
from typing import List, Optional, Set, Tuple

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
MOVE_DELTAS = ((0, 1), (0, -1), (-1, 0), (1, 0))
MOVE_NAMES = ("up", "down", "left", "right")

MAX_HEALTH = 100
FOOD_SPAWN_CHANCE = 0.15
MIN_FOOD = 1


class Snake:
    __slots__ = ("body", "health", "alive", "ate")

    def __init__(self, body: List[Tuple[int, int]]):
        self.body: List[Tuple[int, int]] = list(body)  # body[0] is the head
        self.health: int = MAX_HEALTH
        self.alive: bool = True
        self.ate: bool = False

    def __len__(self) -> int:
        return len(self.body)

    @property
    def head(self) -> Tuple[int, int]:
        return self.body[0]


class Game:
    """One Battlesnake game with `num_snakes` snakes on a WxH board."""

    def __init__(self, width: int = 11, height: int = 11, num_snakes: int = 4,
                 rng: Optional[random.Random] = None):
        self.width = width
        self.height = height
        self.num_snakes = num_snakes
        self.rng = rng or random.Random()
        self.snakes: List[Snake] = []
        self.food: Set[Tuple[int, int]] = set()
        self.turn = 0
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        self.turn = 0
        self.food = set()
        w, h = self.width, self.height
        mn, md, mx = 1, w // 2, w - 2  # 1, 5, 9 on a 11x11 board
        corners = [(mn, mn), (mn, mx), (mx, mn), (mx, mx)]
        cardinals = [(mn, md), (md, mn), (md, mx), (mx, md)]
        self.rng.shuffle(corners)
        self.rng.shuffle(cardinals)
        starts = (corners + cardinals)[: self.num_snakes] \
            if self.num_snakes <= 4 else self._random_starts()
        # official engine picks either all-corners or all-cardinals; corners
        # is the common competitive layout, keep it but shuffle assignment
        self.snakes = [Snake([p, p, p]) for p in starts[: self.num_snakes]]

        # one food near each snake (official: adjacent diagonal-ish cell) --
        # simplified: random free cell within manhattan distance 2, plus centre
        occupied = {s.head for s in self.snakes}
        centre = (w // 2, h // 2)
        for s in self.snakes:
            cands = [
                (x, y)
                for x in range(w) for y in range(h)
                if abs(x - s.head[0]) + abs(y - s.head[1]) == 2
                and (x, y) not in occupied and (x, y) not in self.food
                and (x, y) != centre
            ]
            if cands:
                self.food.add(self.rng.choice(cands))
        if centre not in occupied:
            self.food.add(centre)

    def _random_starts(self) -> List[Tuple[int, int]]:
        cells = [(x, y) for x in range(self.width) for y in range(self.height)]
        self.rng.shuffle(cells)
        return cells

    # ------------------------------------------------------------------- step
    def step(self, moves: List[int]) -> Tuple[List[float], List[bool]]:
        """Advance one turn. `moves[i]` is the move for snake i (ignored if
        dead). Returns (ate_flags, died_this_turn) per snake."""
        self.turn += 1
        alive_idx = [i for i, s in enumerate(self.snakes) if s.alive]

        # 1. move + health
        for i in alive_idx:
            s = self.snakes[i]
            dx, dy = MOVE_DELTAS[moves[i]]
            hx, hy = s.head
            s.body.insert(0, (hx + dx, hy + dy))
            s.body.pop()
            s.health -= 1
            s.ate = False

        # 2. feed (all heads checked against pre-spawn food, simultaneously)
        eaten: Set[Tuple[int, int]] = set()
        for i in alive_idx:
            s = self.snakes[i]
            if s.head in self.food:
                eaten.add(s.head)
                s.health = MAX_HEALTH
                s.body.append(s.body[-1])  # duplicate tail -> grow
                s.ate = True
        self.food -= eaten

        # 3. spawn food
        self._spawn_food()

        # 4. eliminations (simultaneous)
        died = [False] * self.num_snakes
        w, h = self.width, self.height
        for i in alive_idx:
            s = self.snakes[i]
            x, y = s.head
            if s.health <= 0 or x < 0 or x >= w or y < 0 or y >= h:
                died[i] = True
                continue
            # body collisions (own + others'), heads excluded
            for j in alive_idx:
                if s.head in self.snakes[j].body[1:]:
                    died[i] = True
                    break
            if died[i]:
                continue
            # head-to-head
            for j in alive_idx:
                if j != i and self.snakes[j].head == s.head \
                        and len(self.snakes[j]) >= len(s):
                    died[i] = True
                    break

        for i in alive_idx:
            if died[i]:
                self.snakes[i].alive = False

        return [s.ate for s in self.snakes], died

    def _spawn_food(self) -> None:
        need = len(self.food) < MIN_FOOD
        if not need and self.rng.random() >= FOOD_SPAWN_CHANCE:
            return
        occupied = set(self.food)
        for s in self.snakes:
            if s.alive:
                occupied.update(s.body)
        free = [
            (x, y)
            for x in range(self.width) for y in range(self.height)
            if (x, y) not in occupied
        ]
        if free:
            self.food.add(self.rng.choice(free))

    # ------------------------------------------------------------- utilities
    def alive_count(self) -> int:
        return sum(s.alive for s in self.snakes)

    def is_over(self) -> bool:
        # multi-snake game ends when <=1 snake remains; solo when 0
        limit = 1 if self.num_snakes > 1 else 0
        return self.alive_count() <= limit
