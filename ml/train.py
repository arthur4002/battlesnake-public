"""PPO self-play training for Battlesnake on a single NVIDIA GPU (A100).

Usage (on the A100 box):

    pip install -r requirements-train.txt
    python -m ml.train --minutes 75

Everything is wall-clock bounded: the run stops after --minutes, saving
checkpoints every --ckpt-every minutes and exporting the final model
(TorchScript + ONNX) to ml/weights/ automatically.

Design notes
------------
* Pure self-play: one network controls all 4 snakes in every game and is
  trained on every seat's transitions (~4x sample efficiency, symmetric).
* Env is CPU-bound -> sharded across worker processes; the GPU does batched
  forward passes for action selection and the PPO update.
* Rollout storage is [T, A]; dead agents are masked out via `valid`.
* bf16 autocast on the A100 for both rollout inference and updates.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from .env import SubprocVecBattlesnake, VecBattlesnake
from .model import SnakeNet, masked_logits

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=float, default=75.0,
                   help="wall-clock training budget")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--num-snakes", type=int, default=4)
    p.add_argument("--board", type=int, default=11)
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 8))
    p.add_argument("--rollout", type=int, default=96, help="steps per rollout")
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--ent-final", type=float, default=0.002)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--ckpt-every", type=float, default=5.0, help="minutes")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--resume", type=str, default="",
                   help="path to a .pt checkpoint to resume from")
    return p.parse_args()


@torch.no_grad()
def act(net, obs_t, mask_t, autocast_ctx):
    with autocast_ctx():
        logits, value = net(obs_t)
    logits = masked_logits(logits.float(), mask_t)
    dist = torch.distributions.Categorical(logits=logits)
    a = dist.sample()
    return a, dist.log_prob(a), value.float()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    autocast_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) \
        if use_bf16 else (lambda: torch.autocast("cpu", enabled=False))
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    print(f"device={device} bf16={use_bf16} cpus={os.cpu_count()}")

    # env ------------------------------------------------------------------
    env_cls = SubprocVecBattlesnake if args.workers > 1 else VecBattlesnake
    env_kwargs = dict(num_envs=args.num_envs, num_snakes=args.num_snakes,
                      size=args.board, seed=args.seed)
    if args.workers > 1:
        env_kwargs["num_workers"] = args.workers
    env = env_cls(**env_kwargs)
    A = env.num_agents
    C, H, W = env.obs_shape

    # model ----------------------------------------------------------------
    net = SnakeNet(args.channels, args.blocks, args.board).to(device)
    net = net.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, eps=1e-5,
                            weight_decay=0.0)
    start_iter = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_iter = ck.get("iter", 0)
        print(f"resumed from {args.resume} @ iter {start_iter}")

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # rollout buffers [T, A] -------------------------------------------------
    T = args.rollout
    b_obs = torch.zeros((T, A, C, H, W), dtype=torch.float16)
    b_mask = torch.zeros((T, A, 4), dtype=torch.bool)
    b_act = torch.zeros((T, A), dtype=torch.long)
    b_logp = torch.zeros((T, A))
    b_val = torch.zeros((T, A))
    b_rew = torch.zeros((T, A))
    b_done = torch.zeros((T, A))
    b_valid = torch.zeros((T, A), dtype=torch.bool)

    obs, mask, alive = env.reset()
    t_start = time.time()
    t_ckpt = t_start
    deadline = t_start + args.minutes * 60.0
    it = start_iter
    ep_rew = deque(maxlen=200)
    global_steps = 0

    while time.time() < deadline:
        it += 1
        # ---------------------------------------------------------- rollout
        for t in range(T):
            obs_t = torch.from_numpy(obs).to(device, non_blocking=True) \
                .to(memory_format=torch.channels_last)
            mask_t = torch.from_numpy(mask).to(device)
            a, logp, val = act(net, obs_t, mask_t, autocast_ctx)

            b_obs[t] = torch.from_numpy(obs).half()
            b_mask[t] = torch.from_numpy(mask)
            b_act[t] = a.cpu()
            b_logp[t] = logp.cpu()
            b_val[t] = val.cpu()
            b_valid[t] = torch.from_numpy(alive)

            obs, mask, alive, rew, done, valid = env.step(a.cpu().numpy())
            b_rew[t] = torch.from_numpy(rew)
            b_done[t] = torch.from_numpy(done.astype(np.float32))
            # `valid` from env == alive before the step; keep the stricter of
            # the two (they coincide, but be safe)
            b_valid[t] &= torch.from_numpy(valid)
            global_steps += int(valid.sum())
            ep_rew.append(float(rew[valid].sum()) / max(1, int(valid.sum())))

        # bootstrap value for the last obs
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).to(device) \
                .to(memory_format=torch.channels_last)
            with autocast_ctx():
                _, last_val = net(obs_t)
            last_val = last_val.float().cpu()

        # -------------------------------------------------------------- GAE
        adv = torch.zeros((T, A))
        gae = torch.zeros(A)
        next_val = last_val
        for t in reversed(range(T)):
            nonterminal = 1.0 - b_done[t]
            delta = b_rew[t] + args.gamma * next_val * nonterminal - b_val[t]
            gae = delta + args.gamma * args.gae_lambda * nonterminal * gae
            # kill credit flowing through invalid (dead) slots
            gae = gae * b_valid[t].float()
            adv[t] = gae
            next_val = b_val[t]
        ret = adv + b_val

        # ------------------------------------------------------- PPO update
        flat_valid = b_valid.reshape(-1)
        idx_valid = flat_valid.nonzero(as_tuple=True)[0]
        f_obs = b_obs.reshape(-1, C, H, W)[idx_valid]
        f_mask = b_mask.reshape(-1, 4)[idx_valid]
        f_act = b_act.reshape(-1)[idx_valid]
        f_logp = b_logp.reshape(-1)[idx_valid]
        f_adv = adv.reshape(-1)[idx_valid]
        f_ret = ret.reshape(-1)[idx_valid]
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)

        n = f_obs.shape[0]
        mb = max(1, n // args.minibatches)
        frac = min(1.0, (time.time() - t_start) / (args.minutes * 60.0))
        ent_coef = args.ent_coef + (args.ent_final - args.ent_coef) * frac
        for g in opt.param_groups:
            g["lr"] = args.lr * (1.0 - 0.9 * frac)  # linear decay to 10%

        pl = vl = el = 0.0
        nupd = 0
        for _ in range(args.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, mb):
                j = perm[s:s + mb]
                o = f_obs[j].to(device, non_blocking=True).float() \
                    .to(memory_format=torch.channels_last)
                m = f_mask[j].to(device)
                with autocast_ctx():
                    logits, value = net(o)
                logits = masked_logits(logits.float(), m)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(f_act[j].to(device))
                ratio = (logp - f_logp[j].to(device)).exp()
                a_ = f_adv[j].to(device)
                pg1 = -a_ * ratio
                pg2 = -a_ * ratio.clamp(1 - args.clip, 1 + args.clip)
                policy_loss = torch.max(pg1, pg2).mean()
                v_loss = 0.5 * (value.float() - f_ret[j].to(device)).pow(2).mean()
                ent = dist.entropy().mean()
                loss = policy_loss + args.vf_coef * v_loss - ent_coef * ent

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                opt.step()
                pl += policy_loss.item(); vl += v_loss.item()
                el += ent.item(); nupd += 1

        # ------------------------------------------------------------- logs
        el_m = (time.time() - t_start) / 60.0
        sps = global_steps / max(1e-9, time.time() - t_start)
        print(f"iter {it:4d} | {el_m:6.1f} min | steps {global_steps/1e6:6.2f}M "
              f"| {sps/1e3:6.1f}k sps | pi {pl/nupd:+.4f} v {vl/nupd:.4f} "
              f"ent {el/nupd:.3f} | r/step {np.mean(ep_rew):+.4f} "
              f"| lr {opt.param_groups[0]['lr']:.1e} ent_c {ent_coef:.4f}",
              flush=True)

        # ------------------------------------------------------ checkpoints
        if time.time() - t_ckpt > args.ckpt_every * 60.0:
            t_ckpt = time.time()
            save_ckpt(net, opt, it, args)

    save_ckpt(net, opt, it, args)
    export(net, args)
    if hasattr(env, "close"):
        env.close()
    print("done.")


def save_ckpt(net, opt, it, args):
    path = os.path.join(WEIGHTS_DIR, "ckpt_latest.pt")
    torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                "iter": it,
                "cfg": {"channels": args.channels, "blocks": args.blocks,
                        "board": args.board}}, path)
    print(f"[ckpt] saved {path}", flush=True)


def export(net, args):
    """Export CPU inference artifacts: TorchScript + ONNX."""
    from .export import export_model
    export_model(net.cpu().eval(), args.board, WEIGHTS_DIR)


if __name__ == "__main__":
    main()
