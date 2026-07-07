#!/usr/bin/env python3
"""Headless PufferLib Breakout export/training helpers for Tokamech."""

from __future__ import annotations

import argparse
import collections
import contextlib
import glob
import json
import math
import os
import pathlib
import subprocess
import sys
import time
from typing import Any


DEFAULT_PUFFERLIB_DIR = "/home/claude/pathfinder"
DEFAULT_WEIGHTS = "resources/breakout/breakout_weights.bin"


def _repo_imports(pufferlib_dir: str):
    sys.path.insert(0, pufferlib_dir)
    from pufferlib import _C  # pylint: disable=import-error,import-outside-toplevel
    from pufferlib import pufferl  # pylint: disable=import-error,import-outside-toplevel

    return _C, pufferl


@contextlib.contextmanager
def _clean_argv():
    original = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        yield
    finally:
        sys.argv = original


def _load_args(pufferlib_dir: str, total_timesteps: int, total_agents: int, horizon: int,
               num_buffers: int, num_threads: int, checkpoint_dir: str, log_dir: str,
               seed: int) -> tuple[Any, dict[str, Any]]:
    backend, pufferl = _repo_imports(pufferlib_dir)
    with _clean_argv():
        args = pufferl.load_config("breakout")

    args["rank"] = 0
    args["world_size"] = 1
    args["gpu_id"] = 0
    args["nccl_id"] = b""
    args["cudagraphs"] = -1
    args["checkpoint_dir"] = checkpoint_dir
    args["log_dir"] = log_dir
    args["seed"] = seed
    args["reset_state"] = True
    args["vec"]["total_agents"] = total_agents
    args["vec"]["num_buffers"] = num_buffers
    args["vec"]["num_threads"] = num_threads
    args["train"]["gpus"] = 1
    args["train"]["total_timesteps"] = total_timesteps
    args["train"]["horizon"] = horizon
    args["train"]["seed"] = seed
    if args["train"]["minibatch_size"] > total_agents * horizon:
        args["train"]["minibatch_size"] = total_agents * horizon
    return backend, args


def _set_export_env(out_dir: pathlib.Path, source: str, max_rows: int) -> tuple[pathlib.Path, pathlib.Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "transitions-000000.jsonl"
    manifest = out_dir / "manifest.json"
    os.environ["PUFFERLIB_BREAKOUT_EXPORT_JSONL"] = str(jsonl)
    os.environ["PUFFERLIB_BREAKOUT_EXPORT_MANIFEST"] = str(manifest)
    os.environ["PUFFERLIB_BREAKOUT_EXPORT_SOURCE"] = source
    os.environ["PUFFERLIB_BREAKOUT_EXPORT_MAX_ROWS"] = str(max_rows)
    return jsonl, manifest


def _line_count(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("rb") as f:
        for _ in f:
            count += 1
    return count


def _latest_checkpoint(checkpoint_dir: pathlib.Path) -> pathlib.Path:
    candidates = [pathlib.Path(p) for p in glob.glob(str(checkpoint_dir / "breakout" / "**" / "*.bin"), recursive=True)]
    if not candidates:
        raise FileNotFoundError(f"no checkpoint found under {checkpoint_dir / 'breakout'}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _save_run_metadata(out_dir: pathlib.Path, **values: Any) -> None:
    with (out_dir / "run.json").open("w", encoding="utf-8") as f:
        json.dump(values, f, indent=2, sort_keys=True)
        f.write("\n")


def _flatten_log(logs: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in logs.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten_log(value, name))
        elif isinstance(value, (int, float)):
            out[name] = float(value)
    return out


def pretrained_export(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.out_dir)
    jsonl, manifest = _set_export_env(out_dir, "pretrained_export", args.max_rows)
    checkpoint_dir = str(out_dir / "checkpoints")
    log_dir = str(out_dir / "logs")
    backend, cfg = _load_args(
        args.pufferlib_dir, args.total_timesteps, args.total_agents, args.horizon,
        args.num_buffers, args.num_threads, checkpoint_dir, log_dir, args.seed)
    pufferl = backend.create_pufferl(cfg)
    weights = pathlib.Path(args.pufferlib_dir) / args.weights
    backend.load_weights(pufferl, str(weights))
    start = time.time()
    rollouts = 0
    try:
        while _line_count(jsonl) < args.max_rows:
            backend.rollouts(pufferl)
            rollouts += 1
            if rollouts >= args.max_rollouts:
                break
    finally:
        backend.close(pufferl)

    stats = dataset_stats(jsonl, manifest if manifest.exists() else None)
    _save_run_metadata(
        out_dir,
        mode="pretrained-export",
        pufferlib_dir=args.pufferlib_dir,
        weights=str(weights),
        max_rows=args.max_rows,
        rollouts=rollouts,
        elapsed_seconds=time.time() - start,
        jsonl=str(jsonl),
        manifest=str(manifest),
        stats=stats,
    )
    print(json.dumps(stats, sort_keys=True))


def standalone_demo_export(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.out_dir)
    jsonl, manifest = _set_export_env(out_dir, "demo_export", args.max_rows)
    binary = pathlib.Path(args.binary) if args.binary else pathlib.Path(args.pufferlib_dir) / "breakout"
    if not binary.exists():
        raise FileNotFoundError(f"missing standalone breakout binary: {binary}")

    env = os.environ.copy()
    env["PUFFERLIB_BREAKOUT_DEMO_EXPORT_STEPS"] = str(args.steps)
    start = time.time()
    subprocess.run(
        [str(binary)],
        cwd=args.pufferlib_dir,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stats = dataset_stats(jsonl, manifest if manifest.exists() else None)
    _save_run_metadata(
        out_dir,
        mode="standalone-demo-export",
        pufferlib_dir=args.pufferlib_dir,
        binary=str(binary),
        steps=args.steps,
        max_rows=args.max_rows,
        elapsed_seconds=time.time() - start,
        jsonl=str(jsonl),
        manifest=str(manifest),
        stats=stats,
    )
    print(json.dumps(stats, sort_keys=True))


def checkpoint_export(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.out_dir)
    jsonl, manifest = _set_export_env(
        out_dir,
        "trained_checkpoint_greedy_export" if args.greedy else "trained_checkpoint_export",
        args.max_rows)
    binary = pathlib.Path(args.binary)
    if not binary.exists():
        raise FileNotFoundError(f"missing checkpoint exporter binary: {binary}")

    env = os.environ.copy()
    env["TKM_PUFFERLIB_BREAKOUT_FRAMESKIP"] = str(args.frameskip)
    env["TKM_PUFFERLIB_BREAKOUT_EXPORT_GREEDY"] = "1" if args.greedy else "0"
    start = time.time()
    subprocess.run(
        [str(binary), args.weights, str(args.steps)],
        cwd=args.cwd,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stats = dataset_stats(jsonl, manifest if manifest.exists() else None)
    _save_run_metadata(
        out_dir,
        mode="checkpoint-export",
        weights=args.weights,
        binary=str(binary),
        frameskip=args.frameskip,
        greedy=args.greedy,
        steps=args.steps,
        max_rows=args.max_rows,
        elapsed_seconds=time.time() - start,
        jsonl=str(jsonl),
        manifest=str(manifest),
        stats=stats,
    )
    print(json.dumps(stats, sort_keys=True))


def train_then_export(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.out_dir)
    checkpoint_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    backend, cfg = _load_args(
        args.pufferlib_dir, args.total_timesteps, args.total_agents, args.horizon,
        args.num_buffers, args.num_threads, str(checkpoint_dir), str(log_dir), args.seed)
    cfg["checkpoint_interval"] = max(1, args.checkpoint_interval)
    pufferl = backend.create_pufferl(cfg)
    train_epochs = max(1, math.ceil(args.total_timesteps / (args.total_agents * args.horizon)))
    start = time.time()
    flat_logs: dict[str, float] = {}
    latest_path = ""
    try:
        for epoch in range(train_epochs):
            backend.rollouts(pufferl)
            backend.train(pufferl)
            if epoch % args.log_interval == 0 or epoch == train_epochs - 1:
                flat_logs = _flatten_log(backend.log(pufferl))
                print(json.dumps({
                    "phase": "train",
                    "epoch": epoch,
                    "epochs": train_epochs,
                    "global_step": int(pufferl.global_step),
                    "score": flat_logs.get("env/score"),
                    "episode_return": flat_logs.get("env/episode_return"),
                    "sps": flat_logs.get("SPS"),
                }, sort_keys=True), flush=True)
            if epoch % args.checkpoint_interval == 0 or epoch == train_epochs - 1:
                ckpt_dir = checkpoint_dir / "breakout" / str(int(1000 * time.time()))
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                latest_path = str(ckpt_dir / f"{int(pufferl.global_step):016d}.bin")
                backend.save_weights(pufferl, latest_path)
    finally:
        backend.close(pufferl)

    if not latest_path:
        latest_path = str(_latest_checkpoint(checkpoint_dir))

    jsonl, manifest = _set_export_env(out_dir, "trained_policy_export", args.max_rows)
    backend, export_cfg = _load_args(
        args.pufferlib_dir, args.total_timesteps, args.total_agents, args.horizon,
        args.num_buffers, args.num_threads, str(checkpoint_dir), str(log_dir), args.seed)
    export_cfg["reset_state"] = False
    pufferl = backend.create_pufferl(export_cfg)
    backend.load_weights(pufferl, latest_path)
    rollouts = 0
    try:
        while _line_count(jsonl) < args.max_rows:
            backend.rollouts(pufferl)
            rollouts += 1
            if rollouts >= args.max_rollouts:
                break
    finally:
        backend.close(pufferl)

    stats = dataset_stats(jsonl, manifest if manifest.exists() else None)
    _save_run_metadata(
        out_dir,
        mode="train-then-export",
        pufferlib_dir=args.pufferlib_dir,
        total_timesteps=args.total_timesteps,
        total_agents=args.total_agents,
        horizon=args.horizon,
        checkpoint=latest_path,
        rollouts=rollouts,
        elapsed_seconds=time.time() - start,
        last_train_log=flat_logs,
        jsonl=str(jsonl),
        manifest=str(manifest),
        stats=stats,
    )
    print(json.dumps(stats, sort_keys=True))


def dataset_stats(jsonl: pathlib.Path, manifest: pathlib.Path | None = None) -> dict[str, Any]:
    actions: collections.Counter[int] = collections.Counter()
    terminals = 0
    reward_sum = 0.0
    row_count = 0
    prelaunch_rows = 0
    obs_dim_bad = 0
    obs_len_bad = 0
    first_t_actions: collections.Counter[int] = collections.Counter()
    t0_hashes: collections.Counter[tuple[float, ...]] = collections.Counter()
    episode_returns: dict[tuple[int, int], float] = collections.defaultdict(float)
    completed_returns: list[float] = []

    with jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            row_count += 1
            obs = row.get("obs", [])
            action = int(row.get("action", -1))
            reward = float(row.get("reward", 0.0))
            terminal = int(row.get("terminal", 0))
            env_index = int(row.get("env_index", -1))
            episode = int(row.get("episode", -1))
            t = int(row.get("t", -1))
            actions[action] += 1
            terminals += 1 if terminal else 0
            reward_sum += reward
            if row.get("obs_dim") != 118:
                obs_dim_bad += 1
            if len(obs) != 118:
                obs_len_bad += 1
            if len(obs) >= 7 and obs[4] == 0 and obs[5] == 0 and obs[6] == 0:
                prelaunch_rows += 1
            if t == 0:
                first_t_actions[action] += 1
                if len(obs) == 118:
                    t0_hashes[tuple(float(x) for x in obs)] += 1
            episode_returns[(env_index, episode)] += reward
            if terminal:
                completed_returns.append(episode_returns[(env_index, episode)])

    manifest_json: dict[str, Any] | None = None
    if manifest is not None and manifest.exists():
        with manifest.open("r", encoding="utf-8") as f:
            manifest_json = json.load(f)

    returns = list(episode_returns.values())
    top_t0 = t0_hashes.most_common(1)
    return {
        "jsonl": str(jsonl),
        "row_count": row_count,
        "actions": {str(k): v for k, v in sorted(actions.items())},
        "reward_sum": reward_sum,
        "terminal_count": terminals,
        "episode_count_reconstructed": len(episode_returns),
        "completed_episode_count": len(completed_returns),
        "mean_reconstructed_return": sum(returns) / len(returns) if returns else 0.0,
        "max_reconstructed_return": max(returns) if returns else 0.0,
        "prelaunch_rows": prelaunch_rows,
        "obs_dim_bad": obs_dim_bad,
        "obs_len_bad": obs_len_bad,
        "t0_unique_obs": len(t0_hashes),
        "t0_largest_duplicate": top_t0[0][1] if top_t0 else 0,
        "t0_actions": {str(k): v for k, v in sorted(first_t_actions.items())},
        "manifest": manifest_json,
    }


def stats_command(args: argparse.Namespace) -> None:
    jsonl = pathlib.Path(args.jsonl)
    manifest = pathlib.Path(args.manifest) if args.manifest else jsonl.parent / "manifest.json"
    stats = dataset_stats(jsonl, manifest if manifest.exists() else None)
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, sort_keys=True)
            f.write("\n")
    print(json.dumps(stats, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pufferlib-dir", default=DEFAULT_PUFFERLIB_DIR)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_runtime(p: argparse.ArgumentParser) -> None:
        p.add_argument("--out-dir", required=True)
        p.add_argument("--max-rows", type=int, default=100_000)
        p.add_argument("--max-rollouts", type=int, default=64)
        p.add_argument("--total-timesteps", type=int, default=1_048_576)
        p.add_argument("--total-agents", type=int, default=4096)
        p.add_argument("--horizon", type=int, default=64)
        p.add_argument("--num-buffers", type=int, default=8)
        p.add_argument("--num-threads", type=int, default=8)
        p.add_argument("--seed", type=int, default=73)

    p = sub.add_parser("pretrained-export")
    add_runtime(p)
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.set_defaults(func=pretrained_export)

    p = sub.add_parser("standalone-demo-export")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-rows", type=int, default=100_000)
    p.add_argument("--steps", type=int, default=120_000)
    p.add_argument("--binary")
    p.set_defaults(func=standalone_demo_export)

    p = sub.add_parser("checkpoint-export")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--binary", default="build/pufferlib_breakout_checkpoint_export")
    p.add_argument("--cwd", default=".")
    p.add_argument("--max-rows", type=int, default=100_000)
    p.add_argument("--steps", type=int, default=140_000)
    p.add_argument("--frameskip", type=int, default=4)
    p.add_argument("--greedy", action="store_true")
    p.set_defaults(func=checkpoint_export)

    p = sub.add_parser("train-then-export")
    add_runtime(p)
    p.add_argument("--checkpoint-interval", type=int, default=8)
    p.add_argument("--log-interval", type=int, default=4)
    p.set_defaults(func=train_then_export)

    p = sub.add_parser("stats")
    p.add_argument("jsonl")
    p.add_argument("--manifest")
    p.add_argument("--out")
    p.set_defaults(func=stats_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
