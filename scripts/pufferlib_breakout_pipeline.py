#!/usr/bin/env python3
"""Headless PufferLib Breakout export/training helpers for Tokamech."""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import subprocess
import time
from typing import Any


DEFAULT_PUFFERLIB_DIR = "/home/claude/pathfinder"


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


def _save_run_metadata(out_dir: pathlib.Path, **values: Any) -> None:
    with (out_dir / "run.json").open("w", encoding="utf-8") as f:
        json.dump(values, f, indent=2, sort_keys=True)
        f.write("\n")


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

    p = sub.add_parser("standalone-demo-export")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-rows", type=int, default=100_000)
    p.add_argument("--steps", type=int, default=120_000)
    p.add_argument("--binary")
    p.set_defaults(func=standalone_demo_export)

    p = sub.add_parser("stats")
    p.add_argument("jsonl")
    p.add_argument("--manifest")
    p.add_argument("--out")
    p.set_defaults(func=stats_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
