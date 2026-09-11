#!/usr/bin/env python3
"""Record your own slither.io play as demonstration episodes for the bot.

Opens a playable Chrome window, lets you steer with the mouse (click or space
to boost) and, in the background, samples the game at the bot's own cadence:
every frame-skip window it builds the same observation the bot sees, labels
what you did with the nearest discrete action, and scores it with the reward
function of the chosen curriculum stage. Each episode is saved as one file
under --out; feed the folder to the trainer with `--demos DIR`.

    uv run python record_demo.py --stage 1 --episodes 5
    uv run python trainer.py --demos demos --pretrain-steps 3000 --stage 1

Ctrl+C stops after saving the episode in progress (if it is long enough).
"""
import argparse
import os
import sys
import time

from action_space import ACTION_NAMES
from config import Config
from demo import DEFAULT_DEMO_DIR, DemoEpisode
from styles import STYLES


def resolve_style(name):
    """Same loose match trainer.py uses for --style-name."""
    if not name:
        return "Standard (Curriculum)"
    for s in STYLES:
        if name.lower() in s.lower():
            return s
    raise SystemExit(f"style '{name}' not found; choose from: {', '.join(STYLES)}")


def stage_config(style_name, stage):
    style = STYLES[style_name]
    if style["type"] == "static":
        return style["config"], 0
    stages = style["stages"]
    if stage not in stages:
        raise SystemExit(f"stage {stage} not in {style_name} (have {sorted(stages)})")
    return stages[stage], stage


def record_episode(env, max_steps, on_step=None):
    """Play one episode passively.

    Returns (DemoEpisode, cause, truncated, interrupted). `truncated` is True
    when the episode did not end in a death (step cap or Ctrl+C);
    `interrupted` is True only for Ctrl+C, so the caller can stop.
    """
    obs = env.reset()
    ep = DemoEpisode(obs)
    cause = None
    try:
        while len(ep) < max_steps:
            next_obs, reward, done, info = env.step(None)
            if 'human_action' not in info:
                # Invalid frame: no label, same observation — nothing to record.
                if done:
                    cause = info.get('cause')
                    break
                continue
            ep.add(info['human_action'], reward, next_obs, done, length=info.get('length', 0))
            if on_step:
                on_step(ep, info)
            if done:
                cause = info.get('cause')
                return ep, cause, False, False
    except KeyboardInterrupt:
        return ep, None, True, True
    return ep, cause, True, False


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=DEFAULT_DEMO_DIR, help="Folder to write demo files into (default: demos/)")
    p.add_argument("--episodes", type=int, default=0, help="Stop after N saved episodes (default: until Ctrl+C)")
    p.add_argument("--stage", type=int, default=1, help="Curriculum stage whose rewards score the demo (default: 1)")
    p.add_argument("--style-name", type=str, default=None, help="Learning style (default: Standard curriculum)")
    p.add_argument("--min-steps", type=int, default=30, help="Discard episodes shorter than this (default: 30)")
    p.add_argument("--max-steps", type=int, default=0, help="Cap episode length (default: the stage's max_steps)")
    p.add_argument("--nickname", type=str, default="Human", help="In-game nickname")
    p.add_argument("--url", type=str, default="http://slither.io", help="Game URL")
    p.add_argument("--vision-size", type=int, default=0, help="Matrix size override (default: from config)")
    args = p.parse_args(argv)

    cfg = Config()
    if args.vision_size:
        cfg.env.resolution = (args.vision_size, args.vision_size)
    style_name = resolve_style(args.style_name)
    stage_cfg, stage = stage_config(style_name, args.stage)
    max_steps = args.max_steps or int(stage_cfg.get("max_steps", 1000))

    from slither_env import SlitherEnv  # heavy import: after arg validation
    env = SlitherEnv(
        headless=False, nickname=args.nickname, matrix_size=cfg.env.resolution[0],
        base_url=args.url, frame_skip=cfg.env.frame_skip, backend="selenium",
        human_control=True,
    )
    env.set_curriculum_stage(stage_cfg)

    print(f"Recording to {args.out}/ | style={style_name} stage={stage} ({stage_cfg.get('name', 'static')}) "
          f"max_steps={max_steps} frame_skip={cfg.env.frame_skip}")
    print("Play in the Chrome window. Mouse steers, click/space boosts. Ctrl+C here to stop.\n")

    saved = 0
    episode = 0
    counts = {}

    def on_step(ep, info):
        counts[ep.actions[-1]] = counts.get(ep.actions[-1], 0) + 1
        if len(ep) % 50 == 0:
            print(f"  step {len(ep):4d} reward {ep.total_reward:8.2f} length {info.get('length', 0):4d}", end="\r", flush=True)

    def save(ep, cause, truncated):
        nonlocal saved
        if len(ep) < args.min_steps:
            print(f"episode {episode}: {len(ep)} steps < --min-steps {args.min_steps}, discarded")
            return
        path = ep.save(
            args.out, style=style_name, stage=stage, frame_skip=cfg.env.frame_skip,
            cause=str(cause) if cause is not None else None, truncated=truncated,
            nickname=args.nickname, url=args.url,
        )
        saved += 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
        mix = " ".join(f"{ACTION_NAMES[a]}:{n}" for a, n in top)
        print(f"episode {episode}: {len(ep)} steps, reward {ep.total_reward:.1f}, peak length {ep.peak_length}, "
              f"cause {cause} -> {os.path.basename(path)}  [{mix}]")

    try:
        while not args.episodes or saved < args.episodes:
            episode += 1
            counts.clear()
            ep, cause, truncated, interrupted = record_episode(env, max_steps, on_step)
            save(ep, cause, truncated)
            if interrupted:
                break
            time.sleep(0.5)
    finally:
        env.close()
    print(f"\nSaved {saved} episode(s) to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
