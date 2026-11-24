"""Command-line entry points for validation, remote execution, and local inspection."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from uuid import uuid4

from island_community.config import load_config


def main(arguments: list[str] | None = None) -> int:
    """Dispatch commands without importing neural libraries for validation or viewing."""
    parser = argparse.ArgumentParser(description="Island Community: independent learners in a resource economy")
    commands = parser.add_subparsers(dest="command", required=True)
    validate_parser = commands.add_parser("validate", help="Validate rules without running the world")
    validate_parser.add_argument("--config", type=Path)
    run_parser = commands.add_parser("run", help="Run simulation and learning on authorized compute")
    run_parser.add_argument("--config", type=Path)
    run_parser.add_argument("--output", type=Path)
    run_parser.add_argument("--resume", type=Path, help="Resume checkpoint into a new output directory")
    run_parser.add_argument("--turns", type=int, default=100, help="Additional complete turns to execute")
    run_parser.add_argument("--device", default="cpu", help="PyTorch device: cpu, cuda:0, or tpu (Qwen only)")
    run_parser.add_argument("--no-vmap", action="store_true", help="Use equivalent per-island forward loops")
    run_parser.add_argument("--passive", action="store_true", help="Use wait actions for a no-learning control")
    run_parser.add_argument("--checkpoint-every", type=int, default=10)
    view_parser = commands.add_parser("view", help="Inspect recorded outputs without running models")
    view_parser.add_argument("run_directory", type=Path)
    view_parser.add_argument("--host", default="127.0.0.1")
    view_parser.add_argument("--port", type=int, default=8765)
    replay_parser = commands.add_parser("rebuild-replay", help="Reconstruct an older recording on authorized compute without neural inference")
    replay_parser.add_argument("run_directory", type=Path)
    replay_parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(arguments)
    try:
        if options.command == "validate":
            configuration = load_config(options.config)
            print(f"Valid rules: {configuration['initialization']['population']} founders, "
                  f"{len(configuration['items'])} items, {len(configuration['recipes'])} recipes; death replay disabled.")
        elif options.command == "view":
            from island_community.viewer import serve

            serve(options.run_directory, options.host, options.port)
        elif options.command == "rebuild-replay":
            from island_community.replay import rebuild_replay

            output = rebuild_replay(options.run_directory, options.output)
            print(f"Verified replay saved at {output}")
        else:
            from island_community.runner import run

            timestamp = datetime.now(timezone.utc)
            output = options.output or (
                Path.cwd() / "experiment_log" / timestamp.strftime("%Y-%m-%d")
                / f"island-community-{timestamp.strftime('%H%M%S')}-{uuid4().hex[:6]}"
            )
            run(
                output, options.turns, options.config, options.resume, options.device,
                False if options.no_vmap else None, options.passive, options.checkpoint_every,
                [sys.executable, "-m", "island_community", *(arguments if arguments is not None else sys.argv[1:])],
            )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
