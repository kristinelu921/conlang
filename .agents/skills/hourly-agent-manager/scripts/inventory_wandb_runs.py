#!/usr/bin/env python3
"""Export authenticated W&B run metadata for experiment-log reconciliation."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import wandb


DEFAULT_ENTITY = "kaynelu921-massachusetts-institute-of-technology"
SENSITIVE_KEY_TERMS = ("api_key", "credential", "password", "secret")


def redact_sensitive_values(value: Any, key: str = "") -> Any:
    """Redact secret-like fields recursively while preserving public evidence."""
    normalized_key = key.lower()
    is_sensitive = (
        any(term in normalized_key for term in SENSITIVE_KEY_TERMS)
        or normalized_key == "token"
        or normalized_key.endswith("_token")
    )
    if is_sensitive:
        return "<redacted>"
    if isinstance(value, dict):
        return {
            str(nested_key): redact_sensitive_values(nested_value, str(nested_key))
            for nested_key, nested_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_sensitive_values(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def serialize_run(run: Any) -> Dict[str, Any]:
    """Convert one public W&B run into redacted reconciliation evidence."""
    metadata = run.metadata or {}
    return {
        "entity": run.entity,
        "project": run.project,
        "run_id": run.id,
        "name": run.name,
        "state": run.state,
        "url": run.url,
        "user": str(run.user),
        "created_at": run.created_at,
        "last_history_step": run.lastHistoryStep,
        "tags": list(run.tags or []),
        "notes": run.notes,
        "config": redact_sensitive_values(dict(run.config or {})),
        "summary_metrics": redact_sensitive_values(dict(run.summary_metrics or {})),
        "metadata": redact_sensitive_values(dict(metadata)),
    }


def collect_run_inventory(
    api: Any,
    entity: str,
    project_names: Optional[Sequence[str]],
    username: Optional[str],
    states: Sequence[str],
) -> Dict[str, Any]:
    """Collect accessible W&B runs across the requested entity projects."""
    selected_projects = list(project_names or [project.name for project in api.projects(entity=entity)])
    filters: Dict[str, Any] = {}
    if username:
        filters["username"] = username
    if states:
        filters["state"] = {"$in": list(states)}

    records: List[Dict[str, Any]] = []
    inaccessible_projects: List[Dict[str, str]] = []
    for project_name in selected_projects:
        try:
            runs: Iterable[Any] = api.runs(
                path=f"{entity}/{project_name}",
                filters=filters or None,
                order="-created_at",
                per_page=100,
                lazy=False,
            )
            records.extend(serialize_run(run) for run in runs)
        except Exception as error:  # W&B raises transport- and permission-specific exceptions.
            inaccessible_projects.append({"project": project_name, "error": str(error)})

    return {
        "queried_at": datetime.now(timezone.utc).isoformat(),
        "entity": entity,
        "username": username,
        "projects": selected_projects,
        "inaccessible_projects": inaccessible_projects,
        "runs": records,
    }


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the W&B inventory export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entity",
        default=os.getenv("WANDB_ENTITY") or DEFAULT_ENTITY,
        help="W&B entity whose projects should be inspected.",
    )
    parser.add_argument(
        "--project",
        action="append",
        dest="projects",
        help="Restrict inventory to this project; repeat for multiple projects.",
    )
    parser.add_argument(
        "--username",
        help="Restrict runs to this username; defaults to the authenticated viewer.",
    )
    parser.add_argument(
        "--all-users",
        action="store_true",
        help="Include runs created by every user in the entity.",
    )
    parser.add_argument(
        "--state",
        action="append",
        default=[],
        choices=("running", "finished", "crashed", "killed", "preempting", "preempted", "pending"),
        help="Restrict runs to this W&B state; repeat for multiple states.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON to this path instead of standard output.",
    )
    return parser.parse_args()


def main() -> int:
    """Query W&B and emit a redacted run inventory as JSON."""
    arguments = parse_arguments()
    api = wandb.Api()
    username = None if arguments.all_users else arguments.username or api.viewer.username
    inventory = collect_run_inventory(
        api=api,
        entity=arguments.entity,
        project_names=arguments.projects,
        username=username,
        states=arguments.state,
    )
    serialized_inventory = json.dumps(inventory, indent=2, sort_keys=True)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(f"{serialized_inventory}\n", encoding="utf-8")
    else:
        print(serialized_inventory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
