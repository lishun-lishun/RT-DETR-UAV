#!/usr/bin/env python3
"""Find and validate a *full* RT-DETR training checkpoint for strict resume.

Selection policy (in priority order):

1. ``last.pth`` when it is a complete training checkpoint;
2. the complete ``checkpoint*.pth`` with the greatest saved epoch;
3. ``best.pth`` when it is itself a complete training checkpoint.

The current solver writes ``last_epoch`` (rather than ``epoch``) and normally
writes its latest state to ``checkpoint.pth``.  Historical aliases are accepted
so older runs can be audited, but every one of model/optimizer/scheduler/epoch/
EMA/AMP-scaler must be present and structurally usable.  Model-only weights are
never silently accepted as a resume checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


STATE_ALIASES = {
    "model": ("model", "state_dict", "model_state_dict"),
    "optimizer": ("optimizer", "optimizer_state_dict"),
    "lr_scheduler": ("lr_scheduler", "scheduler", "scheduler_state_dict"),
    # The repository's BaseSolver.state_dict() uses last_epoch.
    "epoch": ("last_epoch", "epoch"),
    "ema": ("ema", "model_ema", "ema_state_dict"),
    "scaler": ("scaler", "amp_scaler", "grad_scaler", "scaler_state_dict"),
}

REQUIRED_STATES = tuple(STATE_ALIASES)


def _json_safe(value: Any) -> Any:
    """Return a compact JSON-safe representation of checkpoint metadata."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if torch.is_tensor(value) and value.numel() == 1:
        return value.detach().cpu().item()
    return str(value)


def _find_key(state: Mapping[str, Any], aliases: tuple[str, ...]) -> str | None:
    for key in aliases:
        if key in state:
            return key
    return None


def _saved_epoch(value: Any) -> int | None:
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _nonempty_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value)


def inspect_checkpoint(path: Path, kind: str) -> dict[str, Any]:
    """Load one candidate on CPU and describe strict-resume completeness."""
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "name": path.name,
        "kind": kind,
        "exists": path.is_file(),
        "load_ok": False,
        "complete": False,
        "saved_epoch": None,
        "next_epoch": None,
        "states": {name: False for name in REQUIRED_STATES},
        "keys": {name: None for name in REQUIRED_STATES},
        "best_stat": None,
        "error": None,
    }
    if not path.is_file():
        record["error"] = "file not found"
        return record

    try:
        state = torch.load(str(path), map_location="cpu")
    except Exception as exc:  # corrupt/truncated/incompatible checkpoints are candidates, not crashes
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    if not isinstance(state, Mapping):
        record["error"] = f"checkpoint root is {type(state).__name__}, expected mapping"
        return record

    record["load_ok"] = True
    for canonical, aliases in STATE_ALIASES.items():
        key = _find_key(state, aliases)
        record["keys"][canonical] = key
        if key is None:
            continue
        if canonical == "epoch":
            epoch = _saved_epoch(state[key])
            record["saved_epoch"] = epoch
            record["next_epoch"] = None if epoch is None else epoch + 1
            record["states"][canonical] = epoch is not None and epoch >= -1
        else:
            record["states"][canonical] = _nonempty_mapping(state[key])

    record["best_stat"] = _json_safe(state.get("best_stat"))
    record["complete"] = all(record["states"].values())
    if not record["complete"]:
        missing = [name for name, ok in record["states"].items() if not ok]
        record["error"] = "missing or unusable state: " + ", ".join(missing)

    # Release potentially large tensor dictionaries before inspecting another file.
    del state
    return record


def _checkpoint_candidates(output_dir: Path) -> list[Path]:
    """Return checkpoint*.pth candidates, including current checkpoint.pth."""
    return sorted(
        (p for p in output_dir.glob("checkpoint*.pth") if p.is_file()),
        key=lambda p: p.name,
    )


def inspect_output_dir(output_dir: Path, target_epochs: int = 200) -> dict[str, Any]:
    """Apply the documented priority policy to an experiment output directory."""
    output_dir = output_dir.expanduser().resolve()
    target_last_epoch = target_epochs - 1
    inspected: list[dict[str, Any]] = []
    chosen: dict[str, Any] | None = None

    last_path = output_dir / "last.pth"
    if last_path.is_file():
        last_record = inspect_checkpoint(last_path, "last")
        inspected.append(last_record)
        if last_record["complete"]:
            chosen = last_record

    if chosen is None:
        checkpoint_records = [
            inspect_checkpoint(path, "checkpoint")
            for path in _checkpoint_candidates(output_dir)
        ]
        inspected.extend(checkpoint_records)
        complete = [r for r in checkpoint_records if r["complete"]]
        if complete:
            # Use the epoch stored inside the file, never the filename alone.
            # checkpoint.pth wins an exact epoch tie because it is the solver's
            # rolling latest-state filename; mtime is only a final tie-breaker.
            def rank(record: dict[str, Any]) -> tuple[int, int, int, str]:
                p = Path(record["path"])
                try:
                    mtime = p.stat().st_mtime_ns
                except OSError:
                    mtime = -1
                return (
                    int(record["saved_epoch"]),
                    int(record["name"] == "checkpoint.pth"),
                    mtime,
                    record["name"],
                )

            chosen = max(complete, key=rank)

    if chosen is None:
        best_path = output_dir / "best.pth"
        if best_path.is_file():
            best_record = inspect_checkpoint(best_path, "best")
            inspected.append(best_record)
            if best_record["complete"]:
                chosen = best_record

    possible = chosen is not None
    saved_epoch = None if chosen is None else chosen["saved_epoch"]
    result: dict[str, Any] = {
        "schema_version": 1,
        "output_dir": str(output_dir),
        "target_epochs": target_epochs,
        "target_last_epoch": target_last_epoch,
        "priority_policy": [
            "complete last.pth",
            "complete checkpoint*.pth with greatest saved epoch",
            "complete best.pth",
        ],
        "status": "READY" if possible else "STRICT RESUME NOT POSSIBLE",
        "strict_resume_possible": possible,
        "completed": bool(possible and saved_epoch >= target_last_epoch),
        "checkpoint": None if chosen is None else chosen["path"],
        "checkpoint_kind": None if chosen is None else chosen["kind"],
        "saved_epoch": saved_epoch,
        "next_epoch": None if chosen is None else chosen["next_epoch"],
        "states": (
            {name: False for name in REQUIRED_STATES}
            if chosen is None
            else chosen["states"]
        ),
        "keys": (
            {name: None for name in REQUIRED_STATES}
            if chosen is None
            else chosen["keys"]
        ),
        "best_stat": None if chosen is None else chosen["best_stat"],
        "candidates": inspected,
    }
    return result


def print_human(result: Mapping[str, Any]) -> None:
    yes_no = lambda value: "YES" if value else "NO"
    print(f"STATUS={result['status']}")
    print(f"STRICT_RESUME_POSSIBLE={yes_no(result['strict_resume_possible'])}")
    print(f"RESUME_CHECKPOINT={result['checkpoint'] or 'NONE'}")
    print(f"CHECKPOINT_KIND={result['checkpoint_kind'] or 'NONE'}")
    print(f"SAVED_EPOCH={result['saved_epoch'] if result['saved_epoch'] is not None else 'NONE'}")
    print(f"NEXT_EPOCH={result['next_epoch'] if result['next_epoch'] is not None else 'NONE'}")
    print(f"TARGET_LAST_EPOCH={result['target_last_epoch']}")
    print(f"TRAINING_COMPLETED={yes_no(result['completed'])}")
    for name in REQUIRED_STATES:
        label = {
            "model": "MODEL_STATE_LOADED",
            "optimizer": "OPTIMIZER_STATE_LOADED",
            "lr_scheduler": "SCHEDULER_STATE_LOADED",
            "epoch": "EPOCH_STATE_LOADED",
            "ema": "EMA_STATE_LOADED",
            "scaler": "AMP_SCALER_STATE_LOADED",
        }[name]
        print(f"{label}={yes_no(result['states'][name])}")
    print("BEST_STAT=" + json.dumps(result["best_stat"], ensure_ascii=False, sort_keys=True))
    print(f"CANDIDATES_INSPECTED={len(result['candidates'])}")
    if not result["strict_resume_possible"]:
        print("STRICT RESUME NOT POSSIBLE")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and validate the latest complete RT-DETR resume checkpoint."
    )
    parser.add_argument("output_dir", type=Path, help="experiment output directory")
    parser.add_argument("--target-epochs", type=int, default=200)
    parser.add_argument("--json-out", type=Path, help="also write the complete audit as JSON")
    parser.add_argument(
        "--json-only", action="store_true", help="print JSON only (still uses exit code 2 when blocked)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.target_epochs < 1:
        raise SystemExit("--target-epochs must be positive")
    result = inspect_output_dir(args.output_dir, args.target_epochs)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + os.linesep, encoding="utf-8")
    if args.json_only:
        print(encoded)
    else:
        print_human(result)
    return 0 if result["strict_resume_possible"] else 2


if __name__ == "__main__":
    sys.exit(main())
