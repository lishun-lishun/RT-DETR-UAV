#!/usr/bin/env python3
"""Preview and remove obsolete DUT experiment directories safely.

The script keeps the eight current formal/reference experiments and considers
every other *direct child directory* of the selected output root obsolete. It never
recurses outside that root, never deletes ordinary files, and defaults to a
read-only preview.  Use ``--apply`` only after reviewing the printed manifest.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_OUTPUT = PROJECT_ROOT / "output"
DEFAULT_OUTPUT_ROOT = PROJECT_OUTPUT / "three_gpu_b16_warmup_cosine"

ACTIVE_EXPERIMENTS = (
    "rtdetr_r18vd_dut_anti_uav",
    "rtdetr_hrnetv2_w18_dut_anti_uav",
    "rtdetr_r18vd_dut_anti_uav_spdr",
    "rtdetr_hrnetv2_w18_dut_anti_uav_spdr",
    "rtdetr_r18vd_dut_anti_uav_fdcr",
    "rtdetr_hrnetv2_w18_dut_anti_uav_fdcr",
    "rtdetr_r18vd_dut_anti_uav_rdcf",
    "rtdetr_hrnetv2_w18_dut_anti_uav_rdcf",
)
REQUIRED_BASELINES = (
    "rtdetr_r18vd_dut_anti_uav",
    "rtdetr_hrnetv2_w18_dut_anti_uav",
)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_output_root(value: Path) -> Path:
    path = value.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve(strict=True)
    project_output = PROJECT_OUTPUT.resolve(strict=False)
    if not _is_relative_to(resolved, project_output) or resolved == project_output:
        raise RuntimeError(
            "Refusing cleanup outside a specific subdirectory of the project "
            f"output root: {resolved}"
        )
    if not resolved.is_dir():
        raise RuntimeError(f"Output root is not a directory: {resolved}")
    return resolved


def build_plan(output_root: Path, keep_names: set[str]) -> dict[str, list[Path]]:
    children = sorted(output_root.iterdir(), key=lambda path: path.name)
    directories = [path for path in children if path.is_dir() or path.is_symlink()]
    files = [path for path in children if path not in directories]
    return {
        "keep_existing": [path for path in directories if path.name in keep_names],
        "keep_missing": [output_root / name for name in sorted(keep_names)
                         if not (output_root / name).exists()
                         and not (output_root / name).is_symlink()],
        "delete": [path for path in directories if path.name not in keep_names],
        "ignored_files": files,
    }


def format_manifest(output_root: Path, keep_names: set[str], plan: dict[str, list[Path]]) -> str:
    lines = [
        "DUT OUTPUT CLEANUP PLAN",
        f"output_root: {output_root}",
        "",
        "[PROTECTED NAMES]",
        *(sorted(keep_names)),
        "",
        "[KEEP EXISTING DIRECTORIES]",
        *(str(path) for path in plan["keep_existing"]),
        "",
        "[PROTECTED BUT NOT PRESENT]",
        *(str(path) for path in plan["keep_missing"]),
        "",
        "[DELETE DIRECTORIES]",
        *(str(path) for path in plan["delete"]),
        "",
        "[IGNORED FILES - NEVER DELETED]",
        *(str(path) for path in plan["ignored_files"]),
        "",
        f"keep_count: {len(plan['keep_existing'])}",
        f"delete_count: {len(plan['delete'])}",
        f"ignored_file_count: {len(plan['ignored_files'])}",
        "",
    ]
    return "\n".join(lines)


def remove_direct_directory(path: Path, output_root: Path) -> None:
    # Compare the lexical parent, not the resolved link target. A direct
    # symlink is unlinked; its target is never traversed or removed.
    if path.parent.resolve(strict=True) != output_root.resolve(strict=True):
        raise RuntimeError(f"Refusing non-direct deletion target: {path}")
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    else:
        raise RuntimeError(f"Refusing non-directory deletion target: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="specific experiment-root directory inside this project's output/",
    )
    parser.add_argument(
        "--keep",
        action="append",
        default=[],
        metavar="NAME",
        help="protect one additional direct child directory (repeatable)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("output_cleanup_manifest.txt"),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="delete the listed obsolete directories; default is preview only",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    invalid = [name for name in args.keep
               if not name or Path(name).name != name or name in (".", "..")]
    if invalid:
        raise ValueError(f"--keep accepts plain directory names only: {invalid}")

    output_root = resolve_output_root(args.output_root)
    keep_names = set(ACTIVE_EXPERIMENTS).union(args.keep)
    plan = build_plan(output_root, keep_names)
    manifest = format_manifest(output_root, keep_names, plan)
    manifest_path = args.manifest.expanduser()
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    manifest_path.write_text(manifest, encoding="utf-8")
    print(manifest, end="")
    print(f"Manifest written: {manifest_path.resolve()}")

    if not args.apply:
        print("DRY RUN: nothing was deleted. Review the list, then add --apply.")
        return 0

    missing_baselines = [
        output_root / name / "best.pth" for name in REQUIRED_BASELINES
        if not (output_root / name / "best.pth").is_file()
    ]
    if missing_baselines:
        raise RuntimeError(
            "Refusing cleanup because protected baseline checkpoints are missing: "
            + ", ".join(str(path) for path in missing_baselines)
        )

    for path in plan["delete"]:
        remove_direct_directory(path, output_root)
        print(f"DELETED: {path}")

    remaining = {path.name for path in output_root.iterdir()
                 if path.is_dir() or path.is_symlink()}
    unexpected = remaining - keep_names
    if unexpected:
        raise RuntimeError(
            f"Cleanup verification failed; unexpected directories remain: {sorted(unexpected)}")
    print(
        f"Cleanup complete: deleted={len(plan['delete'])}, "
        f"protected_existing={len(remaining)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
