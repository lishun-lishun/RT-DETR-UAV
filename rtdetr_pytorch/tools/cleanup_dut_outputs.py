#!/usr/bin/env python3
"""Safely remove obsolete DUT experiment outputs after printing a manifest.

The default mode is read-only.  Deletion requires ``--apply`` and is refused
unless both protected baseline directories and their ``best.pth`` files are
present.  Only direct children of the two explicitly resolved result roots can
ever be removed.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROTECTED_EXPERIMENTS = (
    "rtdetr_r18vd_dut_anti_uav",
    "rtdetr_hrnetv2_w18_dut_anti_uav",
)


def _direct_children(root: Path) -> list[Path]:
    return sorted(root.iterdir(), key=lambda path: path.name) if root.is_dir() else []


def _assert_direct_child(path: Path, root: Path) -> None:
    resolved_root = root.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    if resolved_path.parent != resolved_root:
        raise RuntimeError(f"Refusing non-direct deletion target: {resolved_path}")


def build_manifest(output_root: Path, report_root: Path) -> tuple[str, list[Path], list[Path]]:
    output_root = output_root.resolve()
    report_root = report_root.resolve()
    output_children = _direct_children(output_root)
    report_children = _direct_children(report_root)
    protected = [output_root / name for name in PROTECTED_EXPERIMENTS]
    delete_outputs = [path for path in output_children if path not in protected]

    lines = [
        "[KEEP OUTPUT]",
        *(str(path) for path in protected),
        "",
        "[DELETE OUTPUT]",
        *(str(path) for path in delete_outputs),
        "",
        "[DELETE AGGREGATE TEST REPORT]",
        *(str(path) for path in report_children),
        "",
    ]
    return "\n".join(lines), delete_outputs, report_children


def _remove_direct_child(path: Path, root: Path) -> None:
    _assert_direct_child(path, root)
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/three_gpu_b16_warmup_cosine"),
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("output/three_gpu_b16_warmup_cosine_test_results"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("cleanup_manifest_before_new_necks_server.txt"),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the listed deletions; without this flag the tool is read-only",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    report_root = args.report_root.resolve()
    text, delete_outputs, delete_reports = build_manifest(output_root, report_root)
    args.manifest.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"Manifest written: {args.manifest.resolve()}")

    if not args.apply:
        print("DRY RUN: nothing was deleted. Re-run with --apply after reviewing the manifest.")
        return 0

    if not output_root.is_dir():
        raise RuntimeError(f"Output root does not exist: {output_root}")
    for name in PROTECTED_EXPERIMENTS:
        directory = output_root / name
        checkpoint = directory / "best.pth"
        if not directory.is_dir() or not checkpoint.is_file():
            raise RuntimeError(
                "Refusing cleanup because a protected baseline/best.pth is missing: "
                f"{checkpoint}"
            )

    for path in delete_outputs:
        _remove_direct_child(path, output_root)
        print(f"DELETED output: {path}")
    for path in delete_reports:
        _remove_direct_child(path, report_root)
        print(f"DELETED report: {path}")

    # Re-verify protection after cleanup.
    for name in PROTECTED_EXPERIMENTS:
        checkpoint = output_root / name / "best.pth"
        if not checkpoint.is_file():
            raise RuntimeError(f"Protected checkpoint disappeared: {checkpoint}")
    print("Cleanup complete; both protected baseline best.pth files are intact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
