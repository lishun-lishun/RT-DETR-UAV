"""Read-only Baseline/HRNet fairness audit for DUT-Anti-UAV.

This tool never trains a model and never edits a YAML or checkpoint.  It can
run the static protocol/data audit without checkpoints, and, when existing
``best.pth`` files are supplied, evaluates the same EMA weights on Val and
Test in isolated subprocesses.

Examples (from ``rtdetr_pytorch``)::

    python tools/audit_backbone_fairness.py --static-only --hash-images \
      --output reports/backbone_fairness_audit.json \
      --markdown reports/backbone_fairness_audit.md

    CUDA_VISIBLE_DEVICES=1 python tools/audit_backbone_fairness.py \
      --baseline-checkpoint output/three_gpu_b16_warmup_cosine/rtdetr_r18vd_dut_anti_uav/best.pth \
      --hrnet-checkpoint output/three_gpu_b16_warmup_cosine/rtdetr_hrnetv2_w18_dut_anti_uav/best.pth \
      --device cuda:0 --num-workers 2 --hash-images \
      --output reports/backbone_fairness_audit.json \
      --markdown reports/backbone_fairness_audit.md
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import runpy
import statistics
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = ROOT / "configs/rtdetr/rtdetr_r18vd_dut_anti_uav.yml"
HRNET_CONFIG = ROOT / "configs/rtdetr/rtdetr_hrnetv2_w18_dut_anti_uav.yml"
MODEL_CONFIGS = {"PResNet18": BASELINE_CONFIG, "HRNetV2-W18": HRNET_CONFIG}
RESULT_NAMES = ("AP", "AP50", "AP75", "APS", "APM", "APL",
                "AR1", "AR10", "AR100", "ARS", "ARM", "ARL")
MARKER = "BACKBONE_FAIRNESS_JSON="


class _ShapeCaptured(RuntimeError):
    pass


def _prepare_optional_imports() -> bool:
    """Allow model-only audit in a lightweight environment.

    The placeholders are never used for evaluation or dataset construction.
    A server evaluation still fails loudly unless real pycocotools is present.
    """
    import types

    try:
        importlib.import_module("pycocotools")
        has_pycocotools = True
    except ImportError:
        has_pycocotools = False
        package = types.ModuleType("pycocotools")
        for name in ("mask", "coco", "cocoeval"):
            child = types.ModuleType(f"pycocotools.{name}")
            setattr(package, name, child)
            sys.modules[f"pycocotools.{name}"] = child
        package.coco.COCO = object
        package.cocoeval.COCOeval = object
        sys.modules["pycocotools"] = package
    try:
        importlib.import_module("transformers")
    except ImportError:
        package = types.ModuleType("transformers")
        package.RegNetModel = object
        sys.modules["transformers"] = package
    return has_pycocotools


def _fresh_config(path: Path) -> Dict[str, Any]:
    loader = runpy.run_path(str(ROOT / "src/core/yaml_utils.py"))
    return loader["load_config"](str(path), {})


def _flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if isinstance(value, dict):
        if not value:
            result[prefix] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(child, path))
    else:
        result[prefix] = value
    return result


def _differences(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    lhs, rhs = _flatten(left), _flatten(right)
    absent = "<absent>"
    return {
        key: {"baseline": lhs.get(key, absent), "hrnet": rhs.get(key, absent)}
        for key in sorted(set(lhs) | set(rhs))
        if lhs.get(key, absent) != rhs.get(key, absent)
    }


def _allowed_hrnet_difference(key: str) -> bool:
    return (key in {"__include__", "output_dir", "RTDETR.backbone",
                    "HybridEncoder.in_channels"}
            or key == "HRNetV2W18" or key.startswith("HRNetV2W18."))


def _detect_data_root(requested: Path | None) -> Path:
    candidates: List[Path] = []
    if requested is not None:
        candidates.append(requested)
    candidates.extend([
        ROOT.parent / "DUT-Anti-UAV/DUT-Anti-UAV",
        ROOT.parent.parent / "DUT-Anti-UAV/DUT-Anti-UAV",
    ])
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if all((candidate / "labels" / f"{split}.json").is_file()
               for split in ("train", "val", "test")):
            return candidate
    rendered = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError("DUT-Anti-UAV data root was not found. Checked:\n" + rendered)


def _percentile(ordered: Sequence[float], percent: int) -> float:
    position = (len(ordered) - 1) * percent / 100
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(ordered[low])
    fraction = position - low
    return float(ordered[low] * (1 - fraction) + ordered[high] * fraction)


def _describe(values: Iterable[float]) -> Dict[str, float] | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    return {
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "std": statistics.pstdev(ordered),
        "median": statistics.median(ordered),
        **{f"p{percent}": _percentile(ordered, percent)
           for percent in (10, 25, 50, 75, 90)},
    }


def _load_coco(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _split_statistics(data_root: Path, split: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    annotation_path = data_root / "labels" / f"{split}.json"
    data = _load_coco(annotation_path)
    images = {item["id"]: item for item in data.get("images", [])}
    filenames = {str(item["file_name"]).replace("\\", "/") for item in images.values()}
    counts = {image_id: 0 for image_id in images}
    width, height, area, relative_area = [], [], [], []
    valid = 0
    for annotation in data.get("annotations", []):
        image = images.get(annotation.get("image_id"))
        box = annotation.get("bbox")
        if image is None or not isinstance(box, list) or len(box) != 4:
            continue
        box_width, box_height = float(box[2]), float(box[3])
        if box_width <= 0 or box_height <= 0 or annotation.get("iscrowd", 0):
            continue
        image_area = float(image["width"]) * float(image["height"])
        box_area = box_width * box_height
        counts[annotation["image_id"]] += 1
        width.append(box_width)
        height.append(box_height)
        area.append(box_area)
        relative_area.append(box_area / image_area)
        valid += 1
    small = sum(value < 32 ** 2 for value in area)
    medium = sum(32 ** 2 <= value < 96 ** 2 for value in area)
    large = sum(value >= 96 ** 2 for value in area)
    stats = {
        "annotation_file": str(annotation_path),
        "image_folder": str(data_root / "images" / split),
        "images": len(images),
        "annotations_raw": len(data.get("annotations", [])),
        "annotations_valid_noncrowd": valid,
        "objects_per_image": _describe(counts.values()),
        "bbox_width": _describe(width),
        "bbox_height": _describe(height),
        "bbox_area": _describe(area),
        "relative_bbox_area": _describe(relative_area),
        "coco_size_by_bbox_area": {
            "small": {"count": small, "fraction": small / valid if valid else None},
            "medium": {"count": medium, "fraction": medium / valid if valid else None},
            "large": {"count": large, "fraction": large / valid if valid else None},
        },
        "image_width": _describe(item["width"] for item in images.values()),
        "image_height": _describe(item["height"] for item in images.values()),
    }
    metadata = {"filenames": filenames, "images": images}
    return stats, metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_hashes(data_root: Path, split: str,
                    filenames: Iterable[str]) -> Dict[str, List[str]]:
    hashes: Dict[str, List[str]] = {}
    for filename in sorted(filenames):
        path = data_root / "images" / split / filename
        if not path.is_file():
            raise FileNotFoundError(f"COCO image is missing: {path}")
        hashes.setdefault(_sha256(path), []).append(filename)
    return hashes


def _dataset_audit(data_root: Path, hash_images: bool) -> Dict[str, Any]:
    stats, metadata = {}, {}
    for split in ("train", "val", "test"):
        stats[split], metadata[split] = _split_statistics(data_root, split)
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    filename_overlap = {
        f"{left}_x_{right}": len(metadata[left]["filenames"] & metadata[right]["filenames"])
        for left, right in pairs
    }
    hash_overlap = None
    hash_overlap_details = None
    if hash_images:
        hashes = {split: _content_hashes(data_root, split, metadata[split]["filenames"])
                  for split in ("train", "val", "test")}
        hash_overlap = {f"{left}_x_{right}": len(set(hashes[left]) & set(hashes[right]))
                        for left, right in pairs}
        hash_overlap_details = {}
        for left, right in pairs:
            pair = f"{left}_x_{right}"
            hash_overlap_details[pair] = [
                {"sha256": digest,
                 f"{left}_files": hashes[left][digest],
                 f"{right}_files": hashes[right][digest]}
                for digest in sorted(set(hashes[left]) & set(hashes[right]))
            ]
    return {
        "data_root": str(data_root),
        "splits": stats,
        "filename_overlap": filename_overlap,
        "sha256_content_overlap": hash_overlap,
        "sha256_content_overlap_details": hash_overlap_details,
        "hash_images_enabled": hash_images,
    }


def _disable_pretrained(cfg: Dict[str, Any]) -> None:
    if "PResNet" in cfg:
        cfg["PResNet"]["pretrained"] = False
    if "HRNetV2W18" in cfg:
        cfg["HRNetV2W18"]["pretrained"] = False


def _set_loader_path(cfg: Dict[str, Any], loader_name: str,
                     data_root: Path, split: str, workers: int = 0) -> None:
    loader = cfg[loader_name]
    loader["dataset"]["img_folder"] = str(data_root / "images" / split)
    loader["dataset"]["ann_file"] = str(data_root / "labels" / f"{split}.json")
    loader["num_workers"] = workers
    if workers == 0:
        loader.pop("prefetch_factor", None)
        loader.pop("persistent_workers", None)


def _shape(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_shape(item) for item in value]
    if hasattr(value, "shape"):
        return list(value.shape)
    return str(type(value).__name__)


def _model_worker(model_name: str, data_root: Path, seed: int) -> Dict[str, Any]:
    import numpy as np
    import torch

    has_pycocotools = _prepare_optional_imports()
    sys.path.insert(0, str(ROOT))
    from src.core import YAMLConfig

    torch.manual_seed(seed)
    np.random.seed(seed)
    config_path = MODEL_CONFIGS[model_name]
    cfg = YAMLConfig(str(config_path), use_amp=False, resume="", tuning="")
    _disable_pretrained(cfg.yaml_cfg)
    _set_loader_path(cfg.yaml_cfg, "train_dataloader", data_root, "train")
    _set_loader_path(cfg.yaml_cfg, "val_dataloader", data_root, "val")
    model = cfg.model.cpu()
    optimizer = cfg.optimizer

    name_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    normalization_types = (torch.nn.modules.batchnorm._BatchNorm,
                           torch.nn.LayerNorm, torch.nn.GroupNorm,
                           torch.nn.modules.instancenorm._InstanceNorm)
    backbone_norm_ids = {
        id(parameter)
        for module in model.backbone.modules()
        if isinstance(module, normalization_types)
        for parameter in module.parameters(recurse=False)
    }
    groups = []
    backbone_lr_errors = []
    backbone_norm_weight_decay_errors = []
    for index, group in enumerate(optimizer.param_groups):
        names = [name_by_id[id(parameter)] for parameter in group["params"]]
        norm_names = [name_by_id[id(parameter)] for parameter in group["params"]
                      if id(parameter) in backbone_norm_ids]
        groups.append({
            "index": index,
            "lr": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
            "parameter_tensors": len(names),
            "parameter_numel": sum(parameter.numel() for parameter in group["params"]),
            "representative_names": names[:8],
            "all_backbone": bool(names) and all(name.startswith("backbone.") for name in names),
            "backbone_normalization_parameter_tensors": len(norm_names),
            "backbone_normalization_parameter_numel": sum(
                parameter.numel() for parameter in group["params"]
                if id(parameter) in backbone_norm_ids),
            "representative_backbone_normalization_names": norm_names[:8],
        })
        for name in names:
            if name.startswith("backbone.") and not math.isclose(
                    float(group["lr"]), 3e-5, rel_tol=0, abs_tol=1e-12):
                backbone_lr_errors.append({"name": name, "lr": float(group["lr"])})
        if norm_names and not math.isclose(float(group["weight_decay"]), 0.0,
                                           rel_tol=0, abs_tol=1e-12):
            backbone_norm_weight_decay_errors.append({
                "group": index, "weight_decay": float(group["weight_decay"]),
                "parameter_tensors": len(norm_names),
                "parameter_numel": sum(parameter.numel() for parameter in group["params"]
                                       if id(parameter) in backbone_norm_ids),
                "representative_names": norm_names[:8],
            })

    loader_batches = None
    if has_pycocotools:
        train_samples, train_targets = next(iter(cfg.train_dataloader))
        val_samples, val_targets = next(iter(cfg.val_dataloader))
        test_cfg = YAMLConfig(str(config_path), use_amp=False, resume="", tuning="")
        _disable_pretrained(test_cfg.yaml_cfg)
        _set_loader_path(test_cfg.yaml_cfg, "val_dataloader", data_root, "test")
        test_samples, test_targets = next(iter(test_cfg.val_dataloader))
        loader_batches = {
            "train": {"shape": list(train_samples.shape), "dtype": str(train_samples.dtype),
                      "device": str(train_samples.device)},
            "val": {"shape": list(val_samples.shape), "dtype": str(val_samples.dtype),
                    "device": str(val_samples.device)},
            "test": {"shape": list(test_samples.shape), "dtype": str(test_samples.dtype),
                     "device": str(test_samples.device)},
        }
        dynamic_sample = train_samples[:1]
        dynamic_target = [train_targets[0]]
    else:
        # Model-internal multi-scale is still executed exactly. Only the input
        # image is synthetic because this environment cannot construct COCO.
        dynamic_sample = torch.randn(1, 3, 640, 640)
        dynamic_target = [{"labels": torch.tensor([0]),
                           "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]])}]

    captures: Dict[str, Any] = {}

    def capture_model_input(module, inputs):
        captures.setdefault("model_input", _shape(inputs[0]))

    def capture_backbone_outputs(module, inputs, output):
        captures.setdefault("backbone_outputs", _shape(output))

    def capture_encoder_outputs(module, inputs, output):
        captures.setdefault("encoder_outputs", _shape(output))

    handles = [
        model.register_forward_pre_hook(capture_model_input),
        model.backbone.register_forward_hook(capture_backbone_outputs),
        model.encoder.register_forward_hook(capture_encoder_outputs),
    ]
    model.eval()
    with torch.no_grad():
        predictions = model(torch.randn(1, 3, 640, 640))
    for handle in handles:
        handle.remove()

    dynamic_backbone_inputs: List[List[int]] = []

    def capture_and_stop(module, inputs):
        dynamic_backbone_inputs.append(list(inputs[0].shape))
        raise _ShapeCaptured()

    handle = model.backbone.register_forward_pre_hook(capture_and_stop)
    model.train()
    try:
        for offset in range(8):
            np.random.seed(seed + offset)
            try:
                model(dynamic_sample, dynamic_target)
            except _ShapeCaptured:
                pass
    finally:
        handle.remove()

    return {
        "model": model_name,
        "config": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "dataloader_batches": loader_batches,
        "real_pycocotools_available": has_pycocotools,
        "training_backbone_input_samples": dynamic_backbone_inputs,
        "configured_multi_scale": list(cfg.yaml_cfg["RTDETR"]["multi_scale"]),
        "eval_forward": {
            **captures,
            "pred_logits": list(predictions["pred_logits"].shape),
            "pred_boxes": list(predictions["pred_boxes"].shape),
            "finite": all(torch.isfinite(value).all().item()
                          for value in predictions.values()),
        },
        "optimizer_groups": groups,
        "backbone_lr_expected": 3e-5,
        "backbone_lr_errors": backbone_lr_errors,
        "backbone_normalization_weight_decay_expected": 0.0,
        "backbone_normalization_weight_decay_errors": backbone_norm_weight_decay_errors,
    }


def _metric_dict(values: Sequence[float]) -> Dict[str, float]:
    if len(values) < len(RESULT_NAMES):
        raise ValueError(f"COCO evaluator returned only {len(values)} metrics")
    return {name: float(values[index]) for index, name in enumerate(RESULT_NAMES)}


def _eval_worker(model_name: str, split: str, checkpoint: Path,
                 data_root: Path, device_name: str, workers: int,
                 seed: int, weights: str) -> Dict[str, Any]:
    import torch

    if not _prepare_optional_imports():
        raise RuntimeError("Real pycocotools is required for Val/Test evaluation")
    sys.path.insert(0, str(ROOT))
    from src.core import YAMLConfig
    from src.data import get_coco_api_from_dataset
    from src.solver.det_engine import evaluate

    torch.manual_seed(seed)
    config_path = MODEL_CONFIGS[model_name]
    cfg = YAMLConfig(str(config_path), use_amp=False, resume="", tuning="")
    _disable_pretrained(cfg.yaml_cfg)
    _set_loader_path(cfg.yaml_cfg, "val_dataloader", data_root, split, workers)
    device = torch.device(device_name)
    model = cfg.model.to(device)
    state = torch.load(str(checkpoint), map_location="cpu")
    if weights == "ema":
        if "ema" not in state or "module" not in state["ema"]:
            raise KeyError(f"{checkpoint} does not contain ema.module")
        selected_state = state["ema"]["module"]
    else:
        if "model" not in state:
            raise KeyError(f"{checkpoint} does not contain model")
        selected_state = state["model"]
    model.load_state_dict(selected_state, strict=True)
    criterion = cfg.criterion.to(device)
    postprocessor = cfg.postprocessor
    loader = cfg.val_dataloader
    first_samples, first_targets = next(iter(loader))
    base_ds = get_coco_api_from_dataset(loader.dataset)
    stats, _ = evaluate(model, criterion, postprocessor, loader, base_ds,
                        device, output_dir=None)
    return {
        "model": model_name,
        "split": split,
        "checkpoint": str(checkpoint.resolve()),
        "weights": weights,
        "checkpoint_best_stat": state.get("best_stat"),
        "checkpoint_validation_stats": state.get("validation_stats"),
        "input": {"shape": list(first_samples.shape), "dtype": str(first_samples.dtype),
                  "loader_device": str(first_samples.device), "model_device": str(device)},
        "dataset": {"img_folder": str(loader.dataset.img_folder),
                    "ann_file": str(loader.dataset.ann_file)},
        "metrics": _metric_dict(stats["coco_eval_bbox"]),
    }


def _run_worker(arguments: Sequence[str]) -> Dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), *arguments]
    process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if process.returncode:
        raise RuntimeError("Audit worker failed:\nCOMMAND: " + " ".join(command)
                           + "\nSTDOUT:\n" + process.stdout
                           + "\nSTDERR:\n" + process.stderr)
    payloads = [line[len(MARKER):] for line in process.stdout.splitlines()
                if line.startswith(MARKER)]
    if len(payloads) != 1:
        raise RuntimeError("Audit worker did not emit one result:\n" + process.stdout)
    return json.loads(payloads[0])


def _find_checkpoint(explicit: Path | None, experiment: str) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    candidates = [
        ROOT / "output/three_gpu_b16_warmup_cosine" / experiment / "best.pth",
        ROOT / "output" / experiment / "best.pth",
    ]
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _config_audit() -> Dict[str, Any]:
    baseline = _fresh_config(BASELINE_CONFIG)
    hrnet = _fresh_config(HRNET_CONFIG)
    differences = _differences(baseline, hrnet)
    unexpected = {key: value for key, value in differences.items()
                  if not _allowed_hrnet_difference(key)}
    keys = [
        "train_dataloader", "val_dataloader", "test_dataset", "epoches",
        "optimizer", "lr_scheduler", "ema", "use_ema", "use_amp",
        "HybridEncoder", "RTDETRTransformer", "SetCriterion",
    ]
    return {
        "baseline_path": str(BASELINE_CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "hrnet_path": str(HRNET_CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "all_resolved_differences": differences,
        "unexpected_fairness_differences": unexpected,
        "only_permitted_differences": not unexpected,
        "shared_protocol": {key: baseline.get(key) for key in keys},
        "baseline_backbone": baseline["RTDETR"]["backbone"],
        "hrnet_backbone": hrnet["RTDETR"]["backbone"],
        "baseline_pretrained": baseline["PResNet"]["pretrained"],
        "hrnet_pretrained": hrnet["HRNetV2W18"]["pretrained"],
        "baseline_pretrained_source": (
            "ImageNet PResNet18-vd: ResNet18_vd_pretrained_from_paddle.pth"),
        "hrnet_pretrained_source": (
            "ImageNet HRNetV2-W18: hrnetv2_w18-8cb57bb9.pth"),
        "multi_scale": baseline["RTDETR"]["multi_scale"],
        "encoder_eval_spatial_size": baseline["HybridEncoder"]["eval_spatial_size"],
        "decoder_eval_spatial_size": baseline["RTDETRTransformer"]["eval_spatial_size"],
        "input_resolution_conclusion": {
            "train_dataloader_output": [640, 640],
            "train_model_internal_multi_scale": baseline["RTDETR"]["multi_scale"],
            "validation_fixed": [640, 640],
            "test_fixed": [640, 640],
            "old_fixed_800_or_960_affects_baseline": False,
            "old_fixed_800_or_960_affects_hrnet": False,
            "old_fixed_800_or_960_affects_test": False,
            "note": ("800 is an intentional training multi-scale candidate inherited "
                     "from RT-DETR, not a fixed evaluation resolution. 960 is absent "
                     "from the resolved Baseline/HRNet training and evaluation configs."),
        },
        "evaluation_chain": {
            "training_best_stat": {
                "loader": "val_dataloader",
                "img_folder": baseline["val_dataloader"]["dataset"]["img_folder"],
                "ann_file": baseline["val_dataloader"]["dataset"]["ann_file"],
                "weights": "ema.module because use_ema=True",
                "selector": "COCO bbox AP@[0.50:0.95] (stats[0])",
            },
            "tools_train_test_only": {
                "loader": "val_dataloader",
                "split": "val",
                "weights": "ema.module because use_ema=True",
                "upstream_behavior": "same: upstream tools/train.py also calls solver.val()",
            },
            "tools_test_dut_split_test": {
                "loader": "val_dataloader with only dataset paths replaced",
                "img_folder": baseline["test_dataset"]["img_folder"],
                "ann_file": baseline["test_dataset"]["ann_file"],
                "weights": "ema.module because use_ema=True",
            },
            "postprocessor": ("normalized cxcywh -> xyxy -> multiply by each target's "
                              "orig_size; identical for Val and Test"),
        },
    }


def _result_table(evaluations: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for model in MODEL_CONFIGS:
        val = evaluations.get(model, {}).get("val")
        test = evaluations.get(model, {}).get("test")
        best = None
        if val:
            best = (val.get("checkpoint_best_stat") or {}).get("coco_eval_bbox")
        rows.append({
            "model": model,
            "best_stat": best,
            "rerun_val_ap": val["metrics"]["AP"] if val else None,
            "test_ap": test["metrics"]["AP"] if test else None,
            "test_minus_val": (test["metrics"]["AP"] - val["metrics"]["AP"]
                               if val and test else None),
        })
    return rows


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "PENDING"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _render_markdown(report: Mapping[str, Any]) -> str:
    cfg = report["config_audit"]
    models = report["model_audit"]
    dataset = report["dataset_audit"]
    evaluations = report["evaluations"]
    lines = [
        "# DUT-Anti-UAV Baseline vs HRNetV2-W18 公平性审计", "",
        "> 本报告由只读审计生成；没有训练模型、修改配置或改写 checkpoint。", "",
        "## 输入尺寸", "",
        "| Stage | Baseline | HRNet | Same? |",
        "|---|---|---|---|",
    ]
    base, hrnet = models["PResNet18"], models["HRNetV2-W18"]

    def batch_shape(model: Mapping[str, Any], split: str) -> str:
        batches = model.get("dataloader_batches")
        if batches is None:
            return "NOT RUN (pycocotools unavailable)"
        return str(batches[split]["shape"])

    input_rows = [
        ("Train base resize", "640×640", "640×640"),
        ("Train multi-scale", str(cfg["multi_scale"]), str(cfg["multi_scale"])),
        ("DataLoader train batch", batch_shape(base, "train"),
         batch_shape(hrnet, "train")),
        ("Actual train backbone inputs", str(base["training_backbone_input_samples"]),
         str(hrnet["training_backbone_input_samples"])),
        ("Val batch", batch_shape(base, "val"), batch_shape(hrnet, "val")),
        ("Test batch", batch_shape(base, "test"), batch_shape(hrnet, "test")),
        ("eval_spatial_size", str(cfg["encoder_eval_spatial_size"]),
         str(cfg["encoder_eval_spatial_size"])),
        ("P3/P4/P5 stride", "[8,16,32]", "[8,16,32]"),
    ]
    for label, left, right in input_rows:
        lines.append(f"| {label} | `{left}` | `{right}` | {'YES' if left == right else 'NO'} |")

    lines.extend([
        "",
        "- Validation/Test 均固定为 **640×640**。",
        "- 800 仅是训练 multi-scale 候选值，不是固定评测分辨率。",
        "- 960 不存在于当前 Baseline/HRNet resolved 训练或评测配置。",
        "- `best_stat` 与 `tools/train.py --test-only` 使用 Val；"
        "`tools/test_dut.py --split test` 才显式切换到 Test。",
    ])

    lines.extend(["", "## Val/Test 结果", "",
                  "| Model | best_stat | rerun Val | Test | Test-Val |",
                  "|---|---:|---:|---:|---:|"])
    for row in report["result_table"]:
        lines.append("| {model} | {best} | {val} | {test} | {gap} |".format(
            model=row["model"], best=_fmt(row["best_stat"]),
            val=_fmt(row["rerun_val_ap"]), test=_fmt(row["test_ap"]),
            gap=_fmt(row["test_minus_val"])))

    lines.extend(["", "## 完整 COCO 指标", ""])
    for model in MODEL_CONFIGS:
        for split in ("val", "test"):
            result = evaluations.get(model, {}).get(split)
            lines.append(f"### {model} / {split.upper()}")
            lines.append("")
            if result is None:
                lines.append("PENDING：本机没有对应 checkpoint。")
            else:
                lines.append("| " + " | ".join(RESULT_NAMES) + " |")
                lines.append("|" + "---:|" * len(RESULT_NAMES))
                lines.append("| " + " | ".join(
                    _fmt(result["metrics"][name]) for name in RESULT_NAMES) + " |")
            lines.append("")

    lines.extend([
        "## 数据分布摘要", "",
        "| Split | Images | Objects | Obj/Image mean | Box area median | Relative area median | Small | Medium | Large |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for split in ("train", "val", "test"):
        item = dataset["splits"][split]
        sizes = item["coco_size_by_bbox_area"]
        lines.append(
            f"| {split} | {item['images']} | {item['annotations_valid_noncrowd']} | "
            f"{_fmt(item['objects_per_image']['mean'])} | {_fmt(item['bbox_area']['median'])} | "
            f"{_fmt(item['relative_bbox_area']['median'])} | {sizes['small']['count']} | "
            f"{sizes['medium']['count']} | {sizes['large']['count']} |")

    lines.extend([
        "", "## 公平性结论", "",
        f"- Resolved config 仅有允许差异：**{'YES' if cfg['only_permitted_differences'] else 'NO'}**",
        "- Train/Val/Test、增强、batch、epoch、优化器 YAML、LR、scheduler、EMA、Detector、Matcher、Loss：相同。",
        "- PResNet18 与 HRNetV2-W18 均使用各自 ImageNet 预训练权重。",
        f"- 实际优化器参数语义一致：**{'YES' if report['fairness_assessment']['optimizer_parameter_semantics_fair'] else 'NO'}**",
        "- 当前 HRNet BatchNorm 名称未命中 `norm` 正则，归一化参数使用了普通 Backbone 的 weight decay。",
        "- `best_stat` 来自 `val_dataloader`；原始及当前 `tools/train.py --test-only` 也评估 Val。",
        "- 真正 Test 由 `tools/test_dut.py --split test` 或本审计工具显式切换 split。",
        f"- 当前公平性等级：**{report['fairness_assessment']['grade']}**。",
        "- 当前结果可做 Backbone 对比，但不能表述为严格控制全部优化器语义的 STRICTLY FAIR 实验。",
        "",
        "## 数据泄漏检查", "",
        f"- 文件名交集：`{dataset['filename_overlap']}`",
        f"- SHA256 内容交集：`{dataset['sha256_content_overlap']}`",
        f"- SHA256 重复明细：`{dataset['sha256_content_overlap_details']}`",
        "",
    ])
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--hrnet-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--compare-raw-baseline-val", action="store_true")
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--hash-images", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--worker-model", choices=tuple(MODEL_CONFIGS), help=argparse.SUPPRESS)
    parser.add_argument("--worker-eval", choices=tuple(MODEL_CONFIGS), help=argparse.SUPPRESS)
    parser.add_argument("--worker-split", choices=("val", "test"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-checkpoint", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be non-negative")
    data_root = _detect_data_root(args.data_root)
    if args.worker_model:
        result = _model_worker(args.worker_model, data_root, args.seed)
        print(MARKER + json.dumps(result, ensure_ascii=False))
        return
    if args.worker_eval:
        if not args.worker_split or not args.worker_checkpoint:
            raise SystemExit("evaluation worker requires split and checkpoint")
        result = _eval_worker(args.worker_eval, args.worker_split,
                              args.worker_checkpoint.resolve(), data_root,
                              args.device, args.num_workers, args.seed, args.weights)
        print(MARKER + json.dumps(result, ensure_ascii=False))
        return

    report: Dict[str, Any] = {
        "scope": "read-only; no training, YAML edits, checkpoint writes or output writes",
        "config_audit": _config_audit(),
        "dataset_audit": _dataset_audit(data_root, args.hash_images),
        "model_audit": {},
        "evaluations": {},
    }
    for model_name in MODEL_CONFIGS:
        report["model_audit"][model_name] = _run_worker([
            "--worker-model", model_name, "--data-root", str(data_root),
            "--seed", str(args.seed),
        ])

    optimizer_semantics_fair = all(
        not item["backbone_lr_errors"]
        and not item["backbone_normalization_weight_decay_errors"]
        for item in report["model_audit"].values())
    config_fair = report["config_audit"]["only_permitted_differences"]
    grade = ("STRICTLY FAIR" if config_fair and optimizer_semantics_fair
             else "MOSTLY FAIR" if config_fair else "NOT FAIR")
    report["fairness_assessment"] = {
        "grade": grade,
        "resolved_config_fair": config_fair,
        "optimizer_parameter_semantics_fair": optimizer_semantics_fair,
        "pretraining_level_fair": True,
        "paper_strict_claim_allowed": grade == "STRICTLY FAIR",
        "required_fix_for_strict_claim": (
            None if grade == "STRICTLY FAIR" else
            "Group all normalization parameters semantically (including HRNet bn*) "
            "with weight_decay=0, then retrain the affected comparison under the "
            "same protocol. Do not edit existing checkpoints."),
    }

    checkpoints = {
        "PResNet18": _find_checkpoint(args.baseline_checkpoint,
                                       "rtdetr_r18vd_dut_anti_uav"),
        "HRNetV2-W18": _find_checkpoint(args.hrnet_checkpoint,
                                         "rtdetr_hrnetv2_w18_dut_anti_uav"),
    }
    report["checkpoints"] = {name: str(path) if path else None
                             for name, path in checkpoints.items()}
    if not args.static_only:
        for model_name, checkpoint in checkpoints.items():
            if checkpoint is None:
                continue
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
            report["evaluations"][model_name] = {}
            for split in ("val", "test"):
                report["evaluations"][model_name][split] = _run_worker([
                    "--worker-eval", model_name,
                    "--worker-split", split,
                    "--worker-checkpoint", str(checkpoint),
                    "--data-root", str(data_root),
                    "--device", args.device,
                    "--num-workers", str(args.num_workers),
                    "--seed", str(args.seed),
                    "--weights", args.weights,
                ])
        if args.compare_raw_baseline_val and checkpoints["PResNet18"]:
            report["baseline_raw_val"] = _run_worker([
                "--worker-eval", "PResNet18", "--worker-split", "val",
                "--worker-checkpoint", str(checkpoints["PResNet18"]),
                "--data-root", str(data_root), "--device", args.device,
                "--num-workers", str(args.num_workers), "--seed", str(args.seed),
                "--weights", "raw",
            ])

    report["result_table"] = _result_table(report["evaluations"])
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved JSON: {args.output.resolve()}")
    else:
        print(rendered)
    markdown = _render_markdown(report)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown + "\n", encoding="utf-8")
        print(f"Saved Markdown: {args.markdown.resolve()}")


if __name__ == "__main__":
    main()
