"""Audit DUT YAML fairness and measure the real 640 FP32 evaluation forward.

This reports *lower bounds*, not total model FLOPs. PyTorch's profiler does not
assign FLOPs to every operation (notably grid sampling, normalization, softmax,
fused attention, and parts of SECD). Conv/Linear hooks also omit functional
linear operations such as MultiheadAttention projections.

Examples:
    python tools/analyze_dut_models.py --resolved-only
    python tools/analyze_dut_models.py --model-only-import --output reports/dut_metrics.json

The explicit model-only import mode loads the real R18 model source without
importing unrelated dataset/RegNet dependencies. It never supplies dependency
stubs and is not a check of the complete training environment.
"""

import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "rtdetr_r18vd_dut_anti_uav"
# Explicit user-approved deviation from official train=4 / val=8. Methods
# must all share this exact value; other official hyperparameters stay guarded.
DUT_BATCH_SIZE = 16
# Explicit user-approved common training/save policy for all DUT experiments.
DUT_EPOCHES = 200
DUT_CHECKPOINT_STEP = 10
METHODS = {
    "Baseline": "",
    "MERT": "_mert_late_xywh",
    "SECD34": "_secd_34",
    "SECD45": "_secd_45",
    "SECD345": "_secd_345",
    "SECD34+MERT": "_secd_34_mert_late_xywh",
    "SECD345+MERT": "_secd_345_mert_late_xywh",
}
PAIRINGS = [("Baseline", "MERT"), ("SECD34", "SECD34+MERT"),
            ("SECD345", "SECD345+MERT")]


def config_path(method):
    return ROOT / "configs" / "rtdetr" / (PREFIX + METHODS[method] + ".yml")


def fresh_config(path):
    """Use the real include loader, but never its mutable default accumulator."""
    loader = runpy.run_path(str(ROOT / "src" / "core" / "yaml_utils.py"))
    return loader["load_config"](str(path), {})


def flatten(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = prefix + "." + key if prefix else key
            result.update(flatten(child, path))
        if not value:
            result[prefix] = {}
    else:
        result[prefix] = value
    return result


def differences(left, right):
    left, right = flatten(left), flatten(right)
    missing = "<absent>"
    return {key: {"reference": left.get(key, missing),
                  "candidate": right.get(key, missing)}
            for key in sorted(set(left) | set(right))
            if left.get(key, missing) != right.get(key, missing)}


def permitted_difference(key, original=False):
    if key in ("__include__", "output_dir"):
        return True
    if key == "MERT" or key.startswith("MERT."):
        return True
    if key == "SECD" or key.startswith("SECD."):
        return True
    # Optional backbone branches are part of the model-selection surface, just
    # like MERT/SECD.  Their disabled defaults are allowed to be absent from
    # the untouched upstream COCO file, while every DUT method still inherits
    # the same disabled values unless its own YAML explicitly opts in.
    if key == "CCED" or key.startswith("CCED."):
        return True
    if key == "GRER" or key.startswith("GRER."):
        return True
    if key == "PDR" or key.startswith("PDR."):
        return True
    if original:
        return (key in ("num_classes", "remap_mscoco_category")
                or key == "expected_world_size"
                or key in ("BackboneVariant", "BackboneVariant.type")
                or key in ("epoches", "checkpoint_step")
                or key == "plot_training_curves"
                or key == "optimizer" or key.startswith("optimizer.")
                or key == "lr_scheduler" or key.startswith("lr_scheduler.")
                or key == "ema" or key.startswith("ema.")
                or key in ("train_dataloader.batch_size", "val_dataloader.batch_size",
                           "val_dataloader.num_workers")
                or key == "BackboneModification"
                or key.startswith("BackboneModification.")
                or key == "BackboneEnhancement"
                or key.startswith("BackboneEnhancement.")
                or key.startswith("test_dataset.")
                or key == "test_dataset"
                or key in {loader + ".dataset." + field
                           for loader in ("train_dataloader", "val_dataloader")
                           for field in ("img_folder", "ann_file")})
    return False


def resolved_audit():
    original = fresh_config(ROOT / "configs/rtdetr/rtdetr_r18vd_6x_coco.yml")
    baseline = fresh_config(config_path("Baseline"))
    official_diff = differences(original, baseline)
    failures = ["Baseline vs original: " + key for key in official_diff
                if not permitted_difference(key, original=True)]
    if baseline.get("epoches") != DUT_EPOCHES:
        failures.append(f"Baseline.epoches must be {DUT_EPOCHES}")
    if baseline.get("checkpoint_step") != DUT_CHECKPOINT_STEP:
        failures.append(f"Baseline.checkpoint_step must be {DUT_CHECKPOINT_STEP}")
    methods = {}
    for method in METHODS:
        cfg = fresh_config(config_path(method))
        for loader in ("train_dataloader", "val_dataloader"):
            if cfg[loader]["batch_size"] != DUT_BATCH_SIZE:
                failures.append(f"{method}: {loader}.batch_size must be {DUT_BATCH_SIZE}/GPU")
        diff = differences(baseline, cfg)
        failures.extend(method + " vs Baseline: " + key for key in diff
                        if not permitted_difference(key))
        methods[method] = {
            "config": str(config_path(method).relative_to(ROOT)).replace("\\", "/"),
            "baseline_differences": diff,
            "MERT": cfg.get("MERT", {"enabled": False}),
            "SECD": cfg.get("SECD", {"enabled": False}),
        }
    # The names are not enough: verify the switch and transition semantics.
    for method, info in methods.items():
        expected_mert = "MERT" in method
        expected_transitions = (["3to4", "4to5"] if "345" in method else
                                ["3to4"] if "34" in method else
                                ["4to5"] if "45" in method else [])
        if bool(info["MERT"].get("enabled", False)) != expected_mert:
            failures.append(method + ": incorrect MERT switch")
        secd = info["SECD"]
        if bool(secd.get("enabled", False)) != bool(expected_transitions):
            failures.append(method + ": incorrect SECD switch")
        if expected_transitions and sorted(secd.get("transitions", [])) != expected_transitions:
            failures.append(method + ": incorrect SECD transitions")
    return {
        "passed": not failures,
        "failures": failures,
        "baseline_original_differences": official_diff,
        "methods": methods,
        "protocol": {
            "epoches": baseline.get("epoches"),
            "checkpoint_step": baseline.get("checkpoint_step"),
            "train_batch_size_per_rank": baseline["train_dataloader"]["batch_size"],
            "val_batch_size_per_rank": baseline["val_dataloader"]["batch_size"],
            "optimizer": baseline["optimizer"],
            "lr_scheduler": baseline["lr_scheduler"],
            "multi_scale": baseline["RTDETR"]["multi_scale"],
            "train_transforms": baseline["train_dataloader"]["dataset"]["transforms"],
            "val_transforms": baseline["val_dataloader"]["dataset"]["transforms"],
            "encoder_eval_spatial_size": baseline["HybridEncoder"]["eval_spatial_size"],
            "decoder_eval_spatial_size": baseline["RTDETRTransformer"]["eval_spatial_size"],
            "PResNet_pretrained_training_policy": baseline["PResNet"]["pretrained"],
            "use_amp_yaml": baseline.get("use_amp", False),
            "use_ema": baseline.get("use_ema", False),
            "clip_max_norm": baseline.get("clip_max_norm"),
            "seed": "CLI-controlled; use the same --seed for every training run",
        },
    }


def import_model_source(selective=False):
    """Import actual source; selective mode skips package eager-import side effects."""
    sys.path.insert(0, str(ROOT))
    if not selective:
        importlib.import_module("src")
    else:
        for name, relative in [("src", "src"), ("src.nn", "src/nn"),
                               ("src.nn.backbone", "src/nn/backbone"),
                               ("src.zoo", "src/zoo"),
                               ("src.zoo.rtdetr", "src/zoo/rtdetr")]:
            package = types.ModuleType(name)
            package.__path__ = [str(ROOT / relative)]
            package.__package__ = name
            sys.modules[name] = package
        importlib.import_module("src.core")
        for name in ("src.nn.backbone.presnet", "src.zoo.rtdetr.hybrid_encoder",
                     "src.zoo.rtdetr.rtdetr_decoder", "src.zoo.rtdetr.rtdetr"):
            importlib.import_module(name)
    return importlib.import_module("src.core")


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, tensor in tensors:
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def conv_linear_hook(module, inputs, output):
    """Multiply-accumulate count, excluding bias and non-Conv/Linear arithmetic."""
    import torch.nn as nn
    if isinstance(module, nn.Conv2d):
        return output.numel() * (module.in_channels // module.groups) * (
            module.kernel_size[0] * module.kernel_size[1])
    if isinstance(module, nn.Linear):
        return output.numel() * module.in_features
    return 0


def measure_method(method, threads=4, seed=0, selective=False):
    import torch
    import torch.nn as nn
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    core = import_model_source(selective)
    cfg = fresh_config(config_path(method))
    # Only this in-memory profiling copy changes; training YAML stays untouched.
    cfg = copy.deepcopy(cfg)
    cfg["PResNet"]["pretrained"] = False
    cfg.setdefault("SECD", {"enabled": False})
    core.merge_config(cfg)
    model = core.create(cfg["model"]).cpu().eval()
    image = torch.randn(1, 3, 640, 640)
    macs = {"value": 0}
    forward_calls = {"value": 0}
    module_execution = []

    def hook(module, inputs, output):
        macs["value"] += conv_linear_hook(module, inputs, output)

    def count_forward(module, inputs):
        forward_calls["value"] += 1

    handles = [module.register_forward_hook(hook) for module in model.modules()
               if isinstance(module, (nn.Conv2d, nn.Linear))]
    for name, module in model.named_modules():
        handles.append(module.register_forward_pre_hook(
            lambda module, inputs, name=name: module_execution.append(name)))
    handles.append(model.register_forward_pre_hook(count_forward))
    try:
        with torch.no_grad(), torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True, with_flops=True) as profile:
            output = model(image)
    finally:
        for handle in handles:
            handle.remove()
    if forward_calls["value"] != 1:
        raise AssertionError("Evaluation profiling must execute one detector forward")
    if set(output) != {"pred_logits", "pred_boxes"}:
        raise AssertionError("Evaluation unexpectedly returned training/trajectory outputs")
    if any(not torch.isfinite(value).all() for value in output.values()):
        raise AssertionError("Evaluation output contains non-finite values")
    counted = {}
    uncounted = {}
    for event in profile.key_averages():
        if event.flops:
            counted[event.key] = {"flops": int(event.flops), "events": int(event.count)}
        elif event.key.startswith("aten::"):
            uncounted[event.key] = int(event.count)
    profiler_flops = sum(event["flops"] for event in counted.values())
    structure = [(name, type(module).__module__ + "." + type(module).__qualname__)
                 for name, module in model.named_modules()]
    result = {
        "method": method,
        "params": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_params": sum(parameter.numel() for parameter in model.parameters()
                                if parameter.requires_grad),
        "conv_linear_macs_lower_bound": macs["value"],
        "conv_linear_flops_lower_bound_2x_macs": 2 * macs["value"],
        "torch_profiler_flops_lower_bound": profiler_flops,
        "torch_profiler_gflops_lower_bound": profiler_flops / 1e9,
        "counted_profiler_operators": counted,
        "uncounted_operator_events_including_metadata": uncounted,
        "forward_calls": forward_calls["value"],
        "output_shapes": {name: list(value.shape) for name, value in output.items()},
        "output_sha256": tensor_digest(sorted(output.items())),
        "state_sha256": tensor_digest(sorted(model.state_dict().items())),
        "structure_sha256": hashlib.sha256(json.dumps(structure).encode()).hexdigest(),
        "module_execution_sha256": hashlib.sha256(json.dumps(module_execution).encode()).hexdigest(),
        "torch_version": torch.__version__,
        "source_import": "selective-real-model-modules" if selective else "normal-src-import",
    }
    return result


def validate_pairings(metrics):
    checks = {}
    fields = ["params", "trainable_params", "structure_sha256", "state_sha256",
              "output_shapes", "output_sha256", "forward_calls",
              "conv_linear_macs_lower_bound", "torch_profiler_flops_lower_bound",
              "counted_profiler_operators", "module_execution_sha256"]
    for left, right in PAIRINGS:
        mismatch = [field for field in fields if metrics[left][field] != metrics[right][field]]
        checks[left + " == " + right] = {
            "passed": not mismatch, "compared_fields": fields, "mismatches": mismatch,
            "uncounted_operator_event_counts_equal": (
                metrics[left]["uncounted_operator_events_including_metadata"] ==
                metrics[right]["uncounted_operator_events_including_metadata"]),
            "uncounted_events_note": "Diagnostic only: PyTorch 2.0 Windows event counts "
                                     "can vary even for repeats of the same model/config. "
                                     "Actual module execution trace is compared separately.",
        }
        if mismatch:
            raise AssertionError(left + " != " + right + ": " + ", ".join(mismatch))
    return checks


def run_worker(method, args):
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", method,
               "--threads", str(args.threads), "--seed", str(args.seed)]
    if args.model_only_import:
        command.append("--model-only-import")
    process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if process.returncode:
        raise RuntimeError(method + " measurement failed:\n" + process.stdout + process.stderr)
    marker = "DUT_METRICS_JSON="
    lines = [line[len(marker):] for line in process.stdout.splitlines() if line.startswith(marker)]
    if len(lines) != 1:
        raise RuntimeError("No unique worker result for " + method + "\n" + process.stdout)
    return json.loads(lines[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-only", action="store_true", help="No torch/model imports")
    parser.add_argument("--model-only-import", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=list(METHODS), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.worker:
        result = measure_method(args.worker, args.threads, args.seed, args.model_only_import)
        print("DUT_METRICS_JSON=" + json.dumps(result))
        return
    result = {"resolved_audit": resolved_audit()}
    if not result["resolved_audit"]["passed"]:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit("Resolved protocol differs outside permitted DUT/method fields")
    if not args.resolved_only:
        result["measurement_protocol"] = {
            "input": [1, 3, 640, 640], "device": "cpu", "dtype": "float32",
            "model_mode": "eval, non-deploy (same as original validation/test)",
            "AMP": False, "seed": args.seed, "threads": args.threads,
            "pretrained": "Disabled ONLY in profiling memory; no download or YAML edits",
            "weights": "random initialization; not accuracy evaluation",
            "scope": "One detector forward; no transforms/postprocessor/criterion/MERT training",
            "environment_scope": "Selective source import does NOT verify dataset or full "
                                 "training dependencies; no dependency/model stubs are used.",
            "model_process_isolation": "Fresh subprocess per method avoids global registry leakage",
            "flops_warning": "LOWER BOUNDS, NOT total model FLOPs. Profiler zero-FLOP operators "
                             "include grid sampling, normalization, softmax, fused attention, "
                             "and some SECD arithmetic; operator list also includes metadata. "
                             "Hook MACs count only executed Conv2d/Linear modules, not functional "
                             "attention projections. 2*MACs is stated only for the hook metric.",
        }
        result["metrics"] = {}
        for method in METHODS:
            print("Measuring " + method + " (real eval forward, 640 FP32)...", flush=True)
            result["metrics"][method] = run_worker(method, args)
        result["inference_pair_equivalence"] = validate_pairings(result["metrics"])
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print("Saved " + str(args.output.resolve()))
    else:
        print(rendered)


if __name__ == "__main__":
    main()
