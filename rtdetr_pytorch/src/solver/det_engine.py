"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

by lyuwenyu
"""

import math
import os
import sys
import pathlib
from typing import Iterable

import torch
import torch.amp 

from src.data import CocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)
from src.misc.dist import de_parallel
from src.misc.amp import autocast_context
from src.misc.inference_audit import InferenceAudit
from .mert import MERTTrainingPlugin, average_loss_dicts
from .cter_loss import CTERTrainingPlugin


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)
    mert_config = kwargs.get('mert_config') or {}
    mert = MERTTrainingPlugin(
        mert_config, criterion.matcher, current_epoch=epoch
    ) if mert_config.get('enabled', False) else None
    cter_config = kwargs.get('cter_config') or {}
    cter = None
    if cter_config.get('enabled', False):
        backbone = de_parallel(model).backbone
        feature_strides = getattr(backbone, 'out_strides', None)
        return_indices = getattr(backbone, 'return_idx', None)
        if feature_strides is None or return_indices is None:
            raise ValueError('CTER needs backbone.out_strides and backbone.return_idx metadata')
        stage_names = tuple('s{}'.format(index + 2) for index in return_indices)
        cter = CTERTrainingPlugin(cter_config, feature_strides, stage_names)
        if cter.debug:
            print('CTER training debug: stages={}, strides={}, model views={}'.format(
                stage_names, feature_strides, 2 if mert is not None else 1
            ))
    print('Training AMP: {}'.format(
        'enabled' if scaler is not None and device.type == 'cuda' else 'disabled'
    ))

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        pair = mert.prepare(model, samples, targets) if mert is not None else None
        original_backbone_features = None
        original_image_size = None

        def forward_original(images, image_targets):
            nonlocal original_backbone_features, original_image_size
            if cter is None:
                return model(images, image_targets)
            outputs, original_backbone_features, original_image_size = model(
                images, image_targets, return_backbone_features=True
            )
            return outputs

        def forward_train_views():
            nonlocal original_backbone_features, original_image_size
            if mert is None:
                return forward_original(samples, targets)

            with mert.disable_internal_multiscale(model):
                if mert.forward_mode == 'concat':
                    paired_images = torch.cat([
                        pair['original_images'], pair['shifted_images']
                    ], dim=0)
                    paired_targets = pair['original_targets'] + pair['shifted_targets']
                    if cter is None:
                        # The original MERT-only forward call is unchanged.
                        return model(paired_images, paired_targets)
                    outputs, paired_features, original_image_size = model(
                        paired_images, paired_targets, return_backbone_features=True
                    )
                    original_backbone_features = [
                        feature[:samples.shape[0]] for feature in paired_features
                    ]
                    return outputs

                outputs_original = forward_original(
                    pair['original_images'], pair['original_targets']
                )
                outputs_shifted = model(
                    pair['shifted_images'], pair['shifted_targets']
                )
                return outputs_original, outputs_shifted

        def compute_train_losses(outputs):
            if mert is None:
                return criterion(outputs, targets)

            if mert.forward_mode == 'concat':
                paired_targets = pair['original_targets'] + pair['shifted_targets']
                outputs_original, outputs_shifted = \
                    mert.split_concatenated_outputs(outputs, samples.shape[0])
                if mert.shifted_detection_loss:
                    loss_dict = criterion(outputs, paired_targets)
                else:
                    loss_dict = criterion(
                        outputs_original, pair['original_targets']
                    )
            else:
                outputs_original, outputs_shifted = outputs
                if mert.shifted_detection_loss:
                    loss_dict = average_loss_dicts(
                        criterion(outputs_original, pair['original_targets']),
                        criterion(outputs_shifted, pair['shifted_targets']),
                    )
                else:
                    loss_dict = criterion(
                        outputs_original, pair['original_targets']
                    )

            loss_dict['loss_mert'] = mert.calculate_loss(
                outputs_original, outputs_shifted, pair
            )
            return loss_dict

        def compute_all_losses(outputs):
            loss_dict = compute_train_losses(outputs)
            if cter is not None:
                # Only original GT and original pre-encoder features enter
                # CTER. No pair metadata, shifted GT, or decoder output enters.
                loss_dict['loss_cter'] = cter.calculate_loss(
                    original_backbone_features, targets, original_image_size
                )
            return loss_dict

        if scaler is not None:
            with autocast_context(device, enabled=True):
                outputs = forward_train_views()
            
            with autocast_context(device, enabled=False):
                loss_dict = compute_all_losses(outputs)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = forward_train_views()
            loss_dict = compute_all_losses(outputs)
            
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if cter is not None:
        cter_debug = cter.debug_summary()
        if cter_debug:
            print('CTER epoch debug:', cter_debug)
            stats.update(cter_debug)
    return stats



@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors,
             data_loader, base_ds, device, output_dir, amp_enabled=False,
             debug_eval_amp=False):
    model.eval()
    criterion.eval()
    device = torch.device(device)
    eval_amp_enabled = bool(amp_enabled and device.type == 'cuda')
    print('Evaluation AMP: {}'.format('enabled' if eval_amp_enabled else 'disabled'))

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    # if 'panoptic' in postprocessors.keys():
    #     panoptic_evaluator = PanopticEvaluator(
    #         data_loader.dataset.ann_file,
    #         data_loader.dataset.ann_folder,
    #         output_dir=os.path.join(output_dir, "panoptic_eval"),
    #     )

    for batch_index, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, 10, header)
    ):
        samples = samples.to(device, non_blocking=True)
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        if debug_eval_amp and batch_index == 0:
            with InferenceAudit(model) as audit:
                with autocast_context(device, enabled=eval_amp_enabled):
                    outputs = model(samples)
            audit.report(expected_amp=eval_amp_enabled)
        else:
            with autocast_context(device, enabled=eval_amp_enabled):
                outputs = model(samples)

        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict
        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        
        results = postprocessors(outputs, orig_target_sizes)
        # results = postprocessors(outputs, targets)

        # if 'segm' in postprocessors.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # if panoptic_evaluator is not None:
        #     res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
        #     for i, target in enumerate(targets):
        #         image_id = target["image_id"].item()
        #         file_name = f"{image_id:012d}.png"
        #         res_pano[i]["image_id"] = image_id
        #         res_pano[i]["file_name"] = file_name
        #     panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    # panoptic_res = None
    # if panoptic_evaluator is not None:
    #     panoptic_res = panoptic_evaluator.summarize()
    
    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            
    # if panoptic_res is not None:
    #     stats['PQ_all'] = panoptic_res["All"]
    #     stats['PQ_th'] = panoptic_res["Things"]
    #     stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator



