"""Paired contrast-guided deformable Decoder Cross-Attention."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import deformable_attention_sample_func


__all__ = [
    'PairedContrastDeformableAttention',
    'build_decoder_cross_attention',
]


class PairedContrastDeformableAttention(nn.Module):
    """Recalibrate target-point attention using paired surround contrast.

    The surround branch is only a comparator.  Final aggregation always uses
    the original target sampling features and never adds or subtracts surround
    features from the Object Query.
    """

    accepts_paired_contrast_config = True

    def __init__(self,
                 base_attention,
                 layer_index,
                 num_matching_queries,
                 enabled_levels=(0, 1),
                 surround_margin=0.20,
                 contrast_lambda=0.5,
                 normalize_feature=True,
                 small_query_enabled=False,
                 small_query_area_threshold=0.02,
                 enable_surround_for_dn=False,
                 debug=False,
                 eps=1e-6):
        super().__init__()
        required_attributes = (
            'embed_dim', 'num_heads', 'num_levels', 'num_points', 'head_dim',
            'sampling_offsets', 'attention_weights', 'value_proj',
            'output_proj', 'ms_deformable_attn_core')
        missing = [
            name for name in required_attributes
            if not hasattr(base_attention, name)
        ]
        if missing:
            raise TypeError(
                'base_attention is incompatible; missing: {}'.format(
                    ', '.join(missing)))
        if surround_margin <= 0:
            raise ValueError('surround margin must be greater than zero.')
        if contrast_lambda < 0:
            raise ValueError('contrast lambda must be non-negative.')
        if num_matching_queries <= 0:
            raise ValueError('num_matching_queries must be positive.')

        self.embed_dim = base_attention.embed_dim
        self.num_heads = base_attention.num_heads
        self.num_levels = base_attention.num_levels
        self.num_points = base_attention.num_points
        self.total_points = self.num_heads * self.num_levels * self.num_points
        self.head_dim = base_attention.head_dim

        # Adopt the original modules directly so every checkpoint parameter
        # keeps its exact baseline path and initialization.
        self.sampling_offsets = base_attention.sampling_offsets
        self.attention_weights = base_attention.attention_weights
        self.value_proj = base_attention.value_proj
        self.output_proj = base_attention.output_proj
        self.ms_deformable_attn_core = \
            base_attention.ms_deformable_attn_core
        # Calling this unbound method with ``self`` gives a bitwise-identical
        # original path for lambda=0 without retaining a nested base module.
        self._original_forward = type(base_attention).forward

        self.layer_index = int(layer_index)
        self.num_matching_queries = int(num_matching_queries)
        configured_levels = tuple(int(level) for level in enabled_levels)
        if len(configured_levels) != len(set(configured_levels)):
            raise ValueError('enabled_levels must not contain duplicates.')
        self.enabled_levels = tuple(sorted(configured_levels))
        self.surround_margin = float(surround_margin)
        self.contrast_lambda = float(contrast_lambda)
        self.normalize_feature = bool(normalize_feature)
        self.small_query_enabled = bool(small_query_enabled)
        self.small_query_area_threshold = float(
            small_query_area_threshold)
        self.enable_surround_for_dn = bool(enable_surround_for_dn)
        self.debug = bool(debug)
        self.eps = float(eps)

        if not self.enabled_levels:
            raise ValueError('enabled_levels must contain at least one level.')
        for level in self.enabled_levels:
            if level < 0 or level >= self.num_levels:
                raise ValueError(
                    'Invalid enabled level {} for {} levels.'.format(
                        level, self.num_levels))

        self.last_debug_info = {}
        self._debug_recorded = False

    def _get_target_locations(self, reference_points, sampling_offsets):
        """Use the original 4-D box-aware target sampling equation."""
        if reference_points.shape[-1] != 4:
            raise ValueError(
                'Paired box-outside sampling requires 4-D reference boxes; '
                'got last dimension {}.'.format(reference_points.shape[-1]))
        return (
            reference_points[:, :, None, :, None, :2] +
            sampling_offsets / self.num_points *
            reference_points[:, :, None, :, None, 2:] * 0.5)

    def _build_surround_locations(self, reference_points, target_locations,
                                  sampling_offsets):
        """Intersect center-to-target rays with boxes, then move outside."""
        target_enabled = target_locations[:, :, :, self.enabled_levels, :, :]
        center = reference_points[:, :, None, :, None, :2]
        half_size = reference_points[:, :, None, :, None, 2:] * 0.5

        direction = target_enabled - center
        direction_norm = torch.linalg.vector_norm(direction, dim=-1)
        direction_valid = direction_norm >= self.eps

        # If a target point is numerically indistinguishable from the center,
        # fall back to the corresponding learned offset direction.  If that is
        # also degenerate, the point remains invalid and contributes no score.
        offset_direction = sampling_offsets[
            :, :, :, self.enabled_levels, :, :]
        offset_valid = torch.linalg.vector_norm(
            offset_direction, dim=-1) >= self.eps
        use_offset_direction = (~direction_valid) & offset_valid
        direction = torch.where(
            use_offset_direction.unsqueeze(-1), offset_direction, direction)
        direction_valid = direction_valid | use_offset_direction

        lambda_x = half_size[..., 0] / \
            direction[..., 0].abs().clamp_min(self.eps)
        lambda_y = half_size[..., 1] / \
            direction[..., 1].abs().clamp_min(self.eps)
        boundary_scale = torch.minimum(lambda_x, lambda_y)
        raw_surround = center + (
            (1.0 + self.surround_margin) *
            boundary_scale.unsqueeze(-1) * direction)

        image_valid = ((raw_surround >= 0.) &
                       (raw_surround <= 1.)).all(dim=-1)
        box_valid = ((half_size[..., 0] >= self.eps) &
                     (half_size[..., 1] >= self.eps))
        valid = direction_valid & image_valid & box_valid

        # grid_sample itself accepts out-of-range grids, but a safe clamp keeps
        # export/backends stable.  ``valid`` guarantees clamped invalid samples
        # contribute zero contrast and therefore cannot alter attention.
        safe_surround = raw_surround.clamp(0., 1.)
        return target_enabled, raw_surround, safe_surround, valid, center, \
            half_size

    def _get_query_eligibility(self, reference_points, query_length, device):
        eligible = torch.ones(
            reference_points.shape[0], query_length,
            dtype=torch.bool, device=device)

        num_dn_queries = max(query_length - self.num_matching_queries, 0)
        if not self.enable_surround_for_dn and num_dn_queries > 0:
            eligible[:, :num_dn_queries] = False

        if self.small_query_enabled:
            # This project supplies one shared reference box across all levels.
            reference_area = (reference_points[..., 2] *
                              reference_points[..., 3])
            if reference_area.shape[-1] != 1:
                raise ValueError(
                    'Expected one reference box per query, got {}.'.format(
                        reference_area.shape[-1]))
            eligible &= reference_area[..., 0] < \
                self.small_query_area_threshold
        return eligible, num_dn_queries

    def _to_point_features(self, sampled_feature, batch_size, query_length):
        """[B*H,D,Q,P] -> [B,Q,H,P,D]."""
        return sampled_feature.reshape(
            batch_size, self.num_heads, self.head_dim,
            query_length, self.num_points).permute(0, 3, 1, 4, 2)

    def _aggregate_target(self, target_samples, attention_weights,
                          batch_size, query_length):
        """Use the original weighted-sum layout on target features only."""
        attention_weights = attention_weights.permute(
            0, 2, 1, 3, 4).reshape(
                batch_size * self.num_heads, 1, query_length,
                self.num_levels * self.num_points)
        output = (torch.stack(target_samples, dim=-2).flatten(-2) *
                  attention_weights).sum(-1).reshape(
                      batch_size, self.num_heads * self.head_dim,
                      query_length)
        return output.permute(0, 2, 1)

    @staticmethod
    def _masked_stats(values, mask):
        selected = values.masked_select(mask)
        if selected.numel() == 0:
            return 0., 0., 0.
        selected = selected.detach().float()
        return (
            float(selected.mean().cpu()),
            float(selected.std(unbiased=False).cpu()),
            float(selected.max().cpu()),
        )

    def _record_debug_info(self,
                           reference_points,
                           target_locations,
                           raw_surround_locations,
                           target_point_features,
                           surround_point_features,
                           active_contrast_scores,
                           contrast_scores,
                           active_mask,
                           candidate_mask,
                           center,
                           half_size,
                           original_attention_weights,
                           new_attention_weights,
                           output,
                           num_dn_queries):
        if not self.debug or self._debug_recorded:
            return

        candidate_count = int(candidate_mask.sum().detach().cpu())
        valid_count = int(active_mask.sum().detach().cpu())
        valid_ratio = valid_count / max(candidate_count, 1)

        relative_surround = (raw_surround_locations - center).abs()
        inside_box = ((relative_surround[..., 0] <= half_size[..., 0]) &
                      (relative_surround[..., 1] <= half_size[..., 1]))
        inside_valid = inside_box & active_mask
        inside_ratio = float(
            inside_valid.sum().detach().cpu()) / max(
                valid_count, 1)

        contrast_mean, contrast_std, contrast_max = self._masked_stats(
            active_contrast_scores[:, :, :, self.enabled_levels, :],
            active_mask)
        original_flat = original_attention_weights.flatten(-2)
        new_flat = new_attention_weights.flatten(-2)
        original_entropy = -(
            original_flat * original_flat.clamp_min(self.eps).log()
        ).sum(-1).mean()
        new_entropy = -(
            new_flat * new_flat.clamp_min(self.eps).log()
        ).sum(-1).mean()

        query_sample_count = min(8, reference_points.shape[1])
        self.last_debug_info = {
            'layer_index': self.layer_index,
            'enabled_levels': self.enabled_levels,
            'num_denoising_queries': num_dn_queries,
            'target_sampling_locations_shape': tuple(target_locations.shape),
            'surround_sampling_locations_shape': tuple(
                raw_surround_locations.shape),
            'target_features_shape': tuple(target_point_features.shape),
            'surround_features_shape': tuple(
                surround_point_features.shape),
            'contrast_score_shape': tuple(active_contrast_scores.shape),
            'full_contrast_score_shape': tuple(contrast_scores.shape),
            'output_shape': tuple(output.shape),
            'contrast_mean': contrast_mean,
            'contrast_std': contrast_std,
            'contrast_max': contrast_max,
            'valid_surround_ratio': valid_ratio,
            'invalid_surround_ratio': 1.0 - valid_ratio,
            'reference_box_area_mean': float((
                reference_points[..., 2] *
                reference_points[..., 3]).detach().mean().cpu()),
            'original_attention_entropy': float(
                original_entropy.detach().cpu()),
            'new_attention_entropy': float(new_entropy.detach().cpu()),
            'attention_delta_mean': float((
                new_attention_weights - original_attention_weights
            ).detach().abs().mean().cpu()),
            'surround_inside_reference_box_ratio': inside_ratio,
            'reference_points': reference_points[
                :1, :query_sample_count].detach().cpu(),
            'reference_boxes': reference_points[
                :1, :query_sample_count].detach().cpu(),
            'target_sampling_locations': target_locations[
                :1, :query_sample_count].detach().cpu(),
            'surround_sampling_locations': raw_surround_locations[
                :1, :query_sample_count].detach().cpu(),
            'original_attention_weights': original_attention_weights[
                :1, :query_sample_count].detach().cpu(),
            'new_attention_weights': new_attention_weights[
                :1, :query_sample_count].detach().cpu(),
            'contrast_scores': active_contrast_scores[
                :1, :query_sample_count].detach().cpu(),
            'full_contrast_scores': contrast_scores[
                :1].detach().cpu(),
            'valid_surround_mask': active_mask[
                :1, :query_sample_count].detach().cpu(),
        }

        distributed = torch.distributed.is_available() and \
            torch.distributed.is_initialized()
        is_main_process = not distributed or torch.distributed.get_rank() == 0
        if is_main_process:
            print(
                '[PairedContrastDeformableAttention][layer={}] '
                'target_locations={} surround_locations={} target_features={} '
                'surround_features={} contrast={} output={} '
                'contrast(mean/std/max)=({:.6f}/{:.6f}/{:.6f}) '
                'valid/invalid=({:.4f}/{:.4f}) area_mean={:.6f} '
                'entropy(original/new)=({:.6f}/{:.6f}) '
                'attention_delta_mean={:.6f} surround_inside_box={:.6f}'
                .format(
                    self.layer_index,
                    tuple(target_locations.shape),
                    tuple(raw_surround_locations.shape),
                    tuple(target_point_features.shape),
                    tuple(surround_point_features.shape),
                    tuple(active_contrast_scores.shape), tuple(output.shape),
                    contrast_mean, contrast_std, contrast_max,
                    valid_ratio, 1.0 - valid_ratio,
                    self.last_debug_info['reference_box_area_mean'],
                    self.last_debug_info['original_attention_entropy'],
                    self.last_debug_info['new_attention_entropy'],
                    self.last_debug_info['attention_delta_mean'], inside_ratio))
        self._debug_recorded = True

    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_mask=None):
        # Lambda zero is an explicit numerical-equivalence path: it calls the
        # original class's exact forward implementation and does no surround
        # sampling or contrast computation.
        if self.contrast_lambda == 0.:
            return self._original_forward(
                self, query, reference_points, value,
                value_spatial_shapes, value_mask)

        batch_size, query_length = query.shape[:2]
        value_length = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value = value * value_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(
            batch_size, value_length, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).reshape(
            batch_size, query_length, self.num_heads,
            self.num_levels, self.num_points, 2)
        attention_logits = self.attention_weights(query).reshape(
            batch_size, query_length, self.num_heads,
            self.num_levels, self.num_points)
        original_attention_weights = F.softmax(
            attention_logits.flatten(-2), dim=-1).reshape_as(
                attention_logits)

        target_locations = self._get_target_locations(
            reference_points, sampling_offsets)
        query_eligible, num_dn_queries = self._get_query_eligibility(
            reference_points, query_length, query.device)
        contrast_query_start = 0 if self.enable_surround_for_dn \
            else num_dn_queries
        contrast_reference_points = reference_points[
            :, contrast_query_start:]
        (target_enabled_locations,
         raw_surround_locations,
         safe_surround_locations,
         geometry_valid,
         center,
         half_size) = self._build_surround_locations(
             contrast_reference_points,
             target_locations[:, contrast_query_start:],
             sampling_offsets[:, contrast_query_start:])

        target_samples = deformable_attention_sample_func(
            value, value_spatial_shapes, target_locations)
        surround_samples = deformable_attention_sample_func(
            value, value_spatial_shapes, safe_surround_locations,
            enabled_levels=self.enabled_levels)

        candidate_mask = query_eligible[:, contrast_query_start:][
            :, :, None, None, None].expand_as(geometry_valid)
        active_mask = geometry_valid & candidate_mask
        contrast_query_length = query_length - contrast_query_start

        contrast_by_level = []
        surround_point_features = []
        enabled_position = {
            level: position
            for position, level in enumerate(self.enabled_levels)
        }
        for level in range(self.num_levels):
            if level not in enabled_position:
                contrast_by_level.append(attention_logits.new_zeros(
                    batch_size, contrast_query_length,
                    self.num_heads, self.num_points))
                continue

            position = enabled_position[level]
            target_points = self._to_point_features(
                target_samples[level], batch_size, query_length)[
                    :, contrast_query_start:]
            surround_points = self._to_point_features(
                surround_samples[position], batch_size,
                query_length - contrast_query_start)
            surround_point_features.append(surround_points)
            if self.normalize_feature:
                target_for_contrast = F.normalize(
                    target_points, dim=-1, eps=self.eps)
                surround_for_contrast = F.normalize(
                    surround_points, dim=-1, eps=self.eps)
            else:
                target_for_contrast = target_points
                surround_for_contrast = surround_points

            contrast = (target_for_contrast -
                        surround_for_contrast).abs().mean(dim=-1)
            contrast = torch.where(
                active_mask[:, :, :, position, :],
                contrast, torch.zeros_like(contrast))
            contrast_by_level.append(contrast)

        active_contrast_scores = torch.stack(contrast_by_level, dim=3)
        if contrast_query_start > 0:
            dn_contrast_scores = attention_logits.new_zeros(
                batch_size, contrast_query_start, self.num_heads,
                self.num_levels, self.num_points)
            contrast_scores = torch.cat(
                [dn_contrast_scores, active_contrast_scores], dim=1)
        else:
            contrast_scores = active_contrast_scores
        corrected_logits = attention_logits + \
            self.contrast_lambda * contrast_scores
        new_attention_weights = F.softmax(
            corrected_logits.flatten(-2), dim=-1).reshape_as(
                corrected_logits)

        # Surround features stop here.  Only target samples enter aggregation.
        output = self._aggregate_target(
            target_samples, new_attention_weights,
            batch_size, query_length)
        output = self.output_proj(output)

        if self.debug and not self._debug_recorded:
            target_point_features = torch.stack([
                self._to_point_features(
                    sampled, batch_size, query_length)
                [:, contrast_query_start:]
                for sampled in target_samples
            ], dim=3)
            surround_point_features = torch.stack(
                surround_point_features, dim=3)
            self._record_debug_info(
                contrast_reference_points,
                target_enabled_locations,
                raw_surround_locations,
                target_point_features,
                surround_point_features,
                active_contrast_scores,
                contrast_scores,
                active_mask,
                candidate_mask,
                center,
                half_size,
                original_attention_weights[:, contrast_query_start:],
                new_attention_weights[:, contrast_query_start:],
                output,
                num_dn_queries)
        return output


def _pop_section(config, name):
    section = config.pop(name, {})
    if not isinstance(section, dict):
        raise TypeError('{} must be a dict.'.format(name))
    return copy.deepcopy(section)


def build_decoder_cross_attention(base_attention,
                                  config,
                                  layer_index,
                                  num_decoder_layers,
                                  num_matching_queries):
    """Return original attention or the configured paired implementation."""
    if config is None:
        return base_attention
    if not isinstance(config, dict):
        raise TypeError('decoder_cross_attention must be a dict or None.')

    config = copy.deepcopy(config)
    enabled = bool(config.pop('enabled', True))
    attention_type = str(config.pop('type', 'paired_contrast')).lower()
    baseline_names = {
        'ms_deform_attn', 'ms_deformable_attention',
        'msdeformableattention', 'baseline', 'original'
    }
    if not enabled or attention_type in baseline_names:
        return base_attention
    if attention_type not in {
            'paired_contrast', 'paired_contrast_deformable_attention',
            'pairedcontrastdeformableattention'}:
        raise ValueError(
            'Unsupported Decoder Cross-Attention type: {}.'.format(
                attention_type))

    surround = _pop_section(config, 'surround')
    contrast = _pop_section(config, 'contrast')
    layers = _pop_section(config, 'layers')
    levels = _pop_section(config, 'levels')
    small_query = _pop_section(config, 'small_query')
    denoising = _pop_section(config, 'denoising')
    invalid_point = _pop_section(config, 'invalid_point')
    debug = bool(config.pop('debug', False))
    if config:
        raise ValueError(
            'Unknown decoder_cross_attention options: {}.'.format(
                sorted(config.keys())))

    surround_mode = surround.pop('mode', 'box_outside_radial')
    surround_margin = float(surround.pop('margin', 0.20))
    contrast_mode = contrast.pop('mode', 'abs_difference')
    contrast_lambda = float(contrast.pop('lambda', 0.5))
    normalize_feature = bool(contrast.pop('normalize_feature', True))
    contrast_scorer = contrast.pop('scorer', 'mean')
    enabled_layers = layers.pop('enabled_layers', [-2, -1])
    enabled_levels = levels.pop('enabled_levels', [0, 1])
    small_query_enabled = bool(small_query.pop('enabled', False))
    small_query_area_threshold = float(
        small_query.pop('area_threshold', 0.02))
    enable_surround_for_dn = bool(
        denoising.pop('enable_surround_for_dn', False))
    invalid_mode = invalid_point.pop('mode', 'mask')

    unknown_nested = {
        name: sorted(section.keys())
        for name, section in (
            ('surround', surround), ('contrast', contrast),
            ('layers', layers), ('levels', levels),
            ('small_query', small_query), ('denoising', denoising),
            ('invalid_point', invalid_point)) if section
    }
    if unknown_nested:
        raise ValueError('Unknown nested options: {}.'.format(unknown_nested))
    if surround_mode != 'box_outside_radial':
        raise ValueError('Only surround.mode=box_outside_radial is supported.')
    if contrast_mode != 'abs_difference':
        raise ValueError('Only contrast.mode=abs_difference is supported.')
    if contrast_scorer != 'mean':
        raise ValueError('Only contrast.scorer=mean is supported.')
    if invalid_mode != 'mask':
        raise ValueError('Only invalid_point.mode=mask is supported.')

    resolved_layers = set()
    for index in enabled_layers:
        index = int(index)
        resolved = index if index >= 0 else num_decoder_layers + index
        if resolved < 0 or resolved >= num_decoder_layers:
            raise ValueError(
                'Invalid Decoder layer {} for {} layers.'.format(
                    index, num_decoder_layers))
        resolved_layers.add(resolved)
    if layer_index not in resolved_layers:
        return base_attention

    return PairedContrastDeformableAttention(
        base_attention=base_attention,
        layer_index=layer_index,
        num_matching_queries=num_matching_queries,
        enabled_levels=enabled_levels,
        surround_margin=surround_margin,
        contrast_lambda=contrast_lambda,
        normalize_feature=normalize_feature,
        small_query_enabled=small_query_enabled,
        small_query_area_threshold=small_query_area_threshold,
        enable_surround_for_dn=enable_surround_for_dn,
        debug=debug)
