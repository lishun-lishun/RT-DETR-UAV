"""Paired target-surround deformable cross-attention.

This module intentionally reuses the projections, sampling-offset predictor,
attention-weight predictor, and sampling core from the original
MSDeformableAttention instance.  Only the Decoder Cross-Attention sampling and
fusion path is changed.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    'PairedTargetSurroundDeformableAttention',
    'build_decoder_cross_attention',
]


class _CheckpointCompatibleGate(nn.Linear):
    """Linear layer whose new parameters are optional in old checkpoints."""

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # A baseline RT-DETR checkpoint has no differential gate.  Supplying
        # the initialized tensors here lets strict model-only loading retain
        # every compatible baseline tensor and initialize only this new gate.
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        if weight_key not in state_dict:
            state_dict[weight_key] = self.weight.detach().clone()
        if self.bias is not None and bias_key not in state_dict:
            state_dict[bias_key] = self.bias.detach().clone()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class PairedTargetSurroundDeformableAttention(nn.Module):
    """Differential target-surround variant of Decoder Cross-Attention.

    Args:
        base_attention: An initialized original MSDeformableAttention.  Its
            learned submodules are adopted directly so their state-dict names
            and initialization remain compatible with original checkpoints.
        surround_scale: Fixed radial scale ``rho`` used to construct surround
            locations from target locations and the reference center.
        differential_enabled: Use ``z_target - z_surround`` as evidence.  If
            false, use ``z_surround`` for a Target + Surround ablation.
        gated_fusion_enabled: Learn a scalar gate per query.  If false, fuse
            the evidence without a gate.
        gate_init_bias: Initial bias of the scalar gate.
        debug: Cache and print first-forward sampling diagnostics.
    """

    def __init__(self,
                 base_attention,
                 surround_scale=1.8,
                 differential_enabled=True,
                 gated_fusion_enabled=True,
                 gate_init_bias=-2.0,
                 debug=False):
        super().__init__()

        if surround_scale <= 0:
            raise ValueError(
                'surround_scale must be positive, but got {}.'.format(
                    surround_scale))

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
                'base_attention is not a compatible MSDeformableAttention; '
                'missing attributes: {}'.format(', '.join(missing)))

        self.embed_dim = base_attention.embed_dim
        self.num_heads = base_attention.num_heads
        self.num_levels = base_attention.num_levels
        self.num_points = base_attention.num_points
        self.total_points = self.num_heads * self.num_levels * self.num_points
        self.head_dim = base_attention.head_dim

        # Keep the original parameter names exactly.  The temporary base
        # module itself is deliberately not retained as a nested child.
        self.sampling_offsets = base_attention.sampling_offsets
        self.attention_weights = base_attention.attention_weights
        self.value_proj = base_attention.value_proj
        self.output_proj = base_attention.output_proj
        self.ms_deformable_attn_core = \
            base_attention.ms_deformable_attn_core

        self.surround_scale = float(surround_scale)
        self.differential_enabled = bool(differential_enabled)
        self.gated_fusion_enabled = bool(gated_fusion_enabled)
        self.debug = bool(debug)

        if self.gated_fusion_enabled:
            self.diff_gate = _CheckpointCompatibleGate(self.embed_dim, 1)
            nn.init.constant_(self.diff_gate.weight, 0.)
            nn.init.constant_(self.diff_gate.bias, float(gate_init_bias))
        else:
            # Do not create unused trainable parameters in the ungated
            # ablation; the project trains with find_unused_parameters=False.
            self.diff_gate = None

        self.last_debug_info = {}
        self._debug_recorded = False

    def _get_target_locations(self, reference_points, sampling_offsets,
                              value_spatial_shapes):
        """Build the original MSDeformableAttention sampling locations."""
        bs, len_q = reference_points.shape[:2]
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.as_tensor(
                value_spatial_shapes,
                dtype=sampling_offsets.dtype,
                device=sampling_offsets.device)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                1, 1, 1, self.num_levels, 1, 2)
            return reference_points.reshape(
                bs, len_q, 1, self.num_levels, 1, 2
            ) + sampling_offsets / offset_normalizer

        if reference_points.shape[-1] == 4:
            return (
                reference_points[:, :, None, :, None, :2] +
                sampling_offsets / self.num_points *
                reference_points[:, :, None, :, None, 2:] * 0.5)

        raise ValueError(
            'Last dim of reference_points must be 2 or 4, but got {}.'.format(
                reference_points.shape[-1]))

    @staticmethod
    def _get_reference_centers(reference_points):
        # [B, Nq, Lref, 2] -> [B, Nq, 1, Lref, 1, 2].  Lref may be one and
        # broadcasts over all feature levels, as in the original decoder.
        return reference_points[..., :2][:, :, None, :, None, :]

    def _record_debug_info(self, query, reference_points, target_locations,
                           surround_locations, z_target, z_surround, z_diff,
                           z_evidence, gate, output, clamp_ratio):
        if not self.debug or self._debug_recorded:
            return

        query_sample_count = min(8, target_locations.shape[1])
        self.last_debug_info = {
            'query_shape': tuple(query.shape),
            'reference_points_shape': tuple(reference_points.shape),
            'target_locations_shape': tuple(target_locations.shape),
            'surround_locations_shape': tuple(surround_locations.shape),
            'z_target_shape': tuple(z_target.shape),
            'z_surround_shape': tuple(z_surround.shape),
            'z_diff_shape': tuple(z_diff.shape),
            'gate_shape': None if gate is None else tuple(gate.shape),
            'output_shape': tuple(output.shape),
            'surround_scale': self.surround_scale,
            'surround_clamp_ratio': float(clamp_ratio.detach().cpu()),
            'target_location_min': float(target_locations.detach().min().cpu()),
            'target_location_max': float(target_locations.detach().max().cpu()),
            'surround_location_min': float(
                surround_locations.detach().min().cpu()),
            'surround_location_max': float(
                surround_locations.detach().max().cpu()),
            'z_target_abs_mean': float(z_target.detach().abs().mean().cpu()),
            'z_surround_abs_mean': float(
                z_surround.detach().abs().mean().cpu()),
            'z_diff_abs_mean': float(z_diff.detach().abs().mean().cpu()),
            # Small CPU snapshots are sufficient for later visualization and
            # avoid retaining a training graph or a full batch of coordinates.
            'reference_boxes': reference_points[
                :1, :query_sample_count].detach().cpu(),
            'target_locations': target_locations[
                :1, :query_sample_count].detach().cpu(),
            'surround_locations': surround_locations[
                :1, :query_sample_count].detach().cpu(),
            'z_diff': z_diff[:1, :query_sample_count].detach().cpu(),
            'differential_evidence': z_evidence[
                :1, :query_sample_count].detach().cpu(),
        }
        if gate is not None:
            gate_detached = gate.detach()
            self.last_debug_info.update({
                'gate': gate_detached[:1, :query_sample_count].cpu(),
                'gate_mean': float(gate_detached.mean().cpu()),
                'gate_min': float(gate_detached.min().cpu()),
                'gate_max': float(gate_detached.max().cpu()),
            })

        distributed = torch.distributed.is_available() and \
            torch.distributed.is_initialized()
        is_main_process = not distributed or torch.distributed.get_rank() == 0
        if is_main_process:
            gate_stats = 'disabled' if gate is None else \
                'mean={:.4f}, min={:.4f}, max={:.4f}'.format(
                    self.last_debug_info['gate_mean'],
                    self.last_debug_info['gate_min'],
                    self.last_debug_info['gate_max'])
            print(
                '[PairedTargetSurroundDeformableAttention] '
                'query={} reference={} target_locations={} '
                'surround_locations={} z_target={} z_surround={} z_diff={} '
                'gate={} output={} target_range=({:.4f}, {:.4f}) '
                'surround_range=({:.4f}, {:.4f}) clamp_ratio={:.4f} '
                'abs_mean(target/surround/diff)=({:.4f}/{:.4f}/{:.4f})'
                .format(
                    tuple(query.shape),
                    tuple(reference_points.shape),
                    tuple(target_locations.shape),
                    tuple(surround_locations.shape),
                    tuple(z_target.shape),
                    tuple(z_surround.shape),
                    tuple(z_diff.shape), gate_stats, tuple(output.shape),
                    self.last_debug_info['target_location_min'],
                    self.last_debug_info['target_location_max'],
                    self.last_debug_info['surround_location_min'],
                    self.last_debug_info['surround_location_max'],
                    self.last_debug_info['surround_clamp_ratio'],
                    self.last_debug_info['z_target_abs_mean'],
                    self.last_debug_info['z_surround_abs_mean'],
                    self.last_debug_info['z_diff_abs_mean']))
        self._debug_recorded = True

    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_mask=None):
        """Apply paired target-surround deformable cross-attention."""
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value = value * value_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(
            bs, len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, len_q, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).reshape(
            bs, len_q, self.num_heads,
            self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(
            bs, len_q, self.num_heads, self.num_levels, self.num_points)

        # Target locations are identical to the original implementation.
        target_locations = self._get_target_locations(
            reference_points, sampling_offsets, value_spatial_shapes)

        # Pair every target sampling point with a radially expanded point in
        # the same head, feature level, sampling-point index, and direction.
        centers = self._get_reference_centers(reference_points)
        raw_surround_locations = centers + self.surround_scale * (
            target_locations - centers)
        if self.debug and not self._debug_recorded:
            outside = ((raw_surround_locations < 0.) |
                       (raw_surround_locations > 1.)).any(dim=-1)
            clamp_ratio = outside.to(torch.float32).mean()
        else:
            # Avoid the diagnostic reduction in normal training.
            clamp_ratio = raw_surround_locations.new_zeros(())
        surround_locations = raw_surround_locations.clamp(0., 1.)

        z_target = self.ms_deformable_attn_core(
            value, value_spatial_shapes, target_locations, attention_weights)
        z_surround = self.ms_deformable_attn_core(
            value, value_spatial_shapes, surround_locations,
            attention_weights)

        z_diff = z_target - z_surround
        if self.differential_enabled:
            z_evidence = z_diff
        else:
            # Explicit Target + Surround ablation.
            z_evidence = z_surround

        if self.diff_gate is not None:
            gate = torch.sigmoid(self.diff_gate(query))
            fused = z_target + gate * z_evidence
        else:
            gate = None
            fused = z_target + z_evidence

        # Preserve exactly one output projection after fusion.
        output = self.output_proj(fused)
        self._record_debug_info(
            query, reference_points, target_locations, surround_locations,
            z_target, z_surround, z_diff, z_evidence, gate, output,
            clamp_ratio)
        return output


def build_decoder_cross_attention(base_attention, config=None):
    """Build the configured Decoder Cross-Attention implementation.

    ``None`` and ``enabled: false`` return the exact original attention object,
    keeping the baseline parameter keys, operations, and speed unchanged.
    """
    if config is None:
        return base_attention
    if not isinstance(config, dict):
        raise TypeError(
            'decoder_cross_attention must be a dict or None, but got {}.'
            .format(type(config).__name__))

    config = copy.deepcopy(config)
    enabled = bool(config.pop('enabled', True))
    attention_type = str(
        config.pop('type', 'paired_target_surround')).lower()
    baseline_names = {
        'ms_deformable_attention', 'msdeformableattention', 'baseline',
        'original'
    }
    if not enabled or attention_type in baseline_names:
        return base_attention

    paired_names = {
        'paired_target_surround',
        'paired_target_surround_deformable_attention',
        'pairedtargetsurrounddeformableattention',
    }
    if attention_type not in paired_names:
        raise ValueError(
            'Unsupported Decoder Cross-Attention type: {}.'.format(
                attention_type))

    surround = config.pop('surround', {})
    differential = config.pop('differential', {})
    gated_fusion = config.pop('gated_fusion', {})
    debug = bool(config.pop('debug', False))
    if config:
        raise ValueError(
            'Unknown decoder_cross_attention options: {}.'.format(
                ', '.join(sorted(config.keys()))))
    for name, section in (
            ('surround', surround), ('differential', differential),
            ('gated_fusion', gated_fusion)):
        if not isinstance(section, dict):
            raise TypeError('{} must be a dict.'.format(name))

    surround = copy.deepcopy(surround)
    differential = copy.deepcopy(differential)
    gated_fusion = copy.deepcopy(gated_fusion)
    surround_enabled = bool(surround.pop('enabled', True))
    surround_scale = float(surround.pop('scale', 1.8))
    differential_enabled = bool(differential.pop('enabled', True))
    gated_fusion_enabled = bool(gated_fusion.pop('enabled', True))
    gate_init_bias = float(gated_fusion.pop('init_bias', -2.0))
    unknown_nested = {
        name: sorted(section.keys())
        for name, section in (
            ('surround', surround), ('differential', differential),
            ('gated_fusion', gated_fusion)) if section
    }
    if unknown_nested:
        raise ValueError(
            'Unknown Decoder Cross-Attention nested options: {}.'.format(
                unknown_nested))

    # Target-only is the original module, avoiding unused extra parameters and
    # preserving exact baseline speed/state when surround is disabled.
    if not surround_enabled:
        return base_attention

    return PairedTargetSurroundDeformableAttention(
        base_attention=base_attention,
        surround_scale=surround_scale,
        differential_enabled=differential_enabled,
        gated_fusion_enabled=gated_fusion_enabled,
        gate_init_bias=gate_init_bias,
        debug=debug)
