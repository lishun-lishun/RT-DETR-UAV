"""Scale-Aware Query Selection for RT-DETR.

The original RT-DETR query initializer lets every flattened feature level
compete in one global Top-K.  SAQS optionally assigns a deterministic integer
quota to each resolution level, performs Top-K inside each level, and returns
one concatenated set of *global* flattened indices.  The caller must use this
single index tensor to gather encoder features, class logits, and proposal
boxes together.

No ground truth is consumed here.  The optional small-area prior is computed
only from the encoder proposal itself (sigmoid-normalized predicted width and
height), so the exact same selector is valid during training and inference.
"""

import math

import torch
import torch.distributed as dist
import torch.nn as nn


__all__ = ['ScaleAwareQuerySelector', 'SAQS']


def _positive_int(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _finite_float(name, value):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value))):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _positive_int_sequence(name, values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f'{name} must be a nonempty list or tuple')
    result = tuple(_positive_int(f'{name}[{index}]', value)
                   for index, value in enumerate(values))
    if len(set(result)) != len(result):
        raise ValueError(f'{name} values must be unique')
    return result


def _ratio_sequence(values, expected_length):
    if not isinstance(values, (list, tuple)):
        raise ValueError('SAQS.level_ratios must be a list or tuple')
    if len(values) != expected_length:
        raise ValueError(
            'SAQS.level_ratios length must equal the number of feature levels')
    ratios = tuple(_finite_float(
        f'SAQS.level_ratios[{index}]', value)
        for index, value in enumerate(values))
    if any(value < 0.0 for value in ratios):
        raise ValueError('SAQS.level_ratios must be nonnegative')
    if sum(ratios) <= 0.0:
        raise ValueError('SAQS.level_ratios must contain a positive value')
    return ratios


class ScaleAwareQuerySelector(nn.Module):
    """Select a fixed number of queries with per-resolution quotas.

    ``level_ratios`` are specified in high-to-low resolution order.  The
    mapping is derived from ``feature_level_strides`` (ascending stride), not
    from a backbone name or an unchecked assumption about list position.
    Integer quotas use largest-remainder allocation.  If a dynamic input has
    too few candidates at a level, its overflow is deterministically
    redistributed to levels with spare capacity while preserving the exact
    total ``num_queries`` whenever enough valid proposals exist globally.
    """

    def __init__(
            self,
            num_queries=300,
            feature_level_strides=(8, 16, 32),
            level_ratios=(0.6, 0.3, 0.1),
            use_area_prior=False,
            area_prior_lambda=0.25,
            area_prior_tau=0.02,
            area_weight_max=1.5,
            debug=False):
        super().__init__()
        self.num_queries = _positive_int('SAQS.num_queries', num_queries)
        self.feature_level_strides = _positive_int_sequence(
            'SAQS.feature_level_strides', feature_level_strides)
        self.level_ratios = _ratio_sequence(
            level_ratios, len(self.feature_level_strides))

        if not isinstance(use_area_prior, bool):
            raise ValueError('SAQS.use_area_prior must be a boolean')
        self.use_area_prior = use_area_prior
        self.area_prior_lambda = _finite_float(
            'SAQS.area_prior_lambda', area_prior_lambda)
        self.area_prior_tau = _finite_float(
            'SAQS.area_prior_tau', area_prior_tau)
        self.area_weight_max = _finite_float(
            'SAQS.area_weight_max', area_weight_max)
        if self.area_prior_lambda < 0.0:
            raise ValueError('SAQS.area_prior_lambda must be nonnegative')
        if self.area_prior_tau <= 0.0:
            raise ValueError('SAQS.area_prior_tau must be positive')
        if self.area_weight_max < 1.0:
            raise ValueError('SAQS.area_weight_max must be at least 1')
        if not isinstance(debug, bool):
            raise ValueError('SAQS.debug must be a boolean')
        self.debug = debug

        # Rank feature-list positions by actual stride: rank 0 is the highest
        # resolution. Ratios are interpreted in this ranked order.
        self.level_order = tuple(sorted(
            range(len(self.feature_level_strides)),
            key=lambda index: self.feature_level_strides[index]))
        self._debug_printed = False
        self.last_selected_queries = None

    @staticmethod
    def _largest_remainder(total, ratios):
        ratio_sum = sum(ratios)
        raw = [total * ratio / ratio_sum for ratio in ratios]
        allocation = [int(math.floor(value)) for value in raw]
        remainder = total - sum(allocation)
        order = sorted(
            range(len(ratios)),
            key=lambda index: (-(raw[index] - allocation[index]), index))
        for index in order[:remainder]:
            allocation[index] += 1
        return allocation

    @classmethod
    def allocate_queries(cls, num_queries, ratios, capacities):
        """Allocate exact integer quotas, redistributing capacity overflow."""
        if len(ratios) != len(capacities):
            raise ValueError('SAQS ratios and capacities must have equal length')
        capacities = [int(value) for value in capacities]
        if any(value < 0 for value in capacities):
            raise ValueError('SAQS capacities must be nonnegative')
        if sum(capacities) < num_queries:
            raise RuntimeError(
                f'SAQS has only {sum(capacities)} valid proposals for '
                f'{num_queries} queries')

        allocation = cls._largest_remainder(num_queries, ratios)
        allocation = [min(value, capacity)
                      for value, capacity in zip(allocation, capacities)]
        missing = num_queries - sum(allocation)

        # Capacity limitations are rare at formal 480--800 inputs but can
        # occur in compact smoke tests. D'Hondt-style priorities retain the
        # requested proportions while filling only levels with spare slots.
        while missing:
            available = [index for index, capacity in enumerate(capacities)
                         if allocation[index] < capacity]
            if not available:
                raise RuntimeError(
                    'SAQS could not redistribute all capacity overflow')
            index = max(
                available,
                key=lambda value: (
                    ratios[value] / (allocation[value] + 1), -value))
            allocation[index] += 1
            missing -= 1

        if sum(allocation) != num_queries:
            raise RuntimeError('SAQS internal quota sum is inconsistent')
        if any(value > capacity for value, capacity
               in zip(allocation, capacities)):
            raise RuntimeError('SAQS internal quota exceeds level capacity')
        return allocation

    def _validate_inputs(
            self, class_logits, proposal_logits, spatial_shapes, valid_mask):
        if not torch.is_tensor(class_logits) or class_logits.ndim != 3:
            raise RuntimeError(
                'SAQS class_logits must be a [B, L, classes] tensor')
        if not class_logits.is_floating_point() or class_logits.is_complex():
            raise RuntimeError('SAQS class_logits must be real floating point')
        if class_logits.shape[-1] < 1:
            raise RuntimeError('SAQS class_logits needs at least one class')
        if not torch.is_tensor(proposal_logits) or proposal_logits.ndim != 3:
            raise RuntimeError(
                'SAQS proposal_logits must be a [B, L, 4] tensor')
        if proposal_logits.shape[:2] != class_logits.shape[:2] \
                or proposal_logits.shape[-1] != 4:
            raise RuntimeError(
                'SAQS proposal_logits must align with class_logits and end in 4')
        if not proposal_logits.is_floating_point() \
                or proposal_logits.is_complex():
            raise RuntimeError(
                'SAQS proposal_logits must be real floating point')
        if proposal_logits.dtype != class_logits.dtype:
            raise RuntimeError(
                'SAQS class and proposal tensors must share a dtype')
        if proposal_logits.device != class_logits.device:
            raise RuntimeError(
                'SAQS class and proposal tensors must share a device')

        if not isinstance(spatial_shapes, (list, tuple)) \
                or len(spatial_shapes) != len(self.feature_level_strides):
            raise RuntimeError(
                'SAQS spatial_shapes must match feature_level_strides')
        shapes = []
        for level, shape in enumerate(spatial_shapes):
            if not isinstance(shape, (list, tuple)) or len(shape) != 2:
                raise RuntimeError(
                    f'SAQS spatial_shapes[{level}] must contain [H, W]')
            height = _positive_int(
                f'SAQS.spatial_shapes[{level}][0]', shape[0])
            width = _positive_int(
                f'SAQS.spatial_shapes[{level}][1]', shape[1])
            shapes.append((height, width))
        lengths = [height * width for height, width in shapes]
        if sum(lengths) != class_logits.shape[1]:
            raise RuntimeError(
                f'SAQS flattened length {class_logits.shape[1]} does not '
                f'equal spatial level total {sum(lengths)}')

        batch, length = class_logits.shape[:2]
        if batch < 1:
            raise RuntimeError('SAQS requires a nonempty batch')
        if valid_mask is None:
            valid = torch.ones(
                batch, length, dtype=torch.bool, device=class_logits.device)
        else:
            if not torch.is_tensor(valid_mask):
                raise RuntimeError('SAQS valid_mask must be a tensor')
            if valid_mask.dtype != torch.bool:
                raise RuntimeError('SAQS valid_mask must use torch.bool')
            if valid_mask.ndim == 3 and valid_mask.shape[-1] == 1:
                valid_mask = valid_mask.squeeze(-1)
            if valid_mask.ndim != 2 or valid_mask.shape[1] != length \
                    or valid_mask.shape[0] not in (1, batch):
                raise RuntimeError(
                    'SAQS valid_mask must have shape [1|B, L, 1] or [1|B, L]')
            if valid_mask.device != class_logits.device:
                raise RuntimeError(
                    'SAQS valid_mask must share the logits device')
            valid = valid_mask.to(dtype=torch.bool).expand(batch, -1)
        return lengths, valid

    def _selection_scores(self, class_logits, proposal_logits):
        scores = class_logits.max(dim=-1).values
        if not self.use_area_prior:
            return scores

        # The requested multiplicative weight must act on a positive score.
        # Using sigmoid probability avoids reversing the intended preference
        # when the raw classification logit is negative. Only ranking consumes
        # this tensor; gathered encoder logits remain the original raw logits.
        probability = scores.detach().float().sigmoid()
        proposal = proposal_logits.detach().float().sigmoid()
        area = proposal[..., 2] * proposal[..., 3]
        weight = 1.0 + self.area_prior_lambda * torch.exp(
            -area / self.area_prior_tau)
        weight = weight.clamp(min=1.0, max=self.area_weight_max)
        return probability * weight

    def _debug_once(self, allocations_by_rank):
        if not self.debug or self._debug_printed:
            return
        self._debug_printed = True
        rank = dist.get_rank() if dist.is_available() \
            and dist.is_initialized() else 0
        if rank != 0:
            return
        ordered_strides = [self.feature_level_strides[index]
                           for index in self.level_order]
        print('SAQS enabled')
        print(f'SAQS num_queries = {self.num_queries}')
        print(f'SAQS feature levels (flatten positions) = '
              f'{list(self.level_order)}')
        print(f'SAQS feature level strides = {ordered_strides}')
        print(f'SAQS level ratios = {list(self.level_ratios)}')
        print(f'SAQS selected queries = {list(allocations_by_rank)}')

    def forward(
            self, class_logits, proposal_logits, spatial_shapes,
            valid_mask=None):
        lengths, valid = self._validate_inputs(
            class_logits, proposal_logits, spatial_shapes, valid_mask)
        selection_scores = self._selection_scores(
            class_logits, proposal_logits)
        selection_scores = selection_scores.masked_fill(
            ~valid, float('-inf'))

        starts = []
        offset = 0
        for length in lengths:
            starts.append(offset)
            offset += length

        # Capacities use the minimum across the batch so every image can use
        # the same per-level K and torch.topk always returns a dense [B, K].
        capacities_by_position = []
        for start, length in zip(starts, lengths):
            counts = valid[:, start:start + length].sum(dim=1)
            capacities_by_position.append(int(counts.min().item()))
        capacities_by_rank = [capacities_by_position[index]
                              for index in self.level_order]
        allocations_by_rank = self.allocate_queries(
            self.num_queries, self.level_ratios, capacities_by_rank)

        global_indices = []
        for rank, level_index in enumerate(self.level_order):
            count = allocations_by_rank[rank]
            if count == 0:
                continue
            start = starts[level_index]
            level_scores = selection_scores[
                :, start:start + lengths[level_index]]
            local_indices = torch.topk(
                level_scores, count, dim=1).indices
            global_indices.append(local_indices + start)
        if not global_indices:
            raise RuntimeError('SAQS selected no query')
        topk_indices = torch.cat(global_indices, dim=1)
        if topk_indices.shape != (
                class_logits.shape[0], self.num_queries):
            raise RuntimeError(
                f'SAQS selected shape {tuple(topk_indices.shape)}, expected '
                f'({class_logits.shape[0]}, {self.num_queries})')
        if not valid.gather(1, topk_indices).all():
            raise RuntimeError('SAQS selected an invalid anchor')

        self.last_selected_queries = tuple(allocations_by_rank)
        self._debug_once(allocations_by_rank)
        return topk_indices


SAQS = ScaleAwareQuerySelector
