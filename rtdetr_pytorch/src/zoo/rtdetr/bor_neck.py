"""Background-Orthogonal Residual (BOR) neck for RT-DETR.

BOR is deliberately a small post-CCFF refinement.  It receives the original
N3 tensor, estimates a local ring-background prototype at every position,
keeps the component orthogonal to that prototype, and injects a gated version
through a bounded residual.  N4/N5 are outside this module's interface.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['BackgroundOrthogonalResidual', 'BORNeck']


def _positive_int(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _finite_float(name, value):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value))):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _positive_float(name, value):
    value = _finite_float(name, value)
    if value <= 0.0:
        raise ValueError(f'{name} must be positive')
    return value


class BackgroundOrthogonalResidual(nn.Module):
    """Enhance N3 evidence that is novel relative to its ring background.

    Args:
        hidden_dim: Channel count of the HybridEncoder N3 feature.
        outer_kernel: Odd outer window size of the ring (7 in round one).
        inner_kernel: Odd excluded inner window size (3 in round one).
        theta: Fixed novelty threshold.  It is intentionally not learnable.
        tau: Fixed gate temperature.  It is intentionally not learnable.
        alpha_max: Upper bound of the learned residual scale.
        alpha_init: Effective residual scale at initialization.
        eps: Denominator stabilizer for decomposition and novelty ratio.
        debug: Record detached summary tensors in ``last_debug_stats``.

    The ring uses zero-padded, stride-one average pooling, so both pooled
    tensors retain the dynamic input shape.  All reductions, divisions, norms
    and sigmoid gate statistics are evaluated in FP32 under mixed precision;
    only the residual entering the 1x1 projection is cast back.
    """

    def __init__(self, hidden_dim, outer_kernel=7, inner_kernel=3,
                 theta=0.25, tau=0.10, alpha_max=0.30,
                 alpha_init=0.05, eps=1e-6, debug=False):
        super().__init__()
        hidden_dim = _positive_int('BOR.hidden_dim', hidden_dim)
        outer_kernel = _positive_int('BOR.outer_kernel', outer_kernel)
        inner_kernel = _positive_int('BOR.inner_kernel', inner_kernel)
        if outer_kernel % 2 == 0 or inner_kernel % 2 == 0:
            raise ValueError('BOR kernels must both be odd for same-size pooling')
        if outer_kernel <= inner_kernel:
            raise ValueError('BOR.outer_kernel must be larger than inner_kernel')

        theta = _finite_float('BOR.theta', theta)
        tau = _positive_float('BOR.tau', tau)
        alpha_max = _positive_float('BOR.alpha_max', alpha_max)
        alpha_init = _positive_float('BOR.alpha_init', alpha_init)
        eps = _positive_float('BOR.eps', eps)
        if alpha_init >= alpha_max:
            raise ValueError('BOR.alpha_init must be smaller than alpha_max')
        if not isinstance(debug, bool):
            raise ValueError('BOR.debug must be a boolean')

        self.hidden_dim = hidden_dim
        self.outer_kernel = outer_kernel
        self.inner_kernel = inner_kernel
        self.theta = theta
        self.tau = tau
        self.alpha_max = alpha_max
        self.eps = eps
        self.debug = debug

        # No normalization is used: BOR adds only one lightweight projection.
        self.output_projection = nn.Conv2d(hidden_dim, hidden_dim, 1)

        probability = alpha_init / alpha_max
        raw_init = math.log(probability / (1.0 - probability))
        self.raw_alpha = nn.Parameter(torch.tensor(raw_init))

        # Populated only for an explicitly enabled debug path.  Keeping scalar
        # tensors detached avoids retaining an autograd graph or printing/syncing
        # every training iteration.
        self.last_debug_stats = None

    def effective_alpha(self):
        """Return the bounded, differentiable effective residual scale."""
        return self.alpha_max * torch.sigmoid(self.raw_alpha)

    def ring_prototype(self, features):
        """Compute the zero-padded local ring mean in FP32.

        For the prescribed 7/3 kernels this is exactly
        ``(49 * AvgPool7(F) - 9 * AvgPool3(F)) / 40``.
        """
        features_fp32 = features.float()
        outer_area = self.outer_kernel ** 2
        inner_area = self.inner_kernel ** 2
        outer_sum = outer_area * F.avg_pool2d(
            features_fp32, kernel_size=self.outer_kernel, stride=1,
            padding=self.outer_kernel // 2, count_include_pad=True)
        inner_sum = inner_area * F.avg_pool2d(
            features_fp32, kernel_size=self.inner_kernel, stride=1,
            padding=self.inner_kernel // 2, count_include_pad=True)
        return (outer_sum - inner_sum) / float(outer_area - inner_area)

    def orthogonal_decompose(self, features, background):
        """Return background-parallel and background-orthogonal components."""
        if features.shape != background.shape:
            raise RuntimeError(
                'BOR features and background prototype must have identical shapes')
        features_fp32 = features.float()
        background_fp32 = background.float()
        coefficient = (
            (features_fp32 * background_fp32).sum(dim=1, keepdim=True)
            / (background_fp32.square().sum(dim=1, keepdim=True) + self.eps)
        )
        parallel = coefficient * background_fp32
        orthogonal = features_fp32 - parallel
        return parallel, orthogonal

    def _record_debug_stats(self, novelty, gate, orthogonal, features, alpha):
        self.last_debug_stats = {
            'novelty_ratio_mean': novelty.mean().detach(),
            'novelty_ratio_std': novelty.std(unbiased=False).detach(),
            'gate_mean': gate.mean().detach(),
            'gate_std': gate.std(unbiased=False).detach(),
            'alpha_eff': alpha.detach(),
            'orthogonal_residual_norm': torch.linalg.vector_norm(
                orthogonal, ord=2, dim=1).mean().detach(),
            'base_n3_norm': torch.linalg.vector_norm(
                features.float(), ord=2, dim=1).mean().detach(),
        }

    def forward(self, n3, return_aux=False):
        if not torch.is_tensor(n3) or n3.ndim != 4:
            raise RuntimeError('BOR N3 must be a BCHW tensor')
        if n3.shape[1] != self.hidden_dim:
            raise RuntimeError(
                f'BOR N3 has {n3.shape[1]} channels, expected {self.hidden_dim}')
        if not (n3.is_floating_point() or n3.is_complex()):
            raise RuntimeError('BOR N3 must use a floating-point dtype')
        if n3.is_complex():
            raise RuntimeError('BOR N3 must use a real floating-point dtype')

        # Critical local statistics remain FP32 even when the surrounding
        # HybridEncoder runs under CUDA autocast.
        features_fp32 = n3.float()
        background = self.ring_prototype(features_fp32)
        parallel, orthogonal = self.orthogonal_decompose(
            features_fp32, background)
        orthogonal_norm = torch.linalg.vector_norm(
            orthogonal, ord=2, dim=1, keepdim=True)
        feature_norm = torch.linalg.vector_norm(
            features_fp32, ord=2, dim=1, keepdim=True)
        novelty = orthogonal_norm / (feature_norm + self.eps)
        gate = torch.sigmoid((novelty - self.theta) / self.tau)
        raw_residual = gate * orthogonal

        projected = self.output_projection(raw_residual.to(dtype=n3.dtype))
        if projected.dtype != n3.dtype:
            projected = projected.to(dtype=n3.dtype)
        alpha = self.effective_alpha()
        enhanced = n3 + alpha.to(dtype=n3.dtype) * projected

        if self.debug:
            self._record_debug_stats(
                novelty, gate, orthogonal, features_fp32, alpha)
        else:
            self.last_debug_stats = None

        if not return_aux:
            return enhanced
        return enhanced, {
            'background_prototype': background,
            'parallel_component': parallel,
            'orthogonal_component': orthogonal,
            'novelty_ratio': novelty,
            'gate': gate,
            'raw_residual': raw_residual,
            'projected_residual': projected,
            'alpha_eff': alpha,
        }


# Short alias for callers that prefer the configuration name.
BORNeck = BackgroundOrthogonalResidual
