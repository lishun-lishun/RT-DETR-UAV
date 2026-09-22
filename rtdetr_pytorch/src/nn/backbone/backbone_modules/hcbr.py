"""Hollow-context background refinement for PResNet feature maps.

The module has the same input and output shape.  Its residual scale is zero at
initialization so inserting it into a pretrained backbone initially preserves
the backbone's predictions exactly.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class HCBR(nn.Module):
    """Suppress locally predictable context while retaining anomalous detail.

    ``omega`` selects between two hollow neighborhoods independently at every
    spatial position.  The hollow averages exclude the central ``inner_kernel``
    square, including at image boundaries (replicate padding).
    """

    def __init__(self, channels, inner_kernel=3, near_kernel=7,
                 far_kernel=11, gamma=2.0, eps=1e-6, lambda_init=0.0,
                 lambda_max=0.2, debug=False, debug_interval=100):
        super().__init__()
        if not isinstance(channels, int) or isinstance(channels, bool) or channels <= 0:
            raise ValueError('HCBR channels must be a positive integer')
        for name, kernel in (('inner_kernel', inner_kernel),
                             ('near_kernel', near_kernel),
                             ('far_kernel', far_kernel)):
            if not isinstance(kernel, int) or isinstance(kernel, bool) or kernel <= 0 or kernel % 2 != 1:
                raise ValueError(f'HCBR {name} must be a positive odd integer')
        if not inner_kernel < near_kernel < far_kernel:
            raise ValueError('HCBR requires inner_kernel < near_kernel < far_kernel')
        if not (math.isfinite(gamma) and gamma > 0 and math.isfinite(eps) and eps > 0):
            raise ValueError('HCBR gamma and eps must be finite and positive')
        if not (math.isfinite(lambda_init) and math.isfinite(lambda_max) and lambda_max > 0):
            raise ValueError('HCBR lambda parameters must be finite and lambda_max positive')
        if not isinstance(debug, bool) or not isinstance(debug_interval, int) or debug_interval < 1:
            raise ValueError('HCBR debug must be boolean and debug_interval positive')

        self.channels = channels
        self.inner_kernel = inner_kernel
        self.near_kernel = near_kernel
        self.far_kernel = far_kernel
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.lambda_max = float(lambda_max)
        self.scale_router = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.raw_lambda = nn.Parameter(torch.tensor(float(lambda_init)))
        self.debug = debug
        self.debug_interval = debug_interval
        self._debug_iteration = 0

    @property
    def lambda_effective(self):
        return self.lambda_max * self.raw_lambda.tanh()

    @staticmethod
    def _average(x, kernel):
        radius = kernel // 2
        # Replication is defined even if a feature map is smaller than the
        # kernel, whereas reflection padding has a minimum-size requirement.
        padded = F.pad(x, (radius, radius, radius, radius), mode='replicate')
        return F.avg_pool2d(padded, kernel_size=kernel, stride=1, padding=0)

    def _fields(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels or x.shape[-2] < 1 or x.shape[-1] < 1:
            raise ValueError('HCBR expects a nonempty NCHW tensor with the configured channels')

        x32 = x.float()
        area_inner = self.inner_kernel ** 2
        area_near = self.near_kernel ** 2
        area_far = self.far_kernel ** 2
        a_inner = self._average(x32, self.inner_kernel)
        a_near = self._average(x32, self.near_kernel)
        a_far = self._average(x32, self.far_kernel)
        b_near = (area_near * a_near - area_inner * a_inner) / (area_near - area_inner)
        b_far = (area_far * a_far - area_inner * a_inner) / (area_far - area_inner)

        omega = self.scale_router(x).sigmoid().float()
        background = omega * b_near + (1.0 - omega) * b_far
        residual = x32 - background

        # Cosine is calculated channelwise in FP32.  A zero vector uses the
        # epsilon-limited denominator; its residual is also zero when both
        # input and background are zero.
        x_norm = x32.square().sum(dim=1, keepdim=True).sqrt()
        b_norm = background.square().sum(dim=1, keepdim=True).sqrt()
        normalized_x = x32 / (x_norm + self.eps)
        normalized_b = background / (b_norm + self.eps)
        cosine = (normalized_x * normalized_b).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        gate = 1.0 - cosine.pow(self.gamma)
        return {'omega': omega, 'background7': b_near,
                'background11': b_far, 'background': background,
                'residual': residual, 'similarity': cosine, 'gate': gate}

    def compute_components(self, x):
        """Return differentiable fields for tests and feature analysis.

        ``background7`` and ``background11`` refer to the default outer
        kernels; with custom kernels they retain these names for API stability.
        """
        return self._fields(x)

    def diagnostics(self, x):
        """Return detached feature fields for explicit analysis, not training.

        Callers can inspect ``omega``, ``background``, ``residual``, ``similarity``
        and ``gate`` without enabling per-step debug printing.
        """
        with torch.no_grad():
            fields = self._fields(x)
            fields['lambda_effective'] = self.lambda_effective
            return {name: value.detach() for name, value in fields.items()}

    def _print_debug(self, fields, correction):
        with torch.no_grad():
            omega = fields['omega'].float().flatten()
            similarity = fields['similarity'].float().flatten()
            gate = fields['gate'].float().flatten()
            residual_norm = fields['residual'].norm()
            input_norm = (fields['background'] + fields['residual']).norm()
            values = {
                'lambda_effective': self.lambda_effective,
                'omega_mean': omega.mean(),
                'omega_p10': torch.quantile(omega, 0.1),
                'omega_p50': torch.quantile(omega, 0.5),
                'omega_p90': torch.quantile(omega, 0.9),
                'background_similarity_mean': similarity.mean(),
                'background_similarity_p10': torch.quantile(similarity, 0.1),
                'background_similarity_p90': torch.quantile(similarity, 0.9),
                'gate_mean': gate.mean(),
                'gate_p10': torch.quantile(gate, 0.1),
                'gate_p50': torch.quantile(gate, 0.5),
                'gate_p90': torch.quantile(gate, 0.9),
                'input_norm': input_norm,
                'residual_norm': residual_norm,
                'residual/input_ratio': residual_norm / (input_norm + self.eps),
            }
            message = ' '.join(f'{name}={float(value.detach()):.6g}'
                               for name, value in values.items())
            location = getattr(self, 'debug_name', 'feature')
            print(f'[HCBR-{location} iter={self._debug_iteration}] {message}')

    def forward(self, x):
        fields = self._fields(x)
        correction = (self.lambda_effective.float() * fields['gate'] * fields['residual']).to(dtype=x.dtype)
        if self.debug and self._debug_iteration % self.debug_interval == 0:
            self._print_debug(fields, correction)
        self._debug_iteration += 1
        return x + correction
