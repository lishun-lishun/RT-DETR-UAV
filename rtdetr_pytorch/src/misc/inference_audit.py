"""Optional one-forward component counters and actual activation dtype audit."""

import torch
import torch.nn as nn

from .dist import de_parallel


def _dtypes(value):
    if torch.is_tensor(value):
        return str(value.dtype)
    if isinstance(value, dict):
        return {key: _dtypes(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_dtypes(item) for item in value]
    return str(type(value).__name__)


class InferenceAudit(object):
    """Only explicit debug calls install hooks; close() removes every hook."""
    def __init__(self, model):
        self.model = de_parallel(model)
        self.counts = {'model': 0, 'backbone': 0, 'encoder': 0, 'decoder': 0}
        self.values = {}
        self.handles = []

    def __enter__(self):
        for name in self.counts:
            module = self.model if name == 'model' else getattr(self.model, name)
            self.handles.append(module.register_forward_hook(self._component_hook(name)))
        first_conv = next((module for module in self.model.modules() if isinstance(module, nn.Conv2d)), None)
        if first_conv is not None:
            self.handles.append(first_conv.register_forward_hook(self._dtype_hook('first Conv output dtype')))
        # MultiheadAttention.out_proj is an nn.Linear but its forward() is
        # bypassed by functional attention. Capture the first ACTUALLY called
        # Linear/Attention instead of the first registered (possibly unused) one.
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(self._dtype_hook('first Linear output dtype')))
            elif isinstance(module, nn.MultiheadAttention):
                self.handles.append(module.register_forward_hook(self._dtype_hook('first Attention output dtype')))
        return self

    def _dtype_hook(self, name):
        def hook(_module, _inputs, output):
            if name not in self.values:
                self.values[name] = _dtypes(output)
                self.values[name.replace('output dtype', 'CUDA autocast active')] = bool(
                    torch.is_autocast_enabled()
                )
        return hook

    def _component_hook(self, name):
        def hook(_module, inputs, output):
            self.counts[name] += 1
            if name == 'model':
                self.values['input dtype'] = _dtypes(inputs[0])
                self.values['CUDA autocast active'] = bool(torch.is_autocast_enabled())
            elif name == 'backbone':
                return_indices = getattr(self.model.backbone, 'return_idx', range(1, len(output) + 1))
                for stage_index, feature in zip(return_indices, output):
                    self.values['backbone S{} dtype'.format(stage_index + 2)] = _dtypes(feature)
            elif name == 'encoder':
                self.values['encoder output dtype'] = _dtypes(output)
            elif name == 'decoder':
                self.values['decoder logits dtype'] = _dtypes(output['pred_logits'])
                self.values['decoder boxes dtype'] = _dtypes(output['pred_boxes'])
        return hook

    def __exit__(self, *_args):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def report(self, expected_amp=False):
        print('Evaluation first-batch dtype audit:', self.values)
        print('Evaluation forward counts:', self.counts)
        if any(count != 1 for count in self.counts.values()):
            raise RuntimeError('Evaluation must call model/backbone/encoder/decoder exactly once')
        if expected_amp and not self.values.get('CUDA autocast active', False):
            raise RuntimeError('Evaluation requested AMP but actual CUDA autocast was not active')
        if expected_amp and self.values.get('first Conv output dtype') != str(torch.float16):
            raise RuntimeError('Evaluation AMP audit did not observe FP16 convolution output')
