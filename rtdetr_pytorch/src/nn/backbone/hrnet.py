"""Standard HRNetV2-W18 backbone adapter for RT-DETR.

The architecture and module naming are adapted from Microsoft's official
HRNet ImageNet classification implementation and the matching timm port:
https://github.com/HRNet/HRNet-Image-Classification
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/hrnet.py

Portions Copyright (c) Microsoft Corporation, Bin Xiao and Ke Sun.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Only the backbone is retained: no timm dependency and no classification head.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.core import register


__all__ = ['HRNetV2W18', 'HighResolutionModule']

_BN_MOMENTUM = 0.1
_PRETRAINED_URL = (
    'https://github.com/rwightman/pytorch-image-models/releases/'
    'download/v0.1-hrnet/hrnetv2_w18-8cb57bb9.pth')
_EXPECTED_FILENAME = 'hrnetv2_w18-8cb57bb9.pth'
_CLASSIFICATION_PREFIXES = (
    'incre_modules.', 'downsamp_modules.', 'final_layer.',
    'global_pool.', 'classifier.', 'head.',
)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, momentum=_BN_MOMENTUM)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, momentum=_BN_MOMENTUM)
        self.act2 = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act2(x + residual)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, momentum=_BN_MOMENTUM)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, momentum=_BN_MOMENTUM)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, momentum=_BN_MOMENTUM)
        self.act3 = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.act3(x + residual)


_BLOCKS = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}


class HighResolutionModule(nn.Module):
    """Parallel residual branches followed by all-to-all resolution fusion."""

    def __init__(self, num_branches, block, num_blocks, num_in_channels,
                 num_channels, multi_scale_output=True):
        super().__init__()
        sizes = (len(num_blocks), len(num_in_channels), len(num_channels))
        if any(size != num_branches for size in sizes):
            raise ValueError(
                'HRNet branch config mismatch: '
                f'branches={num_branches}, blocks/in/channels={sizes}')
        self.num_branches = num_branches
        self.num_in_channels = list(num_in_channels)
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(block, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.fuse_act = nn.ReLU(inplace=False)

    def _make_one_branch(self, index, block, blocks, channels):
        output_channels = channels[index] * block.expansion
        downsample = None
        if self.num_in_channels[index] != output_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_in_channels[index], output_channels, 1,
                          bias=False),
                nn.BatchNorm2d(output_channels, momentum=_BN_MOMENTUM),
            )
        layers = [block(self.num_in_channels[index], channels[index],
                        downsample=downsample)]
        self.num_in_channels[index] = output_channels
        for _ in range(1, blocks[index]):
            layers.append(block(output_channels, channels[index]))
        return nn.Sequential(*layers)

    def _make_branches(self, block, num_blocks, num_channels):
        return nn.ModuleList([
            self._make_one_branch(index, block, num_blocks, num_channels)
            for index in range(self.num_branches)
        ])

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return nn.Identity()
        output_branches = self.num_branches if self.multi_scale_output else 1
        fuse_layers = []
        for target in range(output_branches):
            transforms = []
            for source in range(self.num_branches):
                if source == target:
                    transforms.append(nn.Identity())
                elif source > target:
                    transforms.append(nn.Sequential(
                        nn.Conv2d(self.num_in_channels[source],
                                  self.num_in_channels[target], 1, bias=False),
                        nn.BatchNorm2d(self.num_in_channels[target],
                                       momentum=_BN_MOMENTUM),
                    ))
                else:
                    steps = []
                    in_channels = self.num_in_channels[source]
                    for step in range(target - source):
                        is_last = step == target - source - 1
                        out_channels = (self.num_in_channels[target]
                                        if is_last else in_channels)
                        layers = [
                            nn.Conv2d(in_channels, out_channels, 3, stride=2,
                                      padding=1, bias=False),
                            nn.BatchNorm2d(out_channels, momentum=_BN_MOMENTUM),
                        ]
                        if not is_last:
                            layers.append(nn.ReLU(inplace=False))
                        steps.append(nn.Sequential(*layers))
                        in_channels = out_channels
                    transforms.append(nn.Sequential(*steps))
            fuse_layers.append(nn.ModuleList(transforms))
        return nn.ModuleList(fuse_layers)

    def get_num_in_channels(self):
        return list(self.num_in_channels)

    def forward(self, features):
        if len(features) != self.num_branches:
            raise ValueError(
                f'HRNet expected {self.num_branches} branches, got {len(features)}')
        features = [branch(feature)
                    for branch, feature in zip(self.branches, features)]
        if self.num_branches == 1:
            return features

        fused = []
        for target, transforms in enumerate(self.fuse_layers):
            target_size = features[target].shape[-2:]
            value = None
            for source, transform in enumerate(transforms):
                contribution = transform(features[source])
                if source > target:
                    contribution = F.interpolate(
                        contribution, size=target_size, mode='nearest')
                value = contribution if value is None else value + contribution
            fused.append(self.fuse_act(value))
        return fused


_HRNET_W18 = {
    'STAGE1': {'NUM_MODULES': 1, 'NUM_BRANCHES': 1,
               'BLOCK': 'BOTTLENECK', 'NUM_BLOCKS': (4,),
               'NUM_CHANNELS': (64,)},
    'STAGE2': {'NUM_MODULES': 1, 'NUM_BRANCHES': 2,
               'BLOCK': 'BASIC', 'NUM_BLOCKS': (4, 4),
               'NUM_CHANNELS': (18, 36)},
    'STAGE3': {'NUM_MODULES': 4, 'NUM_BRANCHES': 3,
               'BLOCK': 'BASIC', 'NUM_BLOCKS': (4, 4, 4),
               'NUM_CHANNELS': (18, 36, 72)},
    'STAGE4': {'NUM_MODULES': 3, 'NUM_BRANCHES': 4,
               'BLOCK': 'BASIC', 'NUM_BLOCKS': (4, 4, 4, 4),
               'NUM_CHANNELS': (18, 36, 72, 144)},
}


@register
class HRNetV2W18(nn.Module):
    """HRNetV2-W18 provider returning RT-DETR P3/P4/P5 only."""

    def __init__(self, pretrained=True, pretrained_path=None):
        super().__init__()
        if not isinstance(pretrained, bool):
            raise ValueError('HRNetV2W18.pretrained must be boolean')
        if pretrained_path is not None and not isinstance(pretrained_path, str):
            raise ValueError('HRNetV2W18.pretrained_path must be a string or null')

        self.out_channels = [36, 72, 144]
        self.out_strides = [8, 16, 32]
        self.pretrained_load_report = None

        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=_BN_MOMENTUM)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=_BN_MOMENTUM)
        self.act2 = nn.ReLU(inplace=True)

        stage1 = _HRNET_W18['STAGE1']
        stage1_block = _BLOCKS[stage1['BLOCK']]
        self.layer1 = self._make_layer(
            stage1_block, 64, stage1['NUM_CHANNELS'][0],
            stage1['NUM_BLOCKS'][0])
        previous_channels = [stage1['NUM_CHANNELS'][0]
                             * stage1_block.expansion]

        self.transition1, self.stage2, previous_channels = self._build_stage(
            _HRNET_W18['STAGE2'], previous_channels)
        self.transition2, self.stage3, previous_channels = self._build_stage(
            _HRNET_W18['STAGE3'], previous_channels)
        self.transition3, self.stage4, previous_channels = self._build_stage(
            _HRNET_W18['STAGE4'], previous_channels)
        if previous_channels != [18, 36, 72, 144]:
            raise RuntimeError(f'Unexpected HRNet-W18 stage4 channels: {previous_channels}')

        self._init_weights()
        if pretrained or pretrained_path:
            self._load_pretrained(pretrained_path)

    @staticmethod
    def _make_layer(block, in_channels, channels, blocks):
        out_channels = channels * block.expansion
        downsample = None
        if in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=_BN_MOMENTUM),
            )
        layers = [block(in_channels, channels, downsample=downsample)]
        layers.extend(block(out_channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    @staticmethod
    def _make_transition_layer(previous_channels, current_channels):
        transitions = []
        for index, channels in enumerate(current_channels):
            if index < len(previous_channels):
                if channels == previous_channels[index]:
                    transitions.append(nn.Identity())
                else:
                    transitions.append(nn.Sequential(
                        nn.Conv2d(previous_channels[index], channels, 3,
                                  padding=1, bias=False),
                        nn.BatchNorm2d(channels, momentum=_BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    ))
            else:
                steps = []
                in_channels = previous_channels[-1]
                for step in range(index + 1 - len(previous_channels)):
                    is_last = step == index - len(previous_channels)
                    out_channels = channels if is_last else in_channels
                    steps.append(nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 3, stride=2,
                                  padding=1, bias=False),
                        nn.BatchNorm2d(out_channels, momentum=_BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    ))
                    in_channels = out_channels
                transitions.append(nn.Sequential(*steps))
        return nn.ModuleList(transitions)

    @staticmethod
    def _make_stage(config, input_channels):
        block = _BLOCKS[config['BLOCK']]
        modules = []
        current = list(input_channels)
        for _ in range(config['NUM_MODULES']):
            module = HighResolutionModule(
                config['NUM_BRANCHES'], block, config['NUM_BLOCKS'], current,
                config['NUM_CHANNELS'], multi_scale_output=True)
            modules.append(module)
            current = module.get_num_in_channels()
        return nn.Sequential(*modules), current

    def _build_stage(self, config, previous_channels):
        block = _BLOCKS[config['BLOCK']]
        current_channels = [channels * block.expansion
                            for channels in config['NUM_CHANNELS']]
        transition = self._make_transition_layer(
            previous_channels, current_channels)
        stage, output_channels = self._make_stage(config, current_channels)
        return transition, stage, output_channels

    @staticmethod
    def _apply_transition(features, transition):
        output = []
        for index, transform in enumerate(transition):
            source = features[index] if index < len(features) else features[-1]
            output.append(transform(source))
        return output

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _unwrap_checkpoint(checkpoint):
        if not isinstance(checkpoint, dict):
            raise RuntimeError('HRNet pretrained checkpoint must be a state dictionary')
        for key in ('state_dict', 'model', 'module'):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                checkpoint = nested
                break
        state = {}
        for key, value in checkpoint.items():
            if not torch.is_tensor(value):
                continue
            for prefix in ('module.', 'model.'):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            state[key] = value
        return state

    def _load_pretrained(self, pretrained_path):
        if pretrained_path:
            path = Path(pretrained_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(
                    f'HRNetV2-W18 pretrained file not found: {path}. Expected '
                    f'a compatible {_EXPECTED_FILENAME} checkpoint.')
            checkpoint = torch.load(str(path), map_location='cpu')
            source = str(path.resolve())
        else:
            try:
                checkpoint = torch.hub.load_state_dict_from_url(
                    _PRETRAINED_URL, map_location='cpu', check_hash=True,
                    file_name=_EXPECTED_FILENAME)
            except Exception as error:
                raise RuntimeError(
                    'Unable to load HRNetV2-W18 ImageNet weights. Download '
                    f'{_EXPECTED_FILENAME} into the Torch Hub checkpoint cache '
                    'or set HRNetV2W18.pretrained_path to a local file.') from error
            source = _PRETRAINED_URL

        state = self._unwrap_checkpoint(checkpoint)
        own_state = self.state_dict()
        ignored = sorted(key for key in state
                         if key.startswith(_CLASSIFICATION_PREFIXES))
        unexpected = sorted(key for key in state
                            if key not in own_state
                            and not key.startswith(_CLASSIFICATION_PREFIXES))
        mismatched = sorted(
            key for key in state if key in own_state
            and tuple(state[key].shape) != tuple(own_state[key].shape))
        compatible = {
            key: value for key, value in state.items()
            if key in own_state and key not in mismatched
        }
        incompatible = self.load_state_dict(compatible, strict=False)
        missing = sorted(key for key in incompatible.missing_keys
                         if not key.endswith('num_batches_tracked'))
        if missing or unexpected or mismatched:
            raise RuntimeError(
                'HRNetV2-W18 pretrained backbone mismatch: '
                f'missing={missing}, unexpected={unexpected}, '
                f'shape_mismatch={mismatched}')
        matched = sorted(compatible)
        self.pretrained_load_report = {
            'source': source,
            'matched_keys': matched,
            'missing_keys': missing,
            'unexpected_keys': unexpected,
            'ignored_classifier_keys': ignored,
        }
        print('HRNetV2-W18 pretrained: '
              f'matched={len(matched)}, missing={missing}, '
              f'unexpected={unexpected}, ignored_classifier={len(ignored)}')

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        features = self.stage2(self._apply_transition([x], self.transition1))
        features = self.stage3(self._apply_transition(features, self.transition2))
        features = self.stage4(self._apply_transition(features, self.transition3))

        # RT-DETR remains a three-level detector. The stride-4 HRNet branch is
        # maintained and fused internally, but is not sent to HybridEncoder.
        outputs = [features[index] for index in (1, 2, 3)]
        # The last stage's stride-4 fusion result is intentionally not a
        # detector level. Keep a zero-valued autograd dependency during
        # training so every standard HRNet fusion parameter participates in
        # DDP reduction even when find_unused_parameters is disabled. This is
        # exactly zero, so it cannot change a feature value or inference graph.
        if self.training and torch.is_grad_enabled():
            outputs[0] = outputs[0] + features[0].sum() * 0.0
        expected = zip(outputs, self.out_channels, self.out_strides)
        for output, channels, stride in expected:
            if output.shape[1] != channels:
                raise RuntimeError(
                    f'HRNet output stride {stride} has {output.shape[1]} '
                    f'channels, expected {channels}')
        return outputs
