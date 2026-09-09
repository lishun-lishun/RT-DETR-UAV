"""Minimal regression tests for paired Decoder Cross-Attention."""

import unittest

import torch

from src.zoo.rtdetr.paired_target_surround_attention import \
    PairedTargetSurroundDeformableAttention
from src.zoo.rtdetr.rtdetr_decoder import (
    MSDeformableAttention,
    RTDETRTransformer,
    TransformerDecoderLayer,
)


class PairedTargetSurroundAttentionTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(7)
        self.batch_size = 2
        self.num_queries = 5
        self.embed_dim = 32
        self.num_heads = 4
        self.num_levels = 3
        self.num_points = 2
        self.spatial_shapes = [[2, 2], [1, 2], [1, 1]]
        self.value_length = sum(h * w for h, w in self.spatial_shapes)

    def _inputs(self, reference_dim=4):
        query = torch.randn(
            self.batch_size, self.num_queries, self.embed_dim,
            requires_grad=True)
        value = torch.randn(
            self.batch_size, self.value_length, self.embed_dim,
            requires_grad=True)
        if reference_dim == 4:
            xy = torch.rand(self.batch_size, self.num_queries, 1, 2)
            wh = 0.1 + 0.4 * torch.rand(
                self.batch_size, self.num_queries, 1, 2)
            reference_points = torch.cat([xy, wh], dim=-1)
        else:
            reference_points = torch.rand(
                self.batch_size, self.num_queries, self.num_levels, 2)
        return query, reference_points, value

    def _base_attention(self):
        return MSDeformableAttention(
            self.embed_dim, self.num_heads,
            self.num_levels, self.num_points)

    def _transformer(self, cross_attention=None):
        return RTDETRTransformer(
            num_classes=1,
            hidden_dim=self.embed_dim,
            num_queries=10,
            feat_channels=[self.embed_dim] * self.num_levels,
            feat_strides=[8, 16, 32],
            num_levels=self.num_levels,
            num_decoder_points=self.num_points,
            nhead=self.num_heads,
            num_decoder_layers=2,
            dim_feedforward=64,
            num_denoising=0,
            eval_spatial_size=None,
            decoder_cross_attention=cross_attention)

    def _transformer_features(self):
        return [
            torch.randn(self.batch_size, self.embed_dim, 4, 4),
            torch.randn(self.batch_size, self.embed_dim, 2, 2),
            torch.randn(self.batch_size, self.embed_dim, 1, 1),
        ]

    def test_baseline_forward_backward(self):
        attention = self._base_attention()
        query, reference_points, value = self._inputs(reference_dim=4)
        output = attention(
            query, reference_points, value, self.spatial_shapes)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.num_queries, self.embed_dim))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertIsNotNone(attention.sampling_offsets.weight.grad)
        self.assertIsNotNone(attention.attention_weights.weight.grad)

    def test_paired_forward_backward_and_debug(self):
        attention = PairedTargetSurroundDeformableAttention(
            self._base_attention(), debug=True)
        query, reference_points, value = self._inputs(reference_dim=4)
        output = attention(
            query, reference_points, value, self.spatial_shapes)

        # Shape and query count remain unchanged.
        self.assertEqual(
            output.shape,
            (self.batch_size, self.num_queries, self.embed_dim))
        self.assertTrue(torch.isfinite(output).all())
        expected_debug_fields = {
            'query_shape', 'reference_points_shape',
            'target_locations_shape', 'surround_locations_shape',
            'z_target_shape', 'z_surround_shape', 'z_diff_shape',
            'gate_shape', 'output_shape', 'gate_mean', 'gate_min', 'gate_max',
            'target_location_min', 'target_location_max',
            'surround_location_min', 'surround_location_max',
            'surround_clamp_ratio', 'z_target_abs_mean',
            'z_surround_abs_mean', 'z_diff_abs_mean', 'reference_boxes',
            'target_locations', 'surround_locations', 'z_diff',
            'differential_evidence',
        }
        self.assertTrue(
            expected_debug_fields.issubset(attention.last_debug_info))

        output.square().mean().backward()
        for parameter in (
                attention.sampling_offsets.weight,
                attention.attention_weights.weight,
                attention.diff_gate.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_point_references_are_supported(self):
        attention = PairedTargetSurroundDeformableAttention(
            self._base_attention())
        query, reference_points, value = self._inputs(reference_dim=2)
        output = attention(
            query, reference_points, value, self.spatial_shapes)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.num_queries, self.embed_dim))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(attention.last_debug_info, {})

    def test_disabled_switch_uses_exact_original_class(self):
        torch.manual_seed(11)
        default_layer = TransformerDecoderLayer(
            d_model=self.embed_dim, n_head=self.num_heads,
            n_levels=self.num_levels, n_points=self.num_points)
        torch.manual_seed(11)
        disabled_layer = TransformerDecoderLayer(
            d_model=self.embed_dim, n_head=self.num_heads,
            n_levels=self.num_levels, n_points=self.num_points,
            decoder_cross_attention={
                'type': 'paired_target_surround',
                'enabled': False,
            })
        self.assertIs(type(default_layer.cross_attn), MSDeformableAttention)
        self.assertIs(type(disabled_layer.cross_attn), MSDeformableAttention)
        for name, tensor in default_layer.state_dict().items():
            self.assertTrue(
                torch.equal(tensor, disabled_layer.state_dict()[name]))

        query, reference_points, value = self._inputs(reference_dim=4)
        default_output = default_layer.cross_attn(
            query, reference_points, value, self.spatial_shapes)
        disabled_output = disabled_layer.cross_attn(
            query, reference_points, value, self.spatial_shapes)
        self.assertTrue(torch.equal(default_output, disabled_output))

    def test_baseline_checkpoint_matches_all_original_parameters(self):
        baseline = self._base_attention()
        paired = PairedTargetSurroundDeformableAttention(
            self._base_attention())
        load_result = paired.load_state_dict(
            baseline.state_dict(), strict=True)
        self.assertEqual(load_result.missing_keys, [])
        self.assertEqual(load_result.unexpected_keys, [])
        for name, tensor in baseline.state_dict().items():
            self.assertTrue(torch.equal(tensor, paired.state_dict()[name]))

    def test_decoder_layer_checkpoint_matching(self):
        baseline_layer = TransformerDecoderLayer(
            d_model=self.embed_dim, n_head=self.num_heads,
            n_levels=self.num_levels, n_points=self.num_points)
        paired_layer = TransformerDecoderLayer(
            d_model=self.embed_dim, n_head=self.num_heads,
            n_levels=self.num_levels, n_points=self.num_points,
            decoder_cross_attention={
                'type': 'paired_target_surround',
                'surround': {'scale': 1.8},
            })

        baseline_state = baseline_layer.state_dict()
        paired_state = paired_layer.state_dict()
        self.assertTrue(set(baseline_state).issubset(set(paired_state)))
        self.assertEqual(
            set(paired_state) - set(baseline_state),
            {
                'cross_attn.diff_gate.weight',
                'cross_attn.diff_gate.bias',
            })
        load_result = paired_layer.load_state_dict(
            baseline_state, strict=True)
        self.assertEqual(load_result.missing_keys, [])
        self.assertEqual(load_result.unexpected_keys, [])

    def test_full_transformer_train_eval_and_query_count(self):
        baseline = self._transformer()
        paired = self._transformer({
            'type': 'paired_target_surround',
            'surround': {'scale': 1.8},
            'differential': {'enabled': True},
            'gated_fusion': {'enabled': True, 'init_bias': -2.0},
        })

        baseline.train()
        train_output = baseline(self._transformer_features())
        train_loss = (train_output['pred_logits'].square().mean() +
                      train_output['pred_boxes'].square().mean())
        train_loss.backward()
        self.assertTrue(torch.isfinite(train_loss))

        baseline.eval()
        paired.eval()
        with torch.no_grad():
            baseline_output = baseline(self._transformer_features())
            paired_output = paired(self._transformer_features())
        self.assertEqual(baseline_output['pred_logits'].shape[1], 10)
        self.assertEqual(paired_output['pred_logits'].shape[1], 10)
        self.assertEqual(
            baseline_output['pred_boxes'].shape,
            paired_output['pred_boxes'].shape)
        self.assertTrue(torch.isfinite(paired_output['pred_logits']).all())
        self.assertTrue(torch.isfinite(paired_output['pred_boxes']).all())


if __name__ == '__main__':
    unittest.main()
