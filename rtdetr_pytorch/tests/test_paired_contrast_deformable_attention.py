"""Regression tests for Paired Contrast Decoder Cross-Attention."""

import unittest

import torch

from src.zoo.rtdetr.paired_contrast_deformable_attention import (
    PairedContrastDeformableAttention,
)
from src.zoo.rtdetr.rtdetr_decoder import (
    MSDeformableAttention,
    RTDETRTransformer,
    TransformerDecoder,
    TransformerDecoderLayer,
)


class PairedContrastDeformableAttentionTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(23)
        self.batch_size = 2
        self.num_matching_queries = 5
        self.embed_dim = 32
        self.num_heads = 4
        self.num_levels = 3
        self.num_points = 2
        self.spatial_shapes = [[3, 3], [2, 2], [1, 1]]
        self.value_length = sum(h * w for h, w in self.spatial_shapes)

    def _base_attention(self):
        return MSDeformableAttention(
            self.embed_dim, self.num_heads,
            self.num_levels, self.num_points)

    def _inputs(self, num_queries=None):
        if num_queries is None:
            num_queries = self.num_matching_queries
        query = torch.randn(
            self.batch_size, num_queries, self.embed_dim,
            requires_grad=True)
        value = torch.randn(
            self.batch_size, self.value_length, self.embed_dim,
            requires_grad=True)
        center = torch.full(
            (self.batch_size, num_queries, 1, 2), 0.5)
        size = torch.full(
            (self.batch_size, num_queries, 1, 2), 0.2)
        reference_boxes = torch.cat([center, size], dim=-1)
        return query, reference_boxes, value

    def _paired_attention(self, contrast_lambda=0.5, debug=False):
        return PairedContrastDeformableAttention(
            self._base_attention(),
            layer_index=1,
            num_matching_queries=self.num_matching_queries,
            enabled_levels=(0, 1),
            surround_margin=0.20,
            contrast_lambda=contrast_lambda,
            normalize_feature=True,
            enable_surround_for_dn=False,
            debug=debug)

    @staticmethod
    def _config(enabled=True, contrast_lambda=0.5, layers=(4, 5)):
        return {
            'type': 'paired_contrast',
            'enabled': enabled,
            'surround': {
                'mode': 'box_outside_radial',
                'margin': 0.20,
            },
            'contrast': {
                'mode': 'abs_difference',
                'scorer': 'mean',
                'lambda': contrast_lambda,
                'normalize_feature': True,
            },
            'layers': {'enabled_layers': list(layers)},
            'levels': {'enabled_levels': [0, 1]},
            'small_query': {
                'enabled': False,
                'area_threshold': 0.02,
            },
            'denoising': {'enable_surround_for_dn': False},
            'invalid_point': {'mode': 'mask'},
            'debug': False,
        }

    def test_baseline_and_disabled_build_original_attention(self):
        original_template = TransformerDecoderLayer(
            self.embed_dim, self.num_heads, n_levels=self.num_levels,
            n_points=self.num_points)
        disabled_template = TransformerDecoderLayer(
            self.embed_dim, self.num_heads, n_levels=self.num_levels,
            n_points=self.num_points,
            decoder_cross_attention=self._config(enabled=False),
            num_matching_queries=self.num_matching_queries)
        original = TransformerDecoder(
            self.embed_dim, original_template, num_layers=6)
        disabled = TransformerDecoder(
            self.embed_dim, disabled_template, num_layers=6)
        for layer in list(original.layers) + list(disabled.layers):
            self.assertIs(type(layer.cross_attn), MSDeformableAttention)

    def test_only_selected_decoder_layers_are_replaced(self):
        template = TransformerDecoderLayer(
            self.embed_dim, self.num_heads, n_levels=self.num_levels,
            n_points=self.num_points,
            decoder_cross_attention=self._config(layers=(4, 5)),
            num_matching_queries=self.num_matching_queries)
        decoder = TransformerDecoder(
            self.embed_dim, template, num_layers=6)
        for index, layer in enumerate(decoder.layers):
            expected = (PairedContrastDeformableAttention
                        if index in (4, 5) else MSDeformableAttention)
            self.assertIs(type(layer.cross_attn), expected)

    def test_forward_backward_shape_and_numerical_stability(self):
        attention = self._paired_attention()
        query, reference_boxes, value = self._inputs()
        output = attention(
            query, reference_boxes, value, self.spatial_shapes)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.num_matching_queries, self.embed_dim))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(attention.last_debug_info, {})

        output.square().mean().backward()
        for parameter in (
                attention.sampling_offsets.weight,
                attention.attention_weights.weight,
                attention.value_proj.weight,
                attention.output_proj.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_checkpoint_keys_are_identical(self):
        baseline = self._base_attention()
        paired = self._paired_attention()
        self.assertEqual(
            set(baseline.state_dict()), set(paired.state_dict()))
        load_result = paired.load_state_dict(
            baseline.state_dict(), strict=True)
        self.assertEqual(load_result.missing_keys, [])
        self.assertEqual(load_result.unexpected_keys, [])

    def test_lambda_zero_is_exact_baseline(self):
        baseline = self._base_attention().eval()
        paired = self._paired_attention(contrast_lambda=0.).eval()
        paired.load_state_dict(baseline.state_dict(), strict=True)
        query, reference_boxes, value = self._inputs()
        baseline_output = baseline(
            query, reference_boxes, value, self.spatial_shapes)
        paired_output = paired(
            query, reference_boxes, value, self.spatial_shapes)
        difference = (baseline_output - paired_output).abs()
        self.assertEqual(float(difference.max()), 0.)
        self.assertEqual(float(difference.mean()), 0.)

    def test_dn_and_disabled_level_contrast_are_zero(self):
        num_dn_queries = 2
        attention = self._paired_attention(debug=True)
        query, reference_boxes, value = self._inputs(
            num_queries=self.num_matching_queries + num_dn_queries)
        output = attention(
            query, reference_boxes, value, self.spatial_shapes)
        self.assertTrue(torch.isfinite(output).all())

        info = attention.last_debug_info
        scores = info['full_contrast_scores']
        self.assertEqual(info['num_denoising_queries'], num_dn_queries)
        self.assertTrue(torch.equal(
            scores[:, :num_dn_queries],
            torch.zeros_like(scores[:, :num_dn_queries])))
        self.assertTrue(torch.equal(
            scores[:, :, :, 2],
            torch.zeros_like(scores[:, :, :, 2])))
        self.assertLessEqual(
            info['surround_inside_reference_box_ratio'], 1e-6)
        self.assertGreaterEqual(info['valid_surround_ratio'], 0.)
        self.assertLessEqual(info['valid_surround_ratio'], 1.)

    def test_degenerate_direction_is_masked_without_nan(self):
        attention = self._paired_attention(debug=True)
        torch.nn.init.constant_(attention.sampling_offsets.weight, 0.)
        torch.nn.init.constant_(attention.sampling_offsets.bias, 0.)
        query, reference_boxes, value = self._inputs()
        output = attention(
            query, reference_boxes, value, self.spatial_shapes)
        info = attention.last_debug_info
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(info['valid_surround_ratio'], 0.)
        self.assertEqual(info['invalid_surround_ratio'], 1.)
        self.assertTrue(torch.equal(
            info['contrast_scores'],
            torch.zeros_like(info['contrast_scores'])))

    def test_out_of_image_surround_is_masked(self):
        attention = self._paired_attention(debug=True)
        query, reference_boxes, value = self._inputs()
        reference_boxes[..., 0] = 0.01
        output = attention(
            query, reference_boxes, value, self.spatial_shapes)
        info = attention.last_debug_info
        enabled_scores = info['contrast_scores'][
            :, :, :, attention.enabled_levels, :]
        valid_mask = info['valid_surround_mask']
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreater(info['invalid_surround_ratio'], 0.)
        self.assertTrue(torch.equal(
            enabled_scores.masked_select(~valid_mask),
            torch.zeros_like(enabled_scores.masked_select(~valid_mask))))

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

    def test_full_transformer_train_eval_and_query_count(self):
        baseline = self._transformer()
        paired = self._transformer(
            self._config(layers=(0, 1)))
        baseline.train()
        train_output = baseline(self._transformer_features())
        train_loss = (train_output['pred_logits'].square().mean() +
                      train_output['pred_boxes'].square().mean())
        train_loss.backward()
        self.assertTrue(torch.isfinite(train_loss))

        paired.train()
        paired_train_output = paired(self._transformer_features())
        paired_train_loss = (
            paired_train_output['pred_logits'].square().mean() +
            paired_train_output['pred_boxes'].square().mean())
        paired_train_loss.backward()
        self.assertTrue(torch.isfinite(paired_train_loss))

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
