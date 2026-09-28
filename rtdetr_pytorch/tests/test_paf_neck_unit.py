"""Focused unit tests for Phase-Adaptive Fusion."""

import unittest

import torch

from tests._support import prepare_imports


prepare_imports()

from src.zoo.rtdetr.paf_neck import PhaseAdaptiveFusion  # noqa: E402


class PhaseAdaptiveFusionTests(unittest.TestCase):

    def test_candidates_have_fixed_order_and_never_wrap(self):
        value = torch.tensor([[[[1., 2., 3.],
                                [4., 5., 6.]]]])
        phases = PhaseAdaptiveFusion.phase_candidates(value)
        expected = torch.tensor([[
            [[[1., 2., 3.], [4., 5., 6.]]],
            [[[0., 1., 2.], [0., 4., 5.]]],
            [[[0., 0., 0.], [1., 2., 3.]]],
            [[[0., 0., 0.], [0., 1., 2.]]],
        ]])
        self.assertTrue(torch.equal(phases, expected))
        # In particular, right/down shifts must not reappear at the opposite
        # boundary as they would with torch.roll.
        self.assertEqual(phases[0, 1, 0, 0, 0].item(), 0.0)
        self.assertEqual(phases[0, 2, 0, 0, -1].item(), 0.0)

    def test_phase_softmax_and_output_shape(self):
        module = PhaseAdaptiveFusion(8, query_dim=4)
        shallow = torch.randn(2, 8, 7, 9)
        upsampled = torch.randn_like(shallow)
        output, aux = module(shallow, upsampled, return_aux=True)
        self.assertEqual(output.shape, shallow.shape)
        self.assertEqual(aux['attention'].shape, (2, 4, 7, 9))
        self.assertEqual(aux['attention'].dtype, torch.float32)
        self.assertTrue(torch.allclose(
            aux['attention'].sum(dim=1), torch.ones(2, 7, 9),
            atol=1e-6, rtol=1e-6))

    def test_dynamic_shapes_and_all_parameters_receive_gradients(self):
        module = PhaseAdaptiveFusion(8, query_dim=4)
        for side in (60, 80, 100):
            shallow = torch.randn(1, 8, side, side, requires_grad=True)
            upsampled = torch.randn(1, 8, side, side, requires_grad=True)
            output = module(shallow, upsampled)
            self.assertEqual(output.shape, shallow.shape)
            output.square().mean().backward()
            for name, parameter in module.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            module.zero_grad(set_to_none=True)

    def test_debug_statistics(self):
        module = PhaseAdaptiveFusion(
            4, query_dim=2, debug=True, debug_interval=1)
        module(torch.randn(1, 4, 3, 5), torch.randn(1, 4, 3, 5))
        self.assertEqual(set(module.last_debug_stats), {
            'phase00_mean', 'phase01_mean', 'phase10_mean',
            'phase11_mean', 'phase_entropy'})
        self.assertAlmostEqual(sum(
            module.last_debug_stats[name] for name in (
                'phase00_mean', 'phase01_mean',
                'phase10_mean', 'phase11_mean')), 1.0, places=6)


if __name__ == '__main__':
    unittest.main()

