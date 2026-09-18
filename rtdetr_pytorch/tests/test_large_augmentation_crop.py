"""Use a small temporary PIL limit to reproduce large augmentation safely."""

import io
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from tests._support import prepare_imports
prepare_imports()

from torchvision import datapoints
import torchvision.transforms.v2 as T
from src.data.transforms import RandomIoUCrop


class LargeAugmentationCropTests(unittest.TestCase):
    def image(self, mode='RGB'):
        channels = {'L': 1, 'P': 1, 'RGB': 3, 'RGBA': 4}[mode]
        shape = (12, 16) if channels == 1 else (12, 16, channels)
        array = (np.arange(np.prod(shape)).reshape(shape) % 256).astype(np.uint8)
        result = Image.fromarray(array, mode=mode)
        result.info['test_metadata'] = 'preserved'
        if mode == 'P':
            result.putpalette([i % 256 for i in range(768)])
        return result

    def params(self, left=1, top=1, width=14, height=10):
        return {'left': left, 'top': top, 'width': width, 'height': height,
                'is_within_crop_area': torch.tensor([True])}

    def test_zoomout_crop_reproduces_error_and_fixed_pipeline_succeeds(self):
        image = self.image()
        target = {'boxes': datapoints.BoundingBox(
            [[4, 3, 12, 9]], format='XYXY', spatial_size=(12, 16), dtype=torch.float32),
            'labels': torch.tensor([0])}
        # Original image: 192 pixels; zoomed canvas: 768 pixels.
        with patch.object(Image, 'MAX_IMAGE_PIXELS', 200):
            zoomed, target = T.RandomZoomOut(side_range=(2., 2.), p=1.)(image, target)
            params = self.params(left=0, top=0, width=32, height=24)
            with self.assertRaises(Image.DecompressionBombError):
                T.RandomIoUCrop()._transform(zoomed, params)
            crop = RandomIoUCrop(p=1.)
            with patch.object(crop, '_get_params', return_value=params):
                cropped, transformed = crop(zoomed, target)
            self.assertEqual(cropped.size, zoomed.size)
            np.testing.assert_array_equal(np.array(cropped), np.array(zoomed))
            torch.testing.assert_close(transformed['boxes'], target['boxes'])
            torch.testing.assert_close(transformed['labels'], target['labels'])
            self.assertEqual(Image.MAX_IMAGE_PIXELS, 200)

    def test_large_bounded_crop_exact_native_pixels_modes_palette_and_metadata(self):
        params = self.params()
        for mode in ('L', 'P', 'RGB', 'RGBA'):
            with self.subTest(mode=mode):
                image = self.image(mode)
                expected = T.RandomIoUCrop()._transform(image, params)
                with patch.object(Image, 'MAX_IMAGE_PIXELS', 50):
                    actual = RandomIoUCrop()._transform(image, params)
                    self.assertEqual(Image.MAX_IMAGE_PIXELS, 50)
                self.assertEqual(actual.mode, expected.mode)
                self.assertEqual(actual.size, expected.size)
                self.assertEqual(actual.info, expected.info)
                self.assertEqual(actual.getpalette(), expected.getpalette())
                np.testing.assert_array_equal(np.array(actual), np.array(expected))

    def test_small_crop_still_uses_native_pil_crop(self):
        image = self.image()
        params = self.params(width=5, height=5)
        with patch.object(Image, 'MAX_IMAGE_PIXELS', 100), \
                patch.object(image, 'crop', wraps=image.crop) as native:
            result = RandomIoUCrop()._transform(image, params)
        native.assert_called_once_with((1, 1, 6, 6))
        self.assertEqual(result.size, (5, 5))

    def test_tensor_and_box_transforms_are_still_exact_native(self):
        params = self.params()
        inputs = [torch.arange(3 * 12 * 16).reshape(3, 12, 16).float(),
                  datapoints.BoundingBox([[3, 2, 12, 9]], format='XYXY',
                                        spatial_size=(12, 16), dtype=torch.float32)]
        for value in inputs:
            expected = T.RandomIoUCrop()._transform(value.clone(), params)
            with patch.object(Image, 'MAX_IMAGE_PIXELS', 50):
                actual = RandomIoUCrop()._transform(value.clone(), params)
            torch.testing.assert_close(actual, expected)
            if isinstance(actual, datapoints.BoundingBox):
                self.assertEqual(actual.spatial_size, expected.spatial_size)

    def test_out_of_bounds_large_crop_cannot_bypass_safety(self):
        with patch.object(Image, 'MAX_IMAGE_PIXELS', 50):
            with self.assertRaises(Image.DecompressionBombError):
                RandomIoUCrop()._transform(self.image(), self.params(width=100, height=100))
            self.assertEqual(Image.MAX_IMAGE_PIXELS, 50)

    def test_image_open_security_remains_enabled(self):
        image = self.image()
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        with patch.object(Image, 'MAX_IMAGE_PIXELS', 50):
            RandomIoUCrop()._transform(image, self.params())
            with self.assertRaises(Image.DecompressionBombError):
                Image.open(buffer)
            self.assertEqual(Image.MAX_IMAGE_PIXELS, 50)

    def test_noninteger_window_still_uses_native_checks(self):
        with patch.object(Image, 'MAX_IMAGE_PIXELS', 50):
            with self.assertRaises(Image.DecompressionBombError):
                RandomIoUCrop()._transform(self.image(), self.params(width=14., height=10.))

    def test_disabled_crop_and_empty_params_return_same_image(self):
        image = self.image()
        self.assertIs(RandomIoUCrop()._transform(image, {}), image)
        with patch.object(Image, 'MAX_IMAGE_PIXELS', 50):
            self.assertIs(RandomIoUCrop(p=0.)(image), image)
            self.assertEqual(Image.MAX_IMAGE_PIXELS, 50)

    def test_user_disabled_pil_limit_uses_native_crop(self):
        image = self.image()
        with patch.object(Image, 'MAX_IMAGE_PIXELS', None), \
                patch.object(image, 'crop', wraps=image.crop) as native:
            RandomIoUCrop()._transform(image, self.params())
        native.assert_called_once()

    def test_large_crop_does_not_change_rng_or_input(self):
        image = self.image()
        before = np.array(image)
        state = torch.get_rng_state().clone()
        limit = Image.MAX_IMAGE_PIXELS
        with patch.object(Image, 'MAX_IMAGE_PIXELS', 50):
            RandomIoUCrop()._transform(image, self.params())
        self.assertEqual(Image.MAX_IMAGE_PIXELS, limit)
        torch.testing.assert_close(torch.get_rng_state(), state)
        np.testing.assert_array_equal(np.array(image), before)


if __name__ == '__main__':
    unittest.main()
