# -*- coding: utf-8 -*-
"""
Product Photo Editor - Image Pipeline Unit Tests
Tests image processing algorithms, OpenCV perspective correction,
color enhancement, procedural backgrounds, and external API mocking.
"""

import base64
import io
import os
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from PIL import Image, ImageDraw

from models.image_pipeline import ImagePipeline, ImagePipelineError


class TestImagePipeline(unittest.TestCase):
    """Deep verification of the ImagePipeline engine."""

    @classmethod
    def setUpClass(cls):
        # Create a synthetic test product image (a blue box on a light grey background)
        cls.img_w, cls.img_h = 400, 300
        img = Image.new('RGB', (cls.img_w, cls.img_h), (230, 230, 230))
        draw = ImageDraw.Draw(img)
        # Draw a product-like colored rectangle in the center
        draw.rectangle([(100, 60), (300, 240)], fill=(40, 100, 210), outline=(20, 60, 150), width=3)
        # Add a white label inside product
        draw.rectangle([(140, 120), (260, 180)], fill=(255, 255, 255))

        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        cls.sample_image_bytes = buffer.getvalue()

    def test_01_local_processing_white_background(self):
        """Test processing with local GrabCut engine and 2000x2000 pure white background."""
        result = ImagePipeline.process_image(
            image_bytes=self.sample_image_bytes,
            background_style='white',
            dimensions='square_2000',
            export_format='JPEG',
            export_quality=85,
            apply_perspective=True,
            apply_color_correction=True,
            service_provider='local',
        )

        self.assertIn('image_bytes', result)
        self.assertEqual(result['format'], 'JPEG')
        self.assertEqual(result['width'], 2000)
        self.assertEqual(result['height'], 2000)
        self.assertGreater(result['file_size'], 1000)
        self.assertGreater(result['duration_sec'], 0)
        self.assertTrue(any("Decoded" in s for s in result['steps_applied']))
        self.assertTrue(any("Segmented" in s for s in result['steps_applied']))

        # Verify output image is valid JPEG and exactly 2000x2000
        out_img = Image.open(io.BytesIO(result['image_bytes']))
        self.assertEqual(out_img.size, (2000, 2000))
        self.assertEqual(out_img.mode, 'RGB')
        # Check top-left pixel is pure white #FFFFFF
        tl_pixel = out_img.getpixel((0, 0))
        self.assertEqual(tl_pixel, (255, 255, 255))

    def test_02_transparent_background_png(self):
        """Test transparent background output with PNG export."""
        result = ImagePipeline.process_image(
            image_bytes=self.sample_image_bytes,
            background_style='transparent',
            dimensions='square_1000',
            export_format='PNG',
            service_provider='local',
        )

        self.assertEqual(result['format'], 'PNG')
        self.assertEqual(result['width'], 1000)
        self.assertEqual(result['height'], 1000)

        out_img = Image.open(io.BytesIO(result['image_bytes']))
        self.assertEqual(out_img.mode, 'RGBA')
        # Corner pixel should be fully transparent (alpha = 0)
        corner_pixel = out_img.getpixel((0, 0))
        self.assertEqual(corner_pixel[3], 0)

    def test_03_transparent_with_jpeg_requested_auto_switch(self):
        """When transparent style is requested with JPEG, format must switch to PNG."""
        result = ImagePipeline.process_image(
            image_bytes=self.sample_image_bytes,
            background_style='transparent',
            dimensions='original',
            export_format='JPEG',
            service_provider='local',
        )
        self.assertEqual(result['format'], 'PNG')
        out_img = Image.open(io.BytesIO(result['image_bytes']))
        self.assertEqual(out_img.mode, 'RGBA')

    def test_04_procedural_background_styles(self):
        """Test all procedural background styles: studio neutral, soft shadow, wood, marble, gradient."""
        styles = [
            'studio_neutral',
            'studio_soft_shadow',
            'lifestyle_wood',
            'lifestyle_marble',
            'lifestyle_gradient',
            'custom_color',
        ]

        for style in styles:
            with self.subTest(style=style):
                result = ImagePipeline.process_image(
                    image_bytes=self.sample_image_bytes,
                    background_style=style,
                    custom_bg_color='#E0F7FA',
                    dimensions='square_1000',
                    export_format='JPEG',
                    service_provider='local',
                )
                self.assertEqual(result['width'], 1000)
                self.assertEqual(result['height'], 1000)
                out_img = Image.open(io.BytesIO(result['image_bytes']))
                self.assertEqual(out_img.size, (1000, 1000))

    def test_05_dimension_presets(self):
        """Verify each dimension preset produces exact requested canvas size."""
        preset_tests = [
            ('square_2000', (2000, 2000)),
            ('square_1000', (1000, 1000)),
            ('portrait_4_5', (1600, 2000)),
            ('landscape_16_9', (1920, 1080)),
            ('original', (self.img_w, self.img_h)),
        ]

        for preset, expected_size in preset_tests:
            with self.subTest(preset=preset):
                result = ImagePipeline.process_image(
                    image_bytes=self.sample_image_bytes,
                    background_style='white',
                    dimensions=preset,
                    service_provider='local',
                )
                self.assertEqual((result['width'], result['height']), expected_size)

    def test_06_custom_target_dimensions(self):
        """Test specifying explicit custom width and height."""
        result = ImagePipeline.process_image(
            image_bytes=self.sample_image_bytes,
            dimensions='original',
            target_width=1200,
            target_height=800,
            service_provider='local',
        )
        self.assertEqual(result['width'], 1200)
        self.assertEqual(result['height'], 800)

    def test_07_auto_white_balance_and_clahe(self):
        """Verify color correction operates on strong color casts."""
        # Create image with strong orange/yellow color cast
        cast_img = Image.new('RGB', (200, 200), (240, 180, 80))
        cast_buf = io.BytesIO()
        cast_img.save(cast_buf, format='JPEG')

        result = ImagePipeline.process_image(
            image_bytes=cast_buf.getvalue(),
            apply_color_correction=True,
            apply_auto_white_balance=True,
            apply_contrast_enhancement=True,
            apply_sharpening=True,
            service_provider='local',
        )
        self.assertIsNotNone(result['image_bytes'])
        self.assertTrue(any("Gray-World" in s for s in result['steps_applied']))
        self.assertTrue(any("CLAHE" in s for s in result['steps_applied']))

    def test_08_perspective_and_deskew(self):
        """Test perspective correction / deskew on a rotated polygon."""
        rot_img = Image.new('RGB', (400, 400), (245, 245, 245))
        draw = ImageDraw.Draw(rot_img)
        # Draw a tilted box
        draw.polygon([(120, 80), (320, 110), (290, 310), (90, 280)], fill=(30, 30, 30))
        rot_buf = io.BytesIO()
        rot_img.save(rot_buf, format='PNG')

        result = ImagePipeline.process_image(
            image_bytes=rot_buf.getvalue(),
            apply_perspective=True,
            service_provider='local',
        )
        self.assertIsNotNone(result['image_bytes'])

    @patch('models.image_pipeline.requests')
    def test_09_mock_remove_bg_api(self, mock_requests):
        """Test external remove.bg API call with valid mock response."""
        # Mock 200 response returning a 200x200 RGBA image
        cutout = Image.new('RGBA', (200, 200), (10, 150, 80, 255))
        c_buf = io.BytesIO()
        cutout.save(c_buf, format='PNG')

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = c_buf.getvalue()
        mock_requests.post.return_value = mock_resp

        result = ImagePipeline.process_image(
            image_bytes=self.sample_image_bytes,
            service_provider='remove_bg',
            provider_config={'remove_bg_api_key': 'fake_test_key_123'},
            dimensions='square_1000',
        )

        mock_requests.post.assert_called_once()
        self.assertEqual(result['width'], 1000)
        self.assertEqual(result['height'], 1000)

    @patch('models.image_pipeline.requests')
    def test_10_mock_clipdrop_api(self, mock_requests):
        """Test Clipdrop API call with mock response."""
        cutout = Image.new('RGBA', (150, 150), (200, 50, 50, 255))
        c_buf = io.BytesIO()
        cutout.save(c_buf, format='PNG')

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = c_buf.getvalue()
        mock_requests.post.return_value = mock_resp

        result = ImagePipeline.process_image(
            image_bytes=self.sample_image_bytes,
            service_provider='clipdrop',
            provider_config={'clipdrop_api_key': 'fake_clipdrop_token'},
            dimensions='square_1000',
        )

        mock_requests.post.assert_called_once()
        self.assertEqual(result['width'], 1000)

    @patch('models.image_pipeline.requests')
    def test_11_mock_photoroom_api(self, mock_requests):
        """Test Photoroom API call with mock response."""
        cutout = Image.new('RGBA', (180, 180), (50, 200, 50, 255))
        c_buf = io.BytesIO()
        cutout.save(c_buf, format='PNG')

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = c_buf.getvalue()
        mock_requests.post.return_value = mock_resp

        result = ImagePipeline.process_image(
            image_bytes=self.sample_image_bytes,
            service_provider='photoroom',
            provider_config={'photoroom_api_key': 'fake_photoroom_token'},
            dimensions='square_1000',
        )

        mock_requests.post.assert_called_once()
        self.assertEqual(result['width'], 1000)

    @patch('models.image_pipeline.requests')
    def test_12_mock_custom_ai_webhook(self, mock_requests):
        """Test custom AI inference endpoint returning JSON with base64 image."""
        cutout = Image.new('RGBA', (160, 160), (120, 80, 220, 255))
        c_buf = io.BytesIO()
        cutout.save(c_buf, format='PNG')
        b64_cutout = base64.b64encode(c_buf.getvalue()).decode('utf-8')

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {'Content-Type': 'application/json'}
        mock_resp.json.return_value = {'image_base64': b64_cutout}
        mock_requests.post.return_value = mock_resp

        result = ImagePipeline.process_image(
            image_bytes=self.sample_image_bytes,
            service_provider='custom_webhook',
            provider_config={
                'custom_ai_endpoint_url': 'https://custom-ai.internal/v1/segment',
                'custom_ai_api_key': 'secret-token',
            },
            dimensions='square_1000',
        )

        mock_requests.post.assert_called_once()
        self.assertEqual(result['width'], 1000)

    def test_13_edge_case_empty_or_corrupted_input(self):
        """Verify errors are raised on invalid input."""
        with self.assertRaises(ImagePipelineError):
            ImagePipeline.process_image(image_bytes=b'')

        with self.assertRaises(ImagePipelineError):
            ImagePipeline.process_image(image_bytes=b'not_a_valid_image_header_bytes')

    def test_14_edge_case_extreme_aspect_ratio(self):
        """Test processing of an ultra-wide banner-like image (600x40)."""
        banner = Image.new('RGB', (600, 40), (220, 220, 220))
        b_draw = ImageDraw.Draw(banner)
        b_draw.rectangle([(50, 10), (550, 30)], fill=(10, 80, 180))
        b_buf = io.BytesIO()
        banner.save(b_buf, format='PNG')

        result = ImagePipeline.process_image(
            image_bytes=b_buf.getvalue(),
            dimensions='square_1000',
            service_provider='local',
        )
        self.assertEqual(result['width'], 1000)
        self.assertEqual(result['height'], 1000)


    def test_15_deskew_positive_and_negative_angles(self):
        """Verify OpenCV deskewing straightens both clockwise and counter-clockwise tilts."""
        for angle in [-10, 10]:
            with self.subTest(angle=angle):
                img = Image.new('RGB', (400, 400), (245, 245, 245))
                draw = ImageDraw.Draw(img)
                box = np.array([[-80, -40], [80, -40], [80, 40], [-80, 40]], dtype=np.float32)
                rad = np.radians(angle)
                rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]], dtype=np.float32)
                poly = (np.dot(box, rot.T) + np.array([200, 200])).astype(np.int32)
                draw.polygon([tuple(p) for p in poly], fill=(30, 30, 30))

                cv_img = ImagePipeline._pil_to_cv2(img)
                _, deskew_applied = ImagePipeline._correct_perspective_or_deskew(cv_img)
                self.assertTrue(deskew_applied, f"Failed to deskew tilt at {angle} degrees")

    def test_16_edge_case_small_dimensions(self):
        """Verify processing succeeds on tiny and high-aspect images without OpenCV crash."""
        for size in [(8, 8), (10, 10), (10, 500)]:
            with self.subTest(size=size):
                tiny = Image.new('RGB', size, (100, 150, 200))
                buf = io.BytesIO()
                tiny.save(buf, format='PNG')
                res = ImagePipeline.process_image(
                    image_bytes=buf.getvalue(),
                    dimensions='square_1000',
                    service_provider='local',
                )
                self.assertEqual(res['width'], 1000)
                self.assertEqual(res['height'], 1000)

    def test_17_transparent_png_input_preserved(self):
        """Verify pre-existing alpha transparency is preserved in transparent export."""
        trans = Image.new('RGBA', (200, 200), (0, 0, 0, 0))
        for x in range(60, 140):
            for y in range(60, 140):
                trans.putpixel((x, y), (220, 50, 50, 255))
        buf = io.BytesIO()
        trans.save(buf, format='PNG')

        res = ImagePipeline.process_image(
            image_bytes=buf.getvalue(),
            background_style='transparent',
            dimensions='square_1000',
            service_provider='local',
        )
        self.assertEqual(res['format'], 'PNG')
        out_img = Image.open(io.BytesIO(res['image_bytes']))
        self.assertEqual(out_img.mode, 'RGBA')
        # Corner should be transparent
        self.assertEqual(out_img.getpixel((0, 0))[3], 0)

    @patch('models.image_pipeline.requests')
    def test_18_external_api_fallback_logging(self, mock_requests):
        """Verify that when external AI API fails, local fallback executes and logs accurately."""
        mock_requests.post.side_effect = Exception("API connection timed out")
        res = ImagePipeline.process_image(
            image_bytes=self.sample_image_bytes,
            service_provider='remove_bg',
            provider_config={'remove_bg_api_key': 'fake_key'},
            dimensions='square_1000',
        )
        self.assertEqual(res['width'], 1000)
        self.assertTrue(any("fallback from remove.bg" in s for s in res['steps_applied']))


if __name__ == '__main__':
    unittest.main()
