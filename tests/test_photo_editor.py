# -*- coding: utf-8 -*-
"""
Product Photo Editor - Model & Workflow Tests
Tests the product.photo.editor model, wizard workflows, product template sync,
and background batch cron execution.
"""

import base64
import io
import unittest
from PIL import Image, ImageDraw

from models.photo_editor import _safe_b64decode, ProductPhotoEditor
from wizard.photo_editor_wizard import ProductPhotoEditorWizard

try:
    from odoo.tests.common import TransactionCase, tagged
except ImportError:
    # Allow test collection in environments without Odoo runtime installed
    TransactionCase = unittest.TestCase
    def tagged(*args):
        return lambda cls: cls


@tagged('post_install', '-at_install', 'product_editor')
class TestProductPhotoEditor(TransactionCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not hasattr(cls, 'env'):
            return

        # Create synthetic test image
        img = Image.new('RGB', (200, 200), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.rectangle([(40, 40), (160, 160)], fill=(20, 120, 220))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        cls.test_image_b64 = base64.b64encode(buf.getvalue())

        # Create test product template
        cls.product = cls.env['product.template'].create({
            'name': 'Test Acoustic Speaker',
            'default_code': 'SPK-001',
            'image_1920': cls.test_image_b64,
        })

    def test_01_create_job_and_process(self):
        """Test creating a photo editor job and executing the local processing pipeline."""
        if not hasattr(self, 'env'):
            self.skipTest("Odoo runtime not loaded")

        job = self.env['product.photo.editor'].create({
            'product_id': self.product.id,
            'image_original': self.test_image_b64,
            'background_style': 'white',
            'dimensions': 'square_2000',
            'service_provider': 'local',
        })

        self.assertTrue(job.name)
        self.assertEqual(job.status, 'draft')
        self.assertFalse(job.image_processed)

        # Trigger processing
        job.action_process()

        self.assertEqual(job.status, 'done')
        self.assertTrue(job.image_processed)
        self.assertEqual(job.processed_width, 2000)
        self.assertEqual(job.processed_height, 2000)
        self.assertGreater(job.duration_sec, 0)
        self.assertFalse(job.applied_to_product)

    def test_02_apply_to_product(self):
        """Test applying the processed image to the product template's image_1920."""
        if not hasattr(self, 'env'):
            self.skipTest("Odoo runtime not loaded")

        job = self.env['product.photo.editor'].create({
            'product_id': self.product.id,
            'image_original': self.test_image_b64,
            'background_style': 'studio_neutral',
            'dimensions': 'square_1000',
            'service_provider': 'local',
        })
        job.action_process()
        self.assertNotEqual(self.product.image_1920, job.image_processed)

        # Apply to product
        job.action_apply_to_product()

        self.assertTrue(job.applied_to_product)
        self.assertTrue(job.applied_date)
        self.assertEqual(self.product.image_1920, job.image_processed)

    def test_03_wizard_preview_and_apply(self):
        """Test interactive wizard generation and catalog update."""
        if not hasattr(self, 'env'):
            self.skipTest("Odoo runtime not loaded")

        wizard = self.env['product.photo.editor.wizard'].create({
            'product_id': self.product.id,
            'image_original': self.test_image_b64,
            'background_style': 'lifestyle_wood',
            'dimensions': 'square_1000',
            'service_provider': 'local',
        })

        # Generate on-the-fly preview
        wizard.action_generate_preview()
        self.assertTrue(wizard.preview_image)
        self.assertTrue(wizard.preview_info)

        # Apply and close
        wizard.action_apply_and_close()
        self.assertEqual(self.product.image_1920, wizard.preview_image)

        # Verify a completed job record was logged
        job = self.env['product.photo.editor'].search([('product_id', '=', self.product.id)], order='id desc', limit=1)
        self.assertEqual(job.status, 'done')
        self.assertTrue(job.applied_to_product)

    def test_04_bulk_batch_optimization_and_cron(self):
        """Test batch enqueuing from product template and automated cron processing."""
        if not hasattr(self, 'env'):
            self.skipTest("Odoo runtime not loaded")

        prod2 = self.env['product.template'].create({
            'name': 'Test Wireless Headphones',
            'image_1920': self.test_image_b64,
        })

        # Multi-record action
        products = self.product | prod2
        products.action_batch_photo_editor()

        pending_jobs = self.env['product.photo.editor'].search([
            ('product_id', 'in', products.ids),
            ('status', '=', 'pending'),
        ])
        self.assertGreaterEqual(len(pending_jobs), 2)

        # Run cron
        self.env['product.photo.editor']._cron_batch_process(batch_limit=10)

        # Verify all are completed
        for pj in pending_jobs:
            self.assertEqual(pj.status, 'done')
            self.assertTrue(pj.image_processed)


class TestPhotoEditorModelStandalone(unittest.TestCase):
    """Verifies helper methods, constraints, and decoders without requiring a database."""

    def test_01_safe_b64decode_variants(self):
        sample = b"dummy pixel payload"
        b64_str = base64.b64encode(sample).decode('ascii')
        data_uri = f"data:image/png;base64,{b64_str}"

        # Standard base64 string
        self.assertEqual(_safe_b64decode(b64_str), sample)
        # Data URI
        self.assertEqual(_safe_b64decode(data_uri), sample)
        # Empty inputs
        self.assertEqual(_safe_b64decode(None), b'')
        self.assertEqual(_safe_b64decode(''), b'')

    def test_02_read_image_dimensions(self):
        img = Image.new('RGB', (120, 80), (100, 100, 100))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        data_uri = f"data:image/png;base64,{b64}"

        dims = ProductPhotoEditor._read_image_dimensions(b64)
        self.assertEqual(dims['width'], 120)
        self.assertEqual(dims['height'], 80)
        self.assertGreater(dims['size'], 0)

        dims_uri = ProductPhotoEditor._read_image_dimensions(data_uri)
        self.assertEqual(dims_uri['width'], 120)
        self.assertEqual(dims_uri['height'], 80)

    def test_03_safe_b64decode_unpadded_and_urlsafe(self):
        """Unpadded and URL-safe base64 strings should be decoded cleanly."""
        img = Image.new('RGB', (20, 20), color=(10, 20, 30))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        raw_jpeg = buf.getvalue()
        b64 = base64.b64encode(raw_jpeg)

        # Unpadded (strip '=')
        unpadded = b64.rstrip(b'=')
        decoded_unpadded = _safe_b64decode(unpadded)
        self.assertEqual(decoded_unpadded[:3], b'\xff\xd8\xff')
        self.assertEqual(Image.open(io.BytesIO(decoded_unpadded)).size, (20, 20))

        # URL-safe (replace + with - and / with _)
        urlsafe = b64.replace(b'+', b'-').replace(b'/', b'_')
        decoded_urlsafe = _safe_b64decode(urlsafe)
        self.assertEqual(decoded_urlsafe[:3], b'\xff\xd8\xff')
        self.assertEqual(Image.open(io.BytesIO(decoded_urlsafe)).size, (20, 20))

    def test_04_safe_b64decode_double_encoded(self):
        """Double-base64 encoded payloads should be unwrapped to raw image bytes."""
        img = Image.new('RGB', (15, 15), color=(50, 100, 150))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        raw_png = buf.getvalue()
        b64_once = base64.b64encode(raw_png)
        b64_twice = base64.b64encode(b64_once)

        decoded = _safe_b64decode(b64_twice)
        self.assertEqual(decoded[:4], b'\x89PNG')
        self.assertEqual(Image.open(io.BytesIO(decoded)).size, (15, 15))

    def test_05_safe_b64decode_raw_bytes_with_comma(self):
        """Raw binary image data containing byte 0x2C (comma) in header must not be split."""
        img = Image.new('RGB', (10, 10), color=(0, 255, 0))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        raw_jpeg = buf.getvalue()

        # Pass raw binary directly
        decoded = _safe_b64decode(raw_jpeg)
        self.assertEqual(decoded, raw_jpeg)
        self.assertEqual(Image.open(io.BytesIO(decoded)).size, (10, 10))

    def test_06_safe_b64decode_memoryview_and_bytearray(self):
        """Memoryview and bytearray objects from ORM or DB driver should be handled."""
        sample = b"binary-data-packet"
        b64 = base64.b64encode(sample)
        mv = memoryview(b64)
        ba = bytearray(b64)

        self.assertEqual(_safe_b64decode(mv), sample)
        self.assertEqual(_safe_b64decode(ba), sample)

    def test_07_safe_b64decode_whitespace_and_quoted(self):
        """Base64 strings wrapped in quotes or containing line breaks should decode cleanly."""
        img = Image.new('RGB', (10, 10), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        raw = buf.getvalue()
        b64_str = base64.b64encode(raw).decode('ascii')
        wrapped = f'  "{b64_str[:50]}\r\n{b64_str[50:]}"  '

        decoded = _safe_b64decode(wrapped)
        self.assertEqual(decoded[:3], b'\xff\xd8\xff')
        self.assertEqual(Image.open(io.BytesIO(decoded)).size, (10, 10))

    def test_08_image_pipeline_unidentified_image_handling(self):
        """ImagePipeline raises descriptive ImagePipelineError for HTML, URLs, and unsupported formats."""
        from models.image_pipeline import ImagePipeline, ImagePipelineError

        # HTML document
        with self.assertRaises(ImagePipelineError) as ctx_html:
            ImagePipeline.process_image(image_bytes=b'<!DOCTYPE html><html>404 Not Found</html>')
        self.assertIn("HTML", str(ctx_html.exception))

        # URL string
        with self.assertRaises(ImagePipelineError) as ctx_url:
            ImagePipeline.process_image(image_bytes=b'https://example.com/products/shoe.jpg')
        self.assertIn("URL", str(ctx_url.exception))

        # SVG without cairosvg installed raises clear explanation
        with self.assertRaises(ImagePipelineError) as ctx_svg:
            ImagePipeline.process_image(image_bytes=b'<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>')
        self.assertTrue("SVG" in str(ctx_svg.exception) or "cairosvg" in str(ctx_svg.exception))


if __name__ == '__main__':
    unittest.main()
