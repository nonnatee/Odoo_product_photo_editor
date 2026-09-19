# -*- coding: utf-8 -*-
"""
Product Photo Editor - Controller & REST API Tests
Tests REST endpoints /api/v1/photo_editor/presets and /api/v1/photo_editor/process.
"""

import base64
import io
import json
import unittest
from unittest.mock import patch, MagicMock

from PIL import Image

from controllers.main import ProductPhotoEditorController
from models.image_pipeline import ImagePipeline


class TestPhotoEditorController(unittest.TestCase):

    def setUp(self):
        self.controller = ProductPhotoEditorController()

        # Create test image
        img = Image.new('RGB', (100, 100), (200, 200, 200))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        self.raw_bytes = buf.getvalue()
        self.b64_image = base64.b64encode(self.raw_bytes).decode('utf-8')

    def test_01_api_presets(self):
        """Verify /api/v1/photo_editor/presets structure."""
        resp = self.controller.api_get_presets()
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data.decode('utf-8'))

        self.assertIn('background_styles', data)
        self.assertIn('dimension_presets', data)
        self.assertIn('providers', data)
        self.assertIn('export_formats', data)

        styles = [s['id'] for s in data['background_styles']]
        self.assertIn('white', styles)
        self.assertIn('transparent', styles)
        self.assertIn('studio_soft_shadow', styles)

    @patch('controllers.main.request')
    def test_02_api_process_json(self, mock_request):
        """Verify /api/v1/photo_editor/process with JSON base64 payload."""
        payload = {
            'image': self.b64_image,
            'background_style': 'white',
            'dimensions': 'square_1000',
            'export_format': 'JPEG',
            'service_provider': 'local',
        }
        json_data = json.dumps(payload).encode('utf-8')

        mock_request.httprequest.content_type = 'application/json'
        mock_request.httprequest.get_data.return_value = json_data
        mock_request.httprequest.files = {}
        mock_request.env = MagicMock()
        mock_request.env['ir.config_parameter'].sudo().get_param.return_value = ''

        resp = self.controller.api_process_photo()
        self.assertEqual(resp.status_code, 200)
        result = json.loads(resp.data.decode('utf-8'))

        self.assertTrue(result['success'])
        self.assertEqual(result['status'], 'done')
        self.assertEqual(result['width'], 1000)
        self.assertEqual(result['height'], 1000)
        self.assertIn('image_processed', result)

    @patch('controllers.main.request')
    def test_03_api_process_missing_image(self, mock_request):
        """Verify 400 error when image parameter is missing."""
        mock_request.httprequest.content_type = 'application/json'
        mock_request.httprequest.get_data.return_value = b'{}'
        mock_request.httprequest.files = {}

        resp = self.controller.api_process_photo()
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.data.decode('utf-8'))
        self.assertIn('error', data)

    @patch('controllers.main.request')
    def test_04_api_process_data_uri(self, mock_request):
        """Verify processing succeeds when image is supplied as a Data URL string."""
        data_uri = f"data:image/png;base64,{self.b64_image}"
        payload = {
            'image': data_uri,
            'background_style': 'white',
            'dimensions': 'square_1000',
            'service_provider': 'local',
        }
        json_data = json.dumps(payload).encode('utf-8')

        mock_request.httprequest.content_type = 'application/json'
        mock_request.httprequest.get_data.return_value = json_data
        mock_request.httprequest.files = {}
        mock_request.env = MagicMock()
        mock_request.env['ir.config_parameter'].sudo().get_param.return_value = ''

        resp = self.controller.api_process_photo()
        self.assertEqual(resp.status_code, 200)
        result = json.loads(resp.data.decode('utf-8'))
        self.assertTrue(result['success'])
        self.assertEqual(result['width'], 1000)


if __name__ == '__main__':
    unittest.main()
