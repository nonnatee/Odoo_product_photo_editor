# -*- coding: utf-8 -*-
"""
Tests for Product Photo Editor Presets, Quick Actions, and Gemini AI Pipeline.
Validates:
1. Preset Model creation, defaults, constraints, and parameter extraction.
2. Gemini 3.1 Flash Image instruction edit via SDK and direct REST fallback.
3. Graceful offline procedural fallback when no API key is configured.
4. Generative canvas expand and generative fill / magic erase.
5. Product template quick action triggers (Studio Minimal, Album w/ Scale, Printed Catalog).
6. Quick Confirmation wizard workflow: Accept & Apply, non-destructive backup, and audit logging.
7. Bulk preset execution from product catalog list view.
"""

import base64
import io
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

from PIL import Image, ImageDraw

# Ensure module root is on python path for standalone test runners
_current_dir = os.path.dirname(os.path.abspath(__file__))
_module_root = os.path.dirname(_current_dir)
if _module_root not in sys.path:
    sys.path.insert(0, _module_root)

from models.image_pipeline import ImagePipeline, ImagePipelineError
from models.photo_editor import _safe_b64decode, ProductPhotoEditor
from models.photo_editor_preset import ProductPhotoEditorPreset
from models.product_template import ProductTemplate
from wizard.photo_editor_quick_confirm import ProductPhotoEditorQuickConfirm


class TestPresetsAndQuickActions(unittest.TestCase):
    """Deep verification of preset models, Gemini pipeline, and quick actions."""

    @classmethod
    def setUpClass(cls):
        # Generate synthetic product image (400x300, blue object on grey background)
        cls.img_w, cls.img_h = 300, 300
        img = Image.new('RGB', (cls.img_w, cls.img_h), (235, 235, 235))
        draw = ImageDraw.Draw(img)
        draw.rectangle([(60, 60), (240, 240)], fill=(30, 90, 200), outline=(10, 40, 120), width=3)
        draw.rectangle([(100, 100), (200, 200)], fill=(255, 255, 255))

        buf = io.BytesIO()
        img.save(buf, format='PNG')
        cls.sample_image_bytes = buf.getvalue()
        cls.sample_image_b64 = base64.b64encode(cls.sample_image_bytes).decode('ascii')

    # -------------------------------------------------------------------------
    # 1. Preset Model & Pipeline Parameter Extraction
    # -------------------------------------------------------------------------
    def test_01_preset_pipeline_params(self):
        """Test preset parameter extraction for ImagePipeline execution."""
        preset = ProductPhotoEditorPreset()
        preset.name = "Studio Minimal"
        preset.code = "studio_minimal"
        preset.prompt_instruction = "Clean commercial studio lighting with grounding shadow"
        preset.ai_mode = "gemini_edit"
        preset.background_style = "studio_soft_shadow"
        preset.custom_bg_color = "#FFFFFF"
        preset.dimensions = "square_2000"
        preset.target_width = 0
        preset.target_height = 0
        preset.padding_percent = 8.0
        preset.export_format = "JPEG"
        preset.export_quality = 90
        preset.apply_perspective = True
        preset.apply_color_correction = True
        preset.apply_auto_white_balance = True
        preset.apply_contrast_enhancement = True
        preset.apply_sharpening = True

        # Mock ensure_one
        preset.ensure_one = lambda: None

        params = ProductPhotoEditorPreset.get_pipeline_params(preset)
        self.assertEqual(params['background_style'], 'studio_soft_shadow')
        self.assertEqual(params['dimensions'], 'square_2000')
        self.assertEqual(params['padding_percent'], 8.0)
        self.assertEqual(params['ai_mode'], 'gemini_edit')
        self.assertIn("Clean commercial studio", params['prompt_instruction'])
        self.assertIsNone(params['target_width'])

    # -------------------------------------------------------------------------
    # 2. Gemini Instruction Edit - Procedural Fallback (No API Key)
    # -------------------------------------------------------------------------
    def test_02_gemini_instruction_edit_procedural_fallback(self):
        """When no API key is provided, gemini_instruction_edit falls back gracefully to OpenCV/PIL."""
        prompts = [
            "Convert to clean studio-minimal e-commerce lighting with grounding shadow",
            "High-end product lookbook presentation with realistic scale perspective",
            "Ultra-sharp print-ready catalog presentation with crisp edge boundaries",
            "Warm golden sunset ambient lighting",
            "Cyberpunk neon aesthetic",
            "Dramatic monochrome black and white",
        ]

        for p in prompts:
            with self.subTest(prompt=p[:30]):
                out_bytes, provider_label = ImagePipeline.gemini_instruction_edit(
                    image_bytes=self.sample_image_bytes,
                    prompt=p,
                    api_key=None,
                )
                self.assertIsNotNone(out_bytes)
                self.assertGreater(len(out_bytes), 500)
                self.assertIn("Local Procedural Engine", provider_label)

                # Verify result is a valid decodable image
                out_img = Image.open(io.BytesIO(out_bytes))
                self.assertEqual(out_img.size, (self.img_w, self.img_h))

    # -------------------------------------------------------------------------
    # 3. Gemini Instruction Edit - Mock SDK Call
    # -------------------------------------------------------------------------
    @patch('models.image_pipeline.genai')
    def test_03_gemini_instruction_edit_mock_sdk(self, mock_genai):
        """Test Gemini SDK execution when google-genai returns valid image candidate."""
        # Create mock response containing a valid 300x300 PNG image
        fake_result_img = Image.new('RGB', (300, 300), (50, 200, 50))
        fake_buf = io.BytesIO()
        fake_result_img.save(fake_buf, format='PNG')
        fake_png_bytes = fake_buf.getvalue()

        # Construct candidate mock structure
        mock_part = MagicMock()
        mock_part.inline_data.data = fake_png_bytes
        del mock_part.as_image  # test inline_data path

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        mock_response = MagicMock()
        mock_response.parts = []
        mock_response.candidates = [mock_candidate]

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        with patch('models.image_pipeline._GENAI_AVAILABLE', True):
            out_bytes, provider_label = ImagePipeline.gemini_instruction_edit(
                image_bytes=self.sample_image_bytes,
                prompt="Studio minimal pure white e-commerce lighting",
                api_key="AIzaSyFakeKeyTest123",
                model="gemini-3.1-flash-image",
            )

        self.assertIsNotNone(out_bytes)
        self.assertIn("google-genai SDK", provider_label)
        mock_client.models.generate_content.assert_called_once()

    # -------------------------------------------------------------------------
    # 4. Gemini Instruction Edit - Direct REST Fallback
    # -------------------------------------------------------------------------
    @patch('models.image_pipeline.requests')
    def test_04_gemini_instruction_edit_mock_rest_fallback(self, mock_requests):
        """When SDK is unavailable or fails, pipeline seamlessly falls back to direct REST HTTP."""
        fake_result_img = Image.new('RGB', (300, 300), (220, 100, 40))
        fake_buf = io.BytesIO()
        fake_result_img.save(fake_buf, format='PNG')
        fake_b64 = base64.b64encode(fake_buf.getvalue()).decode('ascii')

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"inlineData": {"data": fake_b64, "mimeType": "image/png"}}
                        ]
                    }
                }
            ]
        }
        mock_requests.post.return_value = mock_resp

        # Force SDK to raise exception so REST fallback triggers
        with patch('models.image_pipeline._GENAI_AVAILABLE', False):
            out_bytes, provider_label = ImagePipeline.gemini_instruction_edit(
                image_bytes=self.sample_image_bytes,
                prompt="Studio minimal pure white e-commerce lighting",
                api_key="AIzaSyFakeKeyTest123",
                model="gemini-3.1-flash-image",
            )

        self.assertIsNotNone(out_bytes)
        self.assertIn("Direct REST API", provider_label)
        mock_requests.post.assert_called_once()
        self.assertIn("models/gemini-3.1-flash-image:generateContent", mock_requests.post.call_args[0][0])

    # -------------------------------------------------------------------------
    # 5. Generative Expand and Local Inpaint / Magic Erase
    # -------------------------------------------------------------------------
    def test_05_generative_expand_offline(self):
        """Test generative expand with offline OpenCV border reflection fallback."""
        top, bottom, left, right = 40, 40, 50, 50
        out_bytes = ImagePipeline.gemini_expand(
            image_bytes=self.sample_image_bytes,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            prompt="Expand studio background seamlessly",
            api_key=None,
        )
        self.assertIsNotNone(out_bytes)
        out_img = Image.open(io.BytesIO(out_bytes))
        expected_w = self.img_w + left + right
        expected_h = self.img_h + top + bottom
        self.assertEqual(out_img.size, (expected_w, expected_h))

    def test_06_local_inpaint_magic_erase(self):
        """Test local Telea inpainting on a masked region."""
        # Create mask with white rectangle in the center
        mask_img = Image.new('L', (self.img_w, self.img_h), 0)
        draw = ImageDraw.Draw(mask_img)
        draw.rectangle([(120, 120), (180, 180)], fill=255)
        mask_buf = io.BytesIO()
        mask_img.save(mask_buf, format='PNG')

        inpainted_bytes = ImagePipeline.magic_erase(
            image_bytes=self.sample_image_bytes,
            mask_bytes=mask_buf.getvalue(),
            api_key=None,
        )
        self.assertIsNotNone(inpainted_bytes)
        out_img = Image.open(io.BytesIO(inpainted_bytes))
        self.assertEqual(out_img.size, (self.img_w, self.img_h))

    # -------------------------------------------------------------------------
    # 6. Pipeline Process Image with AI Preset & Instructions
    # -------------------------------------------------------------------------
    def test_07_process_image_with_gemini_instruction(self):
        """Test end-to-end ImagePipeline.process_image with prompt instruction."""
        res = ImagePipeline.process_image(
            image_bytes=self.sample_image_bytes,
            prompt_instruction="Studio-minimal e-commerce lighting with grounding shadow",
            ai_mode="gemini_edit",
            background_style="studio_soft_shadow",
            dimensions="square_1000",
            export_format="JPEG",
            export_quality=85,
            service_provider="local",
        )

        self.assertIn('image_bytes', res)
        self.assertEqual(res['format'], 'JPEG')
        self.assertEqual(res['width'], 1000)
        self.assertEqual(res['height'], 1000)
        self.assertTrue(any("Applied AI instruction edit" in s for s in res['steps_applied']))
        self.assertTrue(any("Composed background" in s for s in res['steps_applied']))

    # -------------------------------------------------------------------------
    # 7. Quick Confirmation Wizard & Workflow Logic
    # -------------------------------------------------------------------------
    def test_08_quick_confirm_workflow_apply(self):
        """Test quick confirmation modal Apply & Close logic and non-destructive image preservation."""
        # Mock product.template
        mock_product = MagicMock()
        mock_product.id = 42
        mock_product.name = "Ergonomic Office Chair"
        mock_product.display_name = "Ergonomic Office Chair"
        mock_product.image_1920 = self.sample_image_b64

        # Mock preset
        mock_preset = MagicMock()
        mock_preset.id = 1
        mock_preset.name = "Studio Minimal"
        mock_preset.code = "studio_minimal"
        mock_preset.background_style = "studio_soft_shadow"
        mock_preset.custom_bg_color = "#FFFFFF"
        mock_preset.dimensions = "square_2000"
        mock_preset.target_width = 0
        mock_preset.target_height = 0
        mock_preset.export_format = "JPEG"
        mock_preset.export_quality = 90
        mock_preset.padding_percent = 8.0
        mock_preset.apply_perspective = True
        mock_preset.apply_color_correction = True
        mock_preset.apply_auto_white_balance = True
        mock_preset.apply_contrast_enhancement = True
        mock_preset.apply_sharpening = True
        mock_preset.prompt_instruction = "Clean commercial studio lighting with grounding shadow"
        mock_preset.ai_mode = "gemini_edit"

        # Mock env with product.image and product.photo.editor
        created_records = {}
        def mock_create(model_name):
            def _create_impl(vals):
                rec = MagicMock()
                for k, v in vals.items():
                    setattr(rec, k, v)
                rec.id = 999
                rec.name = vals.get('name', 'REC-001')
                created_records.setdefault(model_name, []).append(vals)
                return rec
            return _create_impl

        mock_env = {
            'product.image': MagicMock(create=mock_create('product.image')),
            'product.photo.editor': MagicMock(create=mock_create('product.photo.editor')),
            'ir.config_parameter': MagicMock(
                sudo=lambda: MagicMock(
                    get_param=lambda key, defval=None: 'True' if key == 'product_photo_editor.backup_original_to_gallery' else defval
                )
            ),
        }

        wizard = ProductPhotoEditorQuickConfirm()
        wizard.product_id = mock_product
        wizard.preset_id = mock_preset
        wizard.preset_code = "studio_minimal"
        wizard.preset_name = "Studio Minimal"
        wizard.image_original = self.sample_image_b64
        fake_preview = base64.b64encode(b"FAKE_OPTIMIZED_IMAGE_BYTES").decode('ascii')
        wizard.image_preview = fake_preview
        wizard.duration_sec = 1.25
        wizard.steps_applied = "Applied CLAHE\nApplied Contact Shadow"
        wizard.env = mock_env
        wizard.ensure_one = lambda: None

        res_action = wizard.action_apply_and_close()

        # 1. Product image_1920 should have been updated with preview
        mock_product.write.assert_called_once_with({'image_1920': fake_preview})

        # 2. Original raw photo should be preserved in product.image
        self.assertIn('product.image', created_records)
        self.assertEqual(len(created_records['product.image']), 1)
        self.assertEqual(created_records['product.image'][0]['product_tmpl_id'], 42)
        self.assertEqual(created_records['product.image'][0]['image_1920'], self.sample_image_b64)

        # 3. Persistent audit job record should be created
        self.assertIn('product.photo.editor', created_records)
        self.assertEqual(len(created_records['product.photo.editor']), 1)
        job = created_records['product.photo.editor'][0]
        self.assertEqual(job['product_id'], 42)
        self.assertEqual(job['status'], 'done')
        self.assertTrue(job['applied_to_product'])
        self.assertEqual(job['image_processed'], fake_preview)
        self.assertEqual(job['transformation_log'], "Applied CLAHE\nApplied Contact Shadow")
        self.assertNotIn('pipeline_log', job, "product.photo.editor schema does not have pipeline_log")
        self.assertEqual(job['preset_id'], 1)
        self.assertEqual(job['ai_mode'], 'gemini_edit')

        # Verify all keys in job exist on ProductPhotoEditor model
        editor_model_fields = [
            'product_id', 'product_variant_id', 'company_id', 'active', 'status',
            'image_original', 'image_original_filename', 'image_processed', 'image_processed_filename',
            'service_provider', 'background_style', 'custom_bg_color', 'dimensions',
            'target_width', 'target_height', 'export_format', 'export_quality',
            'padding_percent', 'apply_perspective', 'apply_color_correction',
            'apply_auto_white_balance', 'apply_contrast_enhancement', 'apply_sharpening',
            'original_width', 'original_height', 'original_file_size',
            'processed_width', 'processed_height', 'processed_file_size',
            'duration_sec', 'error_message', 'transformation_log',
            'applied_to_product', 'applied_date', 'preset_id', 'prompt_instruction', 'ai_mode',
        ]
        for key in job.keys():
            self.assertIn(key, editor_model_fields, f"Key '{key}' is not a valid field on ProductPhotoEditor!")

        # 4. Action returns success notification with parent reload
        self.assertEqual(res_action['type'], 'ir.actions.client')
        self.assertEqual(res_action['params']['type'], 'success')
        self.assertEqual(res_action['params']['next']['tag'], 'reload')

    # -------------------------------------------------------------------------
    # 8. Product Template Quick Action Presets
    # -------------------------------------------------------------------------
    def test_09_product_template_quick_action_missing_image(self):
        """Triggering quick action on a product without an image raises UserError."""
        template = ProductTemplate()
        template.image_1920 = False
        template.display_name = "Empty Product"
        template.ensure_one = lambda: None

        with self.assertRaises(Exception):
            ProductTemplate.action_quick_apply_preset(template, 'studio_minimal')

    def test_10_product_template_quick_action_happy_path(self):
        """Happy path for action_quick_apply_preset: generates in-memory preview and opens quick confirm wizard."""
        # Mock preset
        mock_preset = MagicMock()
        mock_preset.id = 101
        mock_preset.name = "Studio Minimal"
        mock_preset.code = "studio_minimal"
        mock_preset.prompt_instruction = "Clean commercial studio lighting with grounding shadow"
        mock_preset.ai_mode = "gemini_edit"
        mock_preset.background_style = "studio_soft_shadow"
        mock_preset.custom_bg_color = "#FFFFFF"
        mock_preset.dimensions = "square_1000"
        mock_preset.target_width = 0
        mock_preset.target_height = 0
        mock_preset.export_format = "JPEG"
        mock_preset.export_quality = 90
        mock_preset.padding_percent = 8.0
        mock_preset.apply_perspective = True
        mock_preset.apply_color_correction = True
        mock_preset.apply_auto_white_balance = True
        mock_preset.apply_contrast_enhancement = True
        mock_preset.apply_sharpening = True
        mock_preset.get_pipeline_params.return_value = {
            'prompt_instruction': mock_preset.prompt_instruction,
            'ai_mode': mock_preset.ai_mode,
            'background_style': mock_preset.background_style,
            'custom_bg_color': mock_preset.custom_bg_color,
            'dimensions': mock_preset.dimensions,
            'target_width': None,
            'target_height': None,
            'export_format': mock_preset.export_format,
            'export_quality': mock_preset.export_quality,
            'padding_percent': mock_preset.padding_percent,
            'apply_perspective': mock_preset.apply_perspective,
            'apply_color_correction': mock_preset.apply_color_correction,
            'apply_auto_white_balance': mock_preset.apply_auto_white_balance,
            'apply_contrast_enhancement': mock_preset.apply_contrast_enhancement,
            'apply_sharpening': mock_preset.apply_sharpening,
        }

        created_wizards = []
        def mock_wizard_create(vals):
            wiz = MagicMock()
            wiz.id = 555
            for k, v in vals.items():
                setattr(wiz, k, v)
            created_wizards.append(vals)
            return wiz

        mock_env = {
            'product.photo.editor.preset': MagicMock(search=lambda *a, **k: mock_preset),
            'product.photo.editor.quick.confirm': MagicMock(create=mock_wizard_create),
            'ir.config_parameter': MagicMock(
                sudo=lambda: MagicMock(
                    get_param=lambda key, defval=None: 'local' if 'default_provider' in key else ''
                )
            ),
        }

        template = ProductTemplate()
        template.id = 77
        template.name = "Smart Standing Desk"
        template.display_name = "Smart Standing Desk"
        template.image_1920 = self.sample_image_b64
        template.env = mock_env
        template.ensure_one = lambda: None

        # Execute quick action
        act = template.action_quick_apply_preset('studio_minimal')

        self.assertEqual(act['res_model'], 'product.photo.editor.quick.confirm')
        self.assertEqual(act['res_id'], 555)
        self.assertEqual(act['target'], 'new')
        self.assertEqual(len(created_wizards), 1)

        wiz_vals = created_wizards[0]
        self.assertEqual(wiz_vals['product_id'], 77)
        self.assertEqual(wiz_vals['preset_code'], 'studio_minimal')
        self.assertTrue(wiz_vals['image_preview'])
        self.assertGreater(wiz_vals['preview_width'], 0)
        self.assertGreater(wiz_vals['duration_sec'], 0)

    def test_11_product_template_quick_action_convenience_methods(self):
        """Header convenience buttons (Studio Minimal, Album Scale, Printed Catalog) call action_quick_apply_preset."""
        template = ProductTemplate()
        template.action_quick_apply_preset = MagicMock(return_value={'type': 'ir.actions.act_window'})

        template.action_quick_studio_minimal()
        template.action_quick_apply_preset.assert_called_with('studio_minimal')

        template.action_quick_album_scale()
        template.action_quick_apply_preset.assert_called_with('album_scale')

        template.action_quick_printed_catalog()
        template.action_quick_apply_preset.assert_called_with('printed_catalog')

    def test_12_product_template_batch_actions(self):
        """Batch actions enqueue pending jobs with full preset details and prompt instructions."""
        mock_preset = MagicMock()
        mock_preset.id = 202
        mock_preset.name = "Album w/ Scale"
        mock_preset.code = "album_scale"
        mock_preset.prompt_instruction = "High-end lookbook presentation with scale perspective"
        mock_preset.ai_mode = "gemini_edit"
        mock_preset.background_style = "studio_neutral"
        mock_preset.custom_bg_color = "#F4F4F4"
        mock_preset.dimensions = "portrait_4_5"
        mock_preset.target_width = 0
        mock_preset.target_height = 0
        mock_preset.export_format = "JPEG"
        mock_preset.export_quality = 92
        mock_preset.padding_percent = 10.0
        mock_preset.apply_perspective = True
        mock_preset.apply_color_correction = True
        mock_preset.apply_auto_white_balance = True
        mock_preset.apply_contrast_enhancement = True
        mock_preset.apply_sharpening = False

        created_jobs = []
        mock_job_model = MagicMock()
        mock_job_model.create = lambda vals: created_jobs.append(vals) or MagicMock()

        mock_env = {
            'product.photo.editor.preset': MagicMock(search=lambda *a, **k: mock_preset),
            'product.photo.editor': mock_job_model,
        }

        # Create two templates with images, one without
        p1 = MagicMock(id=1, image_1920=self.sample_image_b64, display_name="Chair")
        p2 = MagicMock(id=2, image_1920=self.sample_image_b64, display_name="Table")
        p3 = MagicMock(id=3, image_1920=False, display_name="Ghost")

        template_set = ProductTemplate()
        template_set.env = mock_env
        template_set._records = [p1, p2, p3]

        res = ProductTemplate.action_batch_apply_preset(template_set, 'album_scale')
        self.assertEqual(res['type'], 'ir.actions.client')
        self.assertEqual(len(created_jobs), 2)

        j1 = created_jobs[0]
        self.assertEqual(j1['product_id'], 1)
        self.assertEqual(j1['preset_id'], 202)
        self.assertEqual(j1['prompt_instruction'], "High-end lookbook presentation with scale perspective")
        self.assertEqual(j1['ai_mode'], "gemini_edit")
        self.assertEqual(j1['background_style'], "studio_neutral")
        self.assertEqual(j1['status'], "pending")

    @patch('models.image_pipeline.requests')
    def test_13_gemini_expand_and_fill_rest_fallback(self, mock_requests):
        """gemini_expand and gemini_fill successfully fallback to direct REST HTTP when SDK is unavailable."""
        fake_result_img = Image.new('RGB', (400, 400), (80, 160, 240))
        fake_buf = io.BytesIO()
        fake_result_img.save(fake_buf, format='PNG')
        fake_b64 = base64.b64encode(fake_buf.getvalue()).decode('ascii')

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"inlineData": {"data": fake_b64, "mimeType": "image/png"}}
                        ]
                    }
                }
            ]
        }
        mock_requests.post.return_value = mock_resp

        # Test expand with REST fallback
        with patch('models.image_pipeline._GENAI_AVAILABLE', False):
            expanded_bytes = ImagePipeline.gemini_expand(
                image_bytes=self.sample_image_bytes,
                top=50, bottom=50, left=50, right=50,
                prompt="Expand studio setting",
                api_key="AIzaSyTestRestExpandKey",
            )
            self.assertIsNotNone(expanded_bytes)
            exp_img = Image.open(io.BytesIO(expanded_bytes))
            self.assertEqual(exp_img.size, (self.img_w + 100, self.img_h + 100))

        # Test fill with REST fallback
        mask_img = Image.new('L', (self.img_w, self.img_h), 255)
        mask_buf = io.BytesIO()
        mask_img.save(mask_buf, format='PNG')

        with patch('models.image_pipeline._GENAI_AVAILABLE', False):
            filled_bytes = ImagePipeline.gemini_fill(
                image_bytes=self.sample_image_bytes,
                mask_bytes=mask_buf.getvalue(),
                prompt="Synthesize smooth surface",
                api_key="AIzaSyTestRestFillKey",
            )
            self.assertIsNotNone(filled_bytes)
            fill_img = Image.open(io.BytesIO(filled_bytes))
            self.assertEqual(fill_img.size, (self.img_w, self.img_h))

    def test_14_product_photo_editor_action_process_with_preset(self):
        """ProductPhotoEditor.action_process() forwards preset_id, prompt_instruction, and ai_mode."""
        job = ProductPhotoEditor()
        job.id = 888
        job.name = "PPE-20260927-0001"
        job.product_id = MagicMock(id=10, default_code="PROD-10", name="Coffee Mug")
        job.product_variant_id = False
        job.image_original = self.sample_image_b64
        job.image_processed = False
        job.background_style = "white"
        job.custom_bg_color = "#FFFFFF"
        job.dimensions = "square_1000"
        job.target_width = 0
        job.target_height = 0
        job.export_format = "JPEG"
        job.export_quality = 90
        job.padding_percent = 8.0
        job.apply_perspective = True
        job.apply_color_correction = True
        job.apply_auto_white_balance = True
        job.apply_contrast_enhancement = True
        job.apply_sharpening = True
        job.service_provider = "local"
        job.preset_id = MagicMock(id=5, name="Studio Minimal")
        job.prompt_instruction = "Clean studio minimal lighting"
        job.ai_mode = "gemini_edit"

        written_vals = {}
        job.write = lambda vals: written_vals.update(vals)
        job.ensure_one = lambda: None
        job.env = {
            'ir.config_parameter': MagicMock(
                sudo=lambda: MagicMock(
                    get_param=lambda key, defval='': defval
                )
            )
        }

        fake_pipeline_result = {
            'image_bytes': self.sample_image_bytes,
            'format': 'JPEG',
            'width': 1000,
            'height': 1000,
            'file_size': len(self.sample_image_bytes),
            'duration_sec': 0.85,
            'steps_applied': ['Applied Gray-World AWB', 'Applied CLAHE', 'Contact Shadow'],
        }
        with patch.object(ImagePipeline, 'process_image', return_value=fake_pipeline_result) as spy_process:
            res = job.action_process()
            self.assertEqual(res['type'], 'ir.actions.client')
            self.assertEqual(written_vals['status'], 'done')
            self.assertTrue(written_vals['image_processed'])

            # Verify prompt_instruction and ai_mode were forwarded to pipeline
            call_kwargs = spy_process.call_args[1]
            self.assertEqual(call_kwargs['prompt_instruction'], "Clean studio minimal lighting")
            self.assertEqual(call_kwargs['ai_mode'], "gemini_edit")


if __name__ == '__main__':
    unittest.main()
