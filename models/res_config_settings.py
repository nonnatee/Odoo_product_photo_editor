# -*- coding: utf-8 -*-
"""
Product Photo Editor - Configuration Settings
Manages API keys, default presets, webhook URLs, and batch cron preferences.
"""

from odoo import models, fields
from .image_pipeline import ImagePipeline


class ResConfigSettings(models.TransientModel):
    _inherit = 'res.config.settings'

    photo_editor_default_provider = fields.Selection(
        ImagePipeline.PROVIDERS,
        string='Default AI Provider',
        default='local',
        config_parameter='product_photo_editor.default_provider',
        help="Select default engine for photo editing. 'Local Engine' works without external API keys.",
    )
    photo_editor_remove_bg_api_key = fields.Char(
        string='Remove.bg API Key',
        config_parameter='product_photo_editor.remove_bg_api_key',
    )
    photo_editor_clipdrop_api_key = fields.Char(
        string='Clipdrop API Key',
        config_parameter='product_photo_editor.clipdrop_api_key',
    )
    photo_editor_photoroom_api_key = fields.Char(
        string='Photoroom API Key',
        config_parameter='product_photo_editor.photoroom_api_key',
    )
    photo_editor_gemini_api_key = fields.Char(
        string='Google Gemini API Key',
        config_parameter='product_photo_editor.gemini_api_key',
        help="API Key for Google Gemini 3.1 Flash Image instruction-based editing, relighting, and generative fill.",
    )
    photo_editor_gemini_model = fields.Selection(
        [
            ('gemini-3.1-flash-image', 'Nano Banana 2 (Gemini 3.1 Flash Image - Recommended)'),
            ('gemini-3.1-flash-lite-image', 'Nano Banana 2 Lite (Gemini 3.1 Flash Lite Image)'),
            ('gemini-3-pro-image', 'Nano Banana Pro (Gemini 3 Pro Image)'),
            ('gemini-3.8-flash', 'Gemini 3.8 Flash'),
        ],
        string='Gemini AI Model',
        default='gemini-3.1-flash-image',
        config_parameter='product_photo_editor.gemini_model',
        help="Default model for AI instruction editing and generative outpainting.",
    )
    photo_editor_backup_original_to_gallery = fields.Boolean(
        string='Archive Raw Photo to Extra Media Gallery',
        default=True,
        config_parameter='product_photo_editor.backup_original_to_gallery',
        help="When enabled, accepting an AI optimization preserves the original raw photo by adding it to the product's extra media images.",
    )

    photo_editor_custom_ai_endpoint_url = fields.Char(
        string='Custom AI Endpoint URL',
        config_parameter='product_photo_editor.custom_ai_endpoint_url',
        help="Full URL for custom AI segmentation inference API (e.g. https://ai.mycompany.com/v1/segment).",
    )
    photo_editor_custom_ai_api_key = fields.Char(
        string='Custom AI API Key / Bearer Token',
        config_parameter='product_photo_editor.custom_ai_api_key',
    )

    photo_editor_default_background_style = fields.Selection(
        ImagePipeline.BACKGROUND_STYLES,
        string='Default Background Style',
        default='white',
        config_parameter='product_photo_editor.default_background_style',
    )
    photo_editor_default_dimensions = fields.Selection(
        [
            ('square_2000', '2000 x 2000 px (Marketplace Standard)'),
            ('square_1000', '1000 x 1000 px (Web Catalog Standard)'),
            ('portrait_4_5', '1600 x 2000 px (Fashion / Instagram 4:5)'),
            ('landscape_16_9', '1920 x 1080 px (Hero Banner 16:9)'),
            ('original', 'Preserve Original Aspect Ratio'),
        ],
        string='Default Dimensions',
        default='square_2000',
        config_parameter='product_photo_editor.default_dimensions',
    )
    photo_editor_auto_apply_cron = fields.Boolean(
        string='Auto-Apply to Product upon Cron Completion',
        default=True,
        config_parameter='product_photo_editor.auto_apply_cron',
        help="Automatically replace product primary image when background cron finishes processing.",
    )
    photo_editor_cron_batch_size = fields.Integer(
        string='Cron Batch Size',
        default=25,
        config_parameter='product_photo_editor.cron_batch_size',
        help="Maximum number of photos processed per scheduled cron cycle.",
    )
