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
