# -*- coding: utf-8 -*-
"""
Product Photo Editor - Preset Configuration Model
Allows administrators to define, customize, and order AI image processing profiles,
including Gemini 3.1 Flash Image prompts, background styling, dimensions, and quick action bindings.
"""

from odoo import models, fields, api, _


class ProductPhotoEditorPreset(models.Model):
    _name = 'product.photo.editor.preset'
    _description = 'Product Photo Editor Preset Profile'
    _order = 'sequence asc, id asc'

    name = fields.Char(
        string='Preset Name',
        required=True,
        translate=True,
    )
    code = fields.Char(
        string='Identifier Code',
        required=True,
        index=True,
        help="Unique technical key used by quick action buttons and API endpoints.",
    )
    sequence = fields.Integer(
        string='Sequence',
        default=10,
        help="Display ordering in quick action menus and wizard lists.",
    )
    active = fields.Boolean(
        string='Active',
        default=True,
    )
    description = fields.Text(
        string='Description',
        translate=True,
        help="Summary of the preset workflow and target marketplace compliance.",
    )
    show_in_quick_actions = fields.Boolean(
        string='Show in Quick Actions',
        default=True,
        help="When enabled, appears as a one-click button in the product template form header.",
    )

    # ── AI Pipeline & Instruction Prompt ──────────────────────────────────────
    ai_mode = fields.Selection([
        ('gemini_edit', 'Gemini 3.1 Flash Image (AI Instruct & Relight)'),
        ('cutout_only', 'Foreground Cutout Only (Local/rembg)'),
        ('expand', 'Generative Canvas Expand'),
        ('opencv_only', 'Procedural Computer Vision (Offline)'),
    ], string='AI Pipeline Mode', default='gemini_edit', required=True)

    gemini_model = fields.Selection([
        ('gemini-3.1-flash-image', 'Gemini 3.1 Flash Image (Recommended)'),
        ('gemini-3.1-flash-lite-image', 'Gemini 3.1 Flash Lite Image'),
        ('gemini-3-pro-image', 'Gemini 3 Pro Image'),
        ('gemini-3.8-flash', 'Gemini 3.8 Flash'),
    ], string='Gemini Model', default='gemini-3.1-flash-image')

    prompt_instruction = fields.Text(
        string='Gemini 3.1 Flash Image Prompt Instruction',
        help="Natural language prompt instruction passed to Google Gemini 3.1 Flash Image for contextual relighting, cleaning, or styling.",
    )

    # ── Canvas, Background & Geometry ─────────────────────────────────────────
    background_style = fields.Selection([
        ('white', 'Pure White (#FFFFFF, Studio Standard)'),
        ('studio_neutral', 'Studio Neutral Light Grey (#F4F4F4)'),
        ('studio_soft_shadow', 'Studio White with Grounding Shadow'),
        ('transparent', 'Transparent PNG'),
        ('lifestyle_wood', 'Lifestyle Warm Wood Surface'),
        ('lifestyle_marble', 'Lifestyle White Marble Countertop'),
        ('lifestyle_gradient', 'Modern Clean Studio Gradient'),
        ('custom_color', 'Custom Hex Color'),
    ], string='Background Style', default='white', required=True)

    custom_bg_color = fields.Char(
        string='Custom Color Hex',
        default='#FFFFFF',
        help="Hex color string (e.g. #F0F4F8) when background style is set to Custom Color.",
    )

    dimensions = fields.Selection([
        ('square_2000', 'Square 2000x2000 (Marketplace Amazon/Shopify Standard)'),
        ('square_1000', 'Square 1000x1000 (Standard Web Catalog)'),
        ('portrait_4_5', 'Portrait 1600x2000 (Fashion / 4:5)'),
        ('landscape_16_9', 'Landscape 1920x1080 (Hero Banner / 16:9)'),
        ('original', 'Preserve Original Dimensions'),
    ], string='Dimensions', default='square_2000', required=True)

    target_width = fields.Integer(string='Custom Target Width (px)')
    target_height = fields.Integer(string='Custom Target Height (px)')

    padding_percent = fields.Float(
        string='Subject Padding (%)',
        default=8.0,
        help="Safe margins around the subject relative to total canvas dimension (0-40%).",
    )

    # ── Enhancements & Quality ────────────────────────────────────────────────
    export_format = fields.Selection([
        ('JPEG', 'JPEG (Standard)'),
        ('PNG', 'PNG (Lossless / Transparent)'),
        ('WEBP', 'WebP (Modern Web Standard)'),
    ], string='Export Format', default='JPEG', required=True)

    export_quality = fields.Integer(
        string='Export Quality (%)',
        default=90,
    )

    apply_perspective = fields.Boolean(
        string='Perspective / Deskew',
        default=True,
        help="Automatically detect tilted quadrilateral boxes and correct perspective distortion.",
    )

    apply_color_correction = fields.Boolean(
        string='Color & Lighting Correction',
        default=True,
    )

    apply_auto_white_balance = fields.Boolean(
        string='Gray-World White Balance',
        default=True,
    )

    apply_contrast_enhancement = fields.Boolean(
        string='CLAHE Adaptive Exposure',
        default=True,
    )

    apply_sharpening = fields.Boolean(
        string='Unsharp Mask Sharpening',
        default=True,
    )

    _sql_constraints = [
        ('code_uniq', 'unique(code)', 'Preset code identifier must be unique!'),
    ]

    def get_pipeline_params(self):
        """Extract a clean dictionary of pipeline arguments for ImagePipeline."""
        self.ensure_one()
        return {
            'prompt_instruction': self.prompt_instruction or '',
            'ai_mode': self.ai_mode,
            'gemini_model': self.gemini_model or 'gemini-3.1-flash-image',
            'background_style': self.background_style,
            'custom_bg_color': self.custom_bg_color or '#FFFFFF',
            'dimensions': self.dimensions,
            'target_width': self.target_width if self.dimensions == 'original' and self.target_width else None,
            'target_height': self.target_height if self.dimensions == 'original' and self.target_height else None,
            'export_format': self.export_format,
            'export_quality': self.export_quality,
            'apply_perspective': self.apply_perspective,
            'apply_color_correction': self.apply_color_correction,
            'apply_auto_white_balance': self.apply_auto_white_balance,
            'apply_contrast_enhancement': self.apply_contrast_enhancement,
            'apply_sharpening': self.apply_sharpening,
            'padding_percent': self.padding_percent,
        }
