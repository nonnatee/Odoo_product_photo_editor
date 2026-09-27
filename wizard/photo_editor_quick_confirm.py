# -*- coding: utf-8 -*-
"""
Product Photo Editor - Quick Confirmation Modal
Provides a streamlined Before/After comparison modal with instant Accept & Apply
for one-click quick actions from product templates.
"""

import base64
import io
import logging

from PIL import Image

from odoo import models, fields, api, _
from odoo.exceptions import UserError
try:
    from ..models.image_pipeline import ImagePipeline
except (ImportError, ValueError):
    from models.image_pipeline import ImagePipeline

_logger = logging.getLogger(__name__)


def _format_size(size_bytes):
    if not size_bytes:
        return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


class ProductPhotoEditorQuickConfirm(models.TransientModel):
    _name = 'product.photo.editor.quick.confirm'
    _description = 'AI Photo Quick Action Confirmation'

    product_id = fields.Many2one(
        'product.template',
        string='Product',
        required=True,
        readonly=True,
    )
    preset_id = fields.Many2one(
        'product.photo.editor.preset',
        string='Preset Applied',
        readonly=True,
    )
    preset_code = fields.Char(
        string='Preset Code',
        readonly=True,
    )
    preset_name = fields.Char(
        string='Preset Name',
        readonly=True,
    )

    image_original = fields.Image(
        string='Before (Raw Photo)',
        readonly=True,
    )
    image_preview = fields.Image(
        string='After (AI Optimized)',
        readonly=True,
    )

    original_width = fields.Integer(string='Original Width', readonly=True)
    original_height = fields.Integer(string='Original Height', readonly=True)
    original_file_size = fields.Char(string='Original Size', readonly=True)

    preview_width = fields.Integer(string='Optimized Width', readonly=True)
    preview_height = fields.Integer(string='Optimized Height', readonly=True)
    preview_file_size = fields.Char(string='Optimized Size', readonly=True)

    duration_sec = fields.Float(string='Processing Time (s)', digits=(6, 2), readonly=True)
    steps_applied = fields.Text(string='Pipeline Steps Applied', readonly=True)

    def action_apply_and_close(self):
        """
        Applies optimized image to product template image_1920,
        archives original raw photo into product.image (if configured),
        and creates an audit record in product.photo.editor.
        """
        self.ensure_one()
        product = self.product_id
        if not self.image_preview:
            raise UserError(_("No optimized image is available to apply."))

        ICP = self.env['ir.config_parameter'].sudo()
        backup_gallery = ICP.get_param('product_photo_editor.backup_original_to_gallery', 'True') in ('True', 'true', '1')

        # 1. Non-destructive backup to product.image (extra media gallery)
        if backup_gallery and self.image_original and 'product.image' in self.env:
            try:
                self.env['product.image'].create({
                    'name': _("Original Photo - %s") % (product.name or ""),
                    'product_tmpl_id': product.id,
                    'image_1920': self.image_original,
                })
            except Exception as e:
                _logger.warning("Could not archive original image to gallery: %s", e)

        # 2. Update primary product catalog image
        product.write({
            'image_1920': self.image_preview,
        })

        # 3. Create persistent audit job record
        preset = self.preset_id
        job_vals = {
            'product_id': product.id,
            'image_original': self.image_original,
            'image_processed': self.image_preview,
            'status': 'done',
            'applied_to_product': True,
            'applied_date': fields.Datetime.now() if hasattr(fields.Datetime, 'now') else False,
            'duration_sec': self.duration_sec,
            'transformation_log': self.steps_applied,
            'service_provider': 'local',
        }
        if preset:
            job_vals.update({
                'preset_id': preset.id,
                'prompt_instruction': preset.prompt_instruction,
                'ai_mode': preset.ai_mode,
                'background_style': preset.background_style,
                'custom_bg_color': preset.custom_bg_color,
                'dimensions': preset.dimensions,
                'target_width': preset.target_width,
                'target_height': preset.target_height,
                'export_format': preset.export_format,
                'export_quality': preset.export_quality,
                'padding_percent': preset.padding_percent,
                'apply_perspective': preset.apply_perspective,
                'apply_color_correction': preset.apply_color_correction,
                'apply_auto_white_balance': preset.apply_auto_white_balance,
                'apply_contrast_enhancement': preset.apply_contrast_enhancement,
                'apply_sharpening': preset.apply_sharpening,
            })
        self.env['product.photo.editor'].create(job_vals)

        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': _("Photo Updated Successfully"),
                'message': _("Applied '%(preset)s' optimization to %(product)s.") % {
                    'preset': self.preset_name or _("AI Preset"),
                    'product': product.display_name,
                },
                'type': 'success',
                'sticky': False,
                'next': {'type': 'ir.actions.client', 'tag': 'reload'},
            }
        }

    def action_open_full_wizard(self):
        """Switches to the full photo editor wizard for fine manual adjustments."""
        self.ensure_one()
        context = {
            'default_product_id': self.product_id.id,
            'default_image_original': self.image_original,
            'default_preview_image': self.image_preview,
        }
        if self.preset_id:
            context.update({
                'default_preset_id': self.preset_id.id,
                'default_background_style': self.preset_id.background_style,
                'default_dimensions': self.preset_id.dimensions,
                'default_export_format': self.preset_id.export_format,
                'default_padding_percent': self.preset_id.padding_percent,
            })
        return {
            'name': _("AI Product Photo Editor"),
            'type': 'ir.actions.act_window',
            'res_model': 'product.photo.editor.wizard',
            'view_mode': 'form',
            'target': 'new',
            'context': context,
        }
