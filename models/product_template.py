# -*- coding: utf-8 -*-
"""
Product Photo Editor - Product Template Extension
Adds one-click photo editing, job count smart button, and batch optimization actions.
"""

import base64
import io
from PIL import Image

from odoo import models, fields, api, _
from odoo.exceptions import UserError


class ProductTemplate(models.Model):
    _inherit = 'product.template'

    photo_editor_ids = fields.One2many(
        'product.photo.editor',
        'product_id',
        string='Photo Editor Jobs',
    )
    photo_editor_count = fields.Integer(
        string='Photo Edits Count',
        compute='_compute_photo_editor_count',
    )

    @api.depends('photo_editor_ids')
    def _compute_photo_editor_count(self):
        job_data = self.env['product.photo.editor'].sudo()._read_group(
            domain=[('product_id', 'in', self.ids)],
            groupby=['product_id'],
            aggregates=['__count'],
        )
        counts = {product.id: count for product, count in job_data}
        for template in self:
            template.photo_editor_count = counts.get(template.id, 0)

    def action_open_photo_editor_wizard(self):
        """Opens interactive editing wizard prefilled with this product's primary image."""
        self.ensure_one()
        context = {
            'default_product_id': self.id,
        }
        if self.image_1920:
            context['default_image_original'] = self.image_1920

        # Read default preferences
        ICP = self.env['ir.config_parameter'].sudo()
        def_provider = ICP.get_param('product_photo_editor.default_provider', 'local')
        def_bg = ICP.get_param('product_photo_editor.default_background_style', 'white')
        def_dim = ICP.get_param('product_photo_editor.default_dimensions', 'square_2000')

        context.update({
            'default_service_provider': def_provider,
            'default_background_style': def_bg,
            'default_dimensions': def_dim,
        })

        return {
            'name': _('AI Product Photo Editor'),
            'type': 'ir.actions.act_window',
            'res_model': 'product.photo.editor.wizard',
            'view_mode': 'form',
            'target': 'new',
            'context': context,
        }

    def action_view_photo_editor_jobs(self):
        """Smart button action: Displays all photo editing jobs for this product."""
        self.ensure_one()
        action = None
        for xml_id in ('product_editor.action_product_photo_editor', 'product_photo_editor.action_product_photo_editor'):
            try:
                action = self.env['ir.actions.act_window']._for_xml_id(xml_id)
                break
            except Exception:
                continue
        if not action:
            act_rec = self.env['ir.actions.act_window'].search([('res_model', '=', 'product.photo.editor')], limit=1)
            if act_rec:
                action = {
                    'name': act_rec.name,
                    'type': act_rec.type,
                    'res_model': act_rec.res_model,
                    'view_mode': act_rec.view_mode,
                    'domain': act_rec.domain,
                    'context': act_rec.context,
                    'id': act_rec.id,
                }
            else:
                action = {
                    'name': _('Photo Editor Jobs'),
                    'type': 'ir.actions.act_window',
                    'res_model': 'product.photo.editor',
                    'view_mode': 'list,kanban,form',
                }
        action['domain'] = [('product_id', '=', self.id)]
        action['context'] = {'default_product_id': self.id}
        return action

    def action_batch_photo_editor(self):
        """
        Mass action: Enqueues pending photo editing jobs for selected products
        that have an image_1920.
        """
        ICP = self.env['ir.config_parameter'].sudo()
        def_provider = ICP.get_param('product_photo_editor.default_provider', 'local')
        def_bg = ICP.get_param('product_photo_editor.default_background_style', 'white')
        def_dim = ICP.get_param('product_photo_editor.default_dimensions', 'square_2000')

        created_jobs = self.env['product.photo.editor']
        skipped_products = []
        created_job_count = 0

        for product in self:
            if not product.image_1920:
                skipped_products.append(product.display_name)
                continue

            job = self.env['product.photo.editor'].create({
                'product_id': product.id,
                'image_original': product.image_1920,
                'service_provider': def_provider,
                'background_style': def_bg,
                'dimensions': def_dim,
                'status': 'pending',
            })
            created_jobs |= job
            created_job_count += 1

        message = _("Enqueued %d photo optimization jobs for background batch processing.") % created_job_count
        if skipped_products:
            message += _(" Skipped %d products with no image.") % len(skipped_products)

        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': _('Bulk Photo Optimization Queued'),
                'message': message,
                'type': 'info',
                'sticky': False,
            }
        }

    # ── Quick Action Modal & Preset Triggers ───────────────────────────────────
    def action_quick_apply_preset(self, preset_code=None):
        """
        Executes instant AI optimization in-memory and launches the Quick Confirmation modal
        comparing Before vs After.
        """
        self.ensure_one()
        if not self.image_1920:
            raise UserError(_("Please upload a product photo to '%s' before running AI optimization.") % self.display_name)

        preset_code = preset_code or self.env.context.get('preset_code') or 'studio_minimal'
        preset = self.env['product.photo.editor.preset'].search([('code', '=', preset_code)], limit=1)
        if not preset:
            raise UserError(_("Photo Editor preset '%s' not found. Please verify preset configuration.") % preset_code)

        from .photo_editor import _safe_b64decode
        from .image_pipeline import ImagePipeline

        raw_bytes = _safe_b64decode(self.image_1920)
        orig_len = len(raw_bytes)

        ICP = self.env['ir.config_parameter'].sudo()
        gemini_key = ICP.get_param('product_photo_editor.gemini_api_key', '')
        gemini_model = ICP.get_param('product_photo_editor.gemini_model', 'gemini-3.1-flash-image')
        def_provider = ICP.get_param('product_photo_editor.default_provider', 'local')
        provider_config = {
            'remove_bg_api_key': ICP.get_param('product_photo_editor.remove_bg_api_key', ''),
            'clipdrop_api_key': ICP.get_param('product_photo_editor.clipdrop_api_key', ''),
            'photoroom_api_key': ICP.get_param('product_photo_editor.photoroom_api_key', ''),
            'custom_ai_endpoint_url': ICP.get_param('product_photo_editor.custom_ai_endpoint_url', ''),
            'custom_ai_api_key': ICP.get_param('product_photo_editor.custom_ai_api_key', ''),
            'gemini_api_key': gemini_key,
            'gemini_model': gemini_model,
        }

        params = preset.get_pipeline_params()
        params.update({
            'image_bytes': raw_bytes,
            'service_provider': def_provider,
            'provider_config': provider_config,
            'gemini_api_key': gemini_key,
            'gemini_model': gemini_model,
        })

        res = ImagePipeline.process_image(**params)

        # Inspect dimensions
        try:
            orig_pil = Image.open(io.BytesIO(raw_bytes))
            orig_w, orig_h = orig_pil.size
        except Exception:
            orig_w, orig_h = (0, 0)

        def _format_size(size_bytes):
            if not size_bytes:
                return "0 B"
            s = float(size_bytes)
            for unit in ['B', 'KB', 'MB']:
                if s < 1024.0:
                    return f"{s:.1f} {unit}"
                s /= 1024.0
            return f"{s:.1f} GB"

        preview_b64 = base64.b64encode(res['image_bytes'])
        steps_text = "\n".join(f"• {s}" for s in res.get('steps_applied', []))

        confirm_wizard = self.env['product.photo.editor.quick.confirm'].create({
            'product_id': self.id,
            'preset_id': preset.id,
            'preset_code': preset.code,
            'preset_name': preset.name,
            'image_original': self.image_1920,
            'image_preview': preview_b64,
            'original_width': orig_w,
            'original_height': orig_h,
            'original_file_size': _format_size(orig_len),
            'preview_width': res.get('width', 0),
            'preview_height': res.get('height', 0),
            'preview_file_size': _format_size(res.get('file_size', 0)),
            'duration_sec': res.get('duration_sec', 0.0),
            'steps_applied': steps_text,
        })

        return {
            'name': _("Review AI Optimization - %s") % preset.name,
            'type': 'ir.actions.act_window',
            'res_model': 'product.photo.editor.quick.confirm',
            'res_id': confirm_wizard.id,
            'view_mode': 'form',
            'target': 'new',
        }

    def action_quick_studio_minimal(self):
        """Header Quick Action: Studio Minimal Preset."""
        return self.action_quick_apply_preset('studio_minimal')

    def action_quick_album_scale(self):
        """Header Quick Action: Album w/ Scale Preset."""
        return self.action_quick_apply_preset('album_scale')

    def action_quick_printed_catalog(self):
        """Header Quick Action: Printed Catalog Preset."""
        return self.action_quick_apply_preset('printed_catalog')

    # ── Bulk Preset Execution (List View Multi-Record) ────────────────────────
    def action_batch_apply_preset(self, preset_code=None):
        """Mass action: Enqueues pending batch jobs for selected products using requested preset."""
        preset_code = preset_code or self.env.context.get('preset_code') or 'studio_minimal'
        preset = self.env['product.photo.editor.preset'].search([('code', '=', preset_code)], limit=1)

        created_jobs = self.env['product.photo.editor']
        skipped_products = []
        created_job_count = 0

        for product in self:
            if not product.image_1920:
                skipped_products.append(product.display_name)
                continue

            vals = {
                'product_id': product.id,
                'image_original': product.image_1920,
                'status': 'pending',
                'service_provider': 'local',
            }
            if preset:
                vals.update({
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
            job = self.env['product.photo.editor'].create(vals)
            created_jobs |= job
            created_job_count += 1

        preset_name = preset.name if preset else preset_code
        message = _("Enqueued %d photo optimization jobs with '%s' preset.") % (created_job_count, preset_name)
        if skipped_products:
            message += _(" Skipped %d products without photos.") % len(skipped_products)

        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': _("Bulk Preset Queued"),
                'message': message,
                'type': 'info',
                'sticky': False,
            }
        }

    def action_batch_studio_minimal(self):
        return self.action_batch_apply_preset('studio_minimal')

    def action_batch_album_scale(self):
        return self.action_batch_apply_preset('album_scale')

    def action_batch_printed_catalog(self):
        return self.action_batch_apply_preset('printed_catalog')

    def action_batch_cutout_transparent(self):
        return self.action_batch_apply_preset('cutout_transparent')

