# -*- coding: utf-8 -*-
"""
Product Photo Editor - Interactive Wizard
Enables one-click raw photo upload, preset selection, instant side-by-side preview,
and direct application to product e-commerce catalog.
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



def _safe_b64decode(data):
    """Convert a Binary field value to raw image bytes.

    Handles three cases produced by Odoo 19 Binary / Image fields:
      1. Raw bytes  — Odoo ORM returns decoded bytes directly.
      2. Base64 bytes/str — legacy or RPC-path values.
      3. Data-URI  — browser-side uploads ("data:image/png;base64,...").

    Strategy: try PIL.Image.open() on the data as-is first.  PIL will
    succeed immediately if the data is already raw binary.  Only if that
    fails do we attempt a base64 decode and retry.
    """
    if not data:
        return b''

    # ── Normalise to bytes ────────────────────────────────────────────────────
    if isinstance(data, str):
        # Strip data-URI prefix ("data:image/png;base64,…")
        if ',' in data:
            data = data.split(',', 1)[1]
        try:
            raw = data.strip().encode('ascii')
        except UnicodeEncodeError:
            raw = data.strip().encode('latin-1')
    elif isinstance(data, bytes):
        if b',' in data[:64]:  # data-URI in bytes form
            data = data.split(b',', 1)[1]
        raw = data.strip()
    else:
        return b''

    # ── Pass 1: try to open directly (handles raw-binary Odoo 19 ORM values) ─
    try:
        _test = Image.open(io.BytesIO(raw))
        _test.verify()  # lightweight format check without full decode
        return raw
    except Exception:
        pass

    # ── Pass 2: assume base64-encoded; decode then verify ────────────────────
    try:
        decoded = base64.b64decode(raw)
        _test = Image.open(io.BytesIO(decoded))
        _test.verify()
        return decoded
    except Exception:
        pass

    # ── Pass 3: return raw and let the pipeline surface a clean error ─────────
    return raw


class ProductPhotoEditorWizard(models.TransientModel):
    _name = 'product.photo.editor.wizard'
    _description = 'Interactive Product Photo Editor Wizard'

    @api.model
    def default_get(self, fields_list):
        res = super().default_get(fields_list)
        job_id = self.env.context.get('default_existing_job_id')
        if job_id:
            job = self.env['product.photo.editor'].browse(job_id)
            if job.exists():
                res.setdefault('product_id', job.product_id.id)
                res.setdefault('product_variant_id', job.product_variant_id.id if job.product_variant_id else False)
                res.setdefault('image_original', job.image_original)
                res.setdefault('service_provider', job.service_provider)
                res.setdefault('background_style', job.background_style)
                res.setdefault('custom_bg_color', job.custom_bg_color)
                res.setdefault('dimensions', job.dimensions)
                res.setdefault('target_width', job.target_width)
                res.setdefault('target_height', job.target_height)
                res.setdefault('export_format', job.export_format)
                res.setdefault('export_quality', job.export_quality)
                res.setdefault('padding_percent', job.padding_percent)
                res.setdefault('apply_perspective', job.apply_perspective)
                res.setdefault('apply_color_correction', job.apply_color_correction)
                res.setdefault('apply_auto_white_balance', job.apply_auto_white_balance)
                res.setdefault('apply_contrast_enhancement', job.apply_contrast_enhancement)
                res.setdefault('apply_sharpening', job.apply_sharpening)
                if job.image_processed:
                    res.setdefault('preview_image', job.image_processed)
        product_id = res.get('product_id') or self.env.context.get('default_product_id')
        if product_id and not res.get('image_original'):
            prod = self.env['product.template'].browse(product_id)
            if prod.exists() and prod.image_1920:
                res['image_original'] = prod.image_1920
        return res

    product_id = fields.Many2one(
        'product.template',
        string='Product',
        required=True,
    )
    product_variant_id = fields.Many2one(
        'product.product',
        string='Product Variant',
        domain="[('product_tmpl_id', '=', product_id)]",
    )
    existing_job_id = fields.Many2one(
        'product.photo.editor',
        string='Existing Job',
    )

    # Input Image
    image_original = fields.Binary(
        string='Raw Photo',
        required=True,
        help="Upload raw product photo to optimize.",
    )
    image_original_filename = fields.Char(
        string='Original Filename',
        default='product_raw.png',
    )

    # Editing Options
    service_provider = fields.Selection(
        ImagePipeline.PROVIDERS,
        string='AI Engine',
        default='local',
        required=True,
    )
    background_style = fields.Selection(
        ImagePipeline.BACKGROUND_STYLES,
        string='Background Preset',
        default='white',
        required=True,
    )
    custom_bg_color = fields.Char(
        string='Custom Color',
        default='#FFFFFF',
    )
    dimensions = fields.Selection(
        [
            ('square_2000', '2000 x 2000 px (Marketplace Standard)'),
            ('square_1000', '1000 x 1000 px (Web Catalog Standard)'),
            ('portrait_4_5', '1600 x 2000 px (Fashion / Instagram 4:5)'),
            ('landscape_16_9', '1920 x 1080 px (Hero Banner 16:9)'),
            ('original', 'Preserve Original Aspect Ratio'),
        ],
        string='Dimensions',
        default='square_2000',
        required=True,
    )
    target_width = fields.Integer(string='Custom Width')
    target_height = fields.Integer(string='Custom Height')

    export_format = fields.Selection(
        [
            ('JPEG', 'JPEG (Optimized E-commerce)'),
            ('PNG', 'PNG (Lossless / Transparent)'),
            ('WEBP', 'WebP (Modern Web Compressed)'),
        ],
        string='Format',
        default='JPEG',
        required=True,
    )
    export_quality = fields.Integer(string='Quality', default=90)
    padding_percent = fields.Float(string='Padding (%)', default=8.0)

    # Transformation Toggles
    apply_perspective = fields.Boolean(
        string='Auto Deskew & Perspective',
        default=True,
    )
    apply_color_correction = fields.Boolean(
        string='Color & Lighting Enhancement',
        default=True,
    )
    apply_auto_white_balance = fields.Boolean(
        string='Auto White Balance',
        default=True,
    )
    apply_contrast_enhancement = fields.Boolean(
        string='CLAHE Dynamic Exposure',
        default=True,
    )
    apply_sharpening = fields.Boolean(
        string='Crisp Sharpening',
        default=True,
    )

    # Live Preview Pane
    preview_image = fields.Binary(
        string='Preview Result',
        readonly=True,
    )
    preview_info = fields.Char(
        string='Execution Info',
        readonly=True,
    )

    # -------------------------------------------------------------------------
    # Wizard Actions
    # -------------------------------------------------------------------------
    def action_generate_preview(self):
        """Processes image on-the-fly and displays preview in dialog."""
        self.ensure_one()
        if not self.image_original:
            raise UserError(_("Please upload an image to preview."))

        raw_bytes = _safe_b64decode(self.image_original)
        provider_config = self._get_provider_config()

        try:
            result = ImagePipeline.process_image(
                image_bytes=raw_bytes,
                background_style=self.background_style,
                custom_bg_color=self.custom_bg_color or '#FFFFFF',
                dimensions=self.dimensions,
                target_width=self.target_width if self.target_width > 0 else None,
                target_height=self.target_height if self.target_height > 0 else None,
                export_format=self.export_format,
                export_quality=self.export_quality,
                apply_perspective=self.apply_perspective,
                apply_color_correction=self.apply_color_correction,
                apply_auto_white_balance=self.apply_auto_white_balance,
                apply_contrast_enhancement=self.apply_contrast_enhancement,
                apply_sharpening=self.apply_sharpening,
                padding_percent=self.padding_percent,
                service_provider=self.service_provider,
                provider_config=provider_config,
            )

            processed_b64 = base64.b64encode(result['image_bytes'])
            self.write({
                'preview_image': processed_b64,
                'preview_info': _('Rendered: %sx%s %s in %ss (%.1f KB)') % (
                    result['width'],
                    result['height'],
                    result['format'],
                    result['duration_sec'],
                    result['file_size'] / 1024.0,
                ),
            })

            # Return dialog reload action
            return {
                'name': _('AI Product Photo Editor'),
                'type': 'ir.actions.act_window',
                'res_model': self._name,
                'res_id': self.id,
                'view_mode': 'form',
                'target': 'new',
            }

        except Exception as e:
            _logger.exception("Wizard preview generation failed: %s", e)
            raise UserError(_("Failed to render preview: %s") % str(e))

    def action_apply_and_close(self):
        """Applies processed image to product and records history record."""
        self.ensure_one()
        if not self.image_original:
            raise UserError(_("Please upload an image first."))

        # If preview not yet generated or image changed, generate it
        processed_b64 = self.preview_image
        duration = 0.0
        steps = []
        width = 0
        height = 0
        file_size = 0

        if not processed_b64:
            raw_bytes = _safe_b64decode(self.image_original)
            provider_config = self._get_provider_config()
            res = ImagePipeline.process_image(
                image_bytes=raw_bytes,
                background_style=self.background_style,
                custom_bg_color=self.custom_bg_color or '#FFFFFF',
                dimensions=self.dimensions,
                target_width=self.target_width if self.target_width > 0 else None,
                target_height=self.target_height if self.target_height > 0 else None,
                export_format=self.export_format,
                export_quality=self.export_quality,
                apply_perspective=self.apply_perspective,
                apply_color_correction=self.apply_color_correction,
                apply_auto_white_balance=self.apply_auto_white_balance,
                apply_contrast_enhancement=self.apply_contrast_enhancement,
                apply_sharpening=self.apply_sharpening,
                padding_percent=self.padding_percent,
                service_provider=self.service_provider,
                provider_config=provider_config,
            )
            processed_b64 = base64.b64encode(res['image_bytes'])
            duration = res['duration_sec']
            steps = res['steps_applied']
            width = res['width']
            height = res['height']
            file_size = res['file_size']

        # Update product template primary image
        self.product_id.write({'image_1920': processed_b64})
        if self.product_variant_id:
            self.product_variant_id.write({'image_1920': processed_b64})

        # Create or update history record
        job_vals = {
            'product_id': self.product_id.id,
            'product_variant_id': self.product_variant_id.id if self.product_variant_id else False,
            'image_original': self.image_original,
            'image_processed': processed_b64,
            'status': 'done',
            'background_style': self.background_style,
            'custom_bg_color': self.custom_bg_color,
            'dimensions': self.dimensions,
            'export_format': self.export_format,
            'export_quality': self.export_quality,
            'padding_percent': self.padding_percent,
            'service_provider': self.service_provider,
            'apply_perspective': self.apply_perspective,
            'apply_color_correction': self.apply_color_correction,
            'apply_auto_white_balance': self.apply_auto_white_balance,
            'apply_contrast_enhancement': self.apply_contrast_enhancement,
            'apply_sharpening': self.apply_sharpening,
            'applied_to_product': True,
            'applied_date': fields.Datetime.now(),
        }
        if width > 0:
            job_vals.update({
                'processed_width': width,
                'processed_height': height,
                'processed_file_size': file_size,
                'duration_sec': duration,
                'transformation_log': "\n".join(f"- {s}" for s in steps),
            })

        if self.existing_job_id:
            self.existing_job_id.write(job_vals)
        else:
            self.env['product.photo.editor'].create(job_vals)

        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': _('Product Photo Updated'),
                'message': _('New photo has been applied to "%s".') % self.product_id.display_name,
                'type': 'success',
                'sticky': False,
                'next': {'type': 'ir.actions.client', 'tag': 'reload'},
            }
        }

    def action_queue_background_job(self):
        """Queues the current editing task into background job processing."""
        self.ensure_one()
        job = self.env['product.photo.editor'].create({
            'product_id': self.product_id.id,
            'product_variant_id': self.product_variant_id.id if self.product_variant_id else False,
            'image_original': self.image_original,
            'background_style': self.background_style,
            'custom_bg_color': self.custom_bg_color,
            'dimensions': self.dimensions,
            'export_format': self.export_format,
            'export_quality': self.export_quality,
            'padding_percent': self.padding_percent,
            'service_provider': self.service_provider,
            'apply_perspective': self.apply_perspective,
            'apply_color_correction': self.apply_color_correction,
            'apply_auto_white_balance': self.apply_auto_white_balance,
            'apply_contrast_enhancement': self.apply_contrast_enhancement,
            'apply_sharpening': self.apply_sharpening,
            'status': 'pending',
        })

        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': _('Job Queued'),
                'message': _('Photo editor job %s enqueued for background processing.') % job.name,
                'type': 'info',
                'sticky': False,
            }
        }

    # -------------------------------------------------------------------------
    # Helper
    # -------------------------------------------------------------------------
    def _get_provider_config(self):
        ICP = self.env['ir.config_parameter'].sudo()
        return {
            'remove_bg_api_key': ICP.get_param('product_photo_editor.remove_bg_api_key', ''),
            'clipdrop_api_key': ICP.get_param('product_photo_editor.clipdrop_api_key', ''),
            'photoroom_api_key': ICP.get_param('product_photo_editor.photoroom_api_key', ''),
            'custom_ai_endpoint_url': ICP.get_param('product_photo_editor.custom_ai_endpoint_url', ''),
            'custom_ai_api_key': ICP.get_param('product_photo_editor.custom_ai_api_key', ''),
        }
