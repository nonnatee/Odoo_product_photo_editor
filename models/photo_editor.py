# -*- coding: utf-8 -*-
"""
Product Photo Editor - Data Model
Stores photo editing jobs, metadata, original vs processed images,
configuration settings, and handles processing pipelines and product sync.
"""

import base64
import io
import logging
from typing import Dict, Any

from PIL import Image

from odoo import models, fields, api, _
from odoo.exceptions import UserError, ValidationError
from .image_pipeline import ImagePipeline, ImagePipelineError

_logger = logging.getLogger(__name__)


def _safe_b64decode(data):
    """Safely decode base64 string or bytes, stripping data URI headers and whitespace."""
    if not data:
        return b''
    if isinstance(data, str):
        data = data.encode('ascii')
    if b',' in data:
        data = data.split(b',', 1)[1]
    return base64.b64decode(data.strip())


class ProductPhotoEditor(models.Model):
    _name = 'product.photo.editor'
    _description = 'Product Photo Editor Job'
    _order = 'create_date desc, id desc'
    _rec_name = 'name'

    # Display / Identification
    name = fields.Char(
        string='Job Reference',
        copy=False,
        readonly=True,
        default=lambda self: _('New'),
    )

    # Relational Links
    product_id = fields.Many2one(
        'product.template',
        string='Product',
        required=True,
        ondelete='cascade',
        index=True,
    )
    product_variant_id = fields.Many2one(
        'product.product',
        string='Product Variant',
        ondelete='set null',
        domain="[('product_tmpl_id', '=', product_id)]",
    )
    company_id = fields.Many2one(
        'res.company',
        string='Company',
        default=lambda self: self.env.company,
    )
    active = fields.Boolean(default=True)

    # Workflow Status
    status = fields.Selection(
        [
            ('draft', 'Draft'),
            ('pending', 'Pending (Queued)'),
            ('processing', 'Processing'),
            ('done', 'Completed'),
            ('failed', 'Failed'),
        ],
        string='Status',
        default='draft',
        required=True,
        index=True,
        copy=False,
    )

    # Image Fields
    image_original = fields.Binary(
        string='Original Image',
        required=True,
        attachment=True,
    )
    image_original_filename = fields.Char(
        string='Original Filename',
        default='product_original.png',
    )
    image_processed = fields.Binary(
        string='Edited Image',
        attachment=True,
        copy=False,
    )
    image_processed_filename = fields.Char(
        string='Processed Filename',
        default='product_processed.jpg',
        copy=False,
    )

    # Editing Pipeline Configurations
    service_provider = fields.Selection(
        ImagePipeline.PROVIDERS,
        string='AI Provider',
        default='local',
        required=True,
        help="Select AI segmentation service. 'Local Engine' runs offline with OpenCV GrabCut.",
    )
    background_style = fields.Selection(
        ImagePipeline.BACKGROUND_STYLES,
        string='Background Style',
        default='white',
        required=True,
    )
    custom_bg_color = fields.Char(
        string='Custom Hex Color',
        default='#FFFFFF',
        help="Used when Background Style is 'Custom Hex Color'. E.g. #FFFFFF or #F0F4F8",
    )
    dimensions = fields.Selection(
        [
            ('square_2000', '2000 x 2000 px (Marketplace Standard)'),
            ('square_1000', '1000 x 1000 px (Web Catalog Standard)'),
            ('portrait_4_5', '1600 x 2000 px (Fashion / Instagram 4:5)'),
            ('landscape_16_9', '1920 x 1080 px (Hero Banner 16:9)'),
            ('original', 'Preserve Original Aspect Ratio'),
        ],
        string='Dimensions Preset',
        default='square_2000',
        required=True,
    )
    target_width = fields.Integer(string='Custom Width (px)')
    target_height = fields.Integer(string='Custom Height (px)')

    export_format = fields.Selection(
        [
            ('JPEG', 'JPEG (Optimized E-commerce)'),
            ('PNG', 'PNG (Lossless / Transparent)'),
            ('WEBP', 'WebP (Modern Web Compressed)'),
        ],
        string='Export Format',
        default='JPEG',
        required=True,
    )
    export_quality = fields.Integer(
        string='Export Quality',
        default=90,
        help="Compression quality (1-100). Default is 90.",
    )
    padding_percent = fields.Float(
        string='Product Padding (%)',
        default=8.0,
        help="Padding margin around product inside canvas (default 8% matches Amazon standard).",
    )

    # Transformation Toggles
    apply_perspective = fields.Boolean(
        string='Perspective Correction & Deskew',
        default=True,
        help="Automatically detects perspective skew and straightens product quad or orientation.",
    )
    apply_color_correction = fields.Boolean(
        string='Color & Lighting Correction',
        default=True,
        help="Enables automatic white balancing, exposure correction, and sharpening.",
    )
    apply_auto_white_balance = fields.Boolean(
        string='Auto White Balance',
        default=True,
        help="Gray-World algorithm to remove artificial color casts.",
    )
    apply_contrast_enhancement = fields.Boolean(
        string='CLAHE Contrast Enhancement',
        default=True,
        help="Enhances dynamic range and lifts dark shadows without blowing highlights.",
    )
    apply_sharpening = fields.Boolean(
        string='Unsharp Mask Sharpening',
        default=True,
        help="Subtle edge sharpening for crisp e-commerce marketplace presentation.",
    )

    # Technical Metadata & Logs
    original_width = fields.Integer(string='Original Width (px)', readonly=True)
    original_height = fields.Integer(string='Original Height (px)', readonly=True)
    original_file_size = fields.Integer(string='Original Size (Bytes)', readonly=True)
    processed_width = fields.Integer(string='Processed Width (px)', readonly=True)
    processed_height = fields.Integer(string='Processed Height (px)', readonly=True)
    processed_file_size = fields.Integer(string='Processed Size (Bytes)', readonly=True)
    duration_sec = fields.Float(string='Processing Time (s)', readonly=True)
    error_message = fields.Text(string='Error Details', readonly=True, copy=False)
    transformation_log = fields.Text(string='Transformation Log', readonly=True, copy=False)

    # Product Sync Tracking
    applied_to_product = fields.Boolean(
        string='Applied to Product Image',
        default=False,
        readonly=True,
        copy=False,
    )
    applied_date = fields.Datetime(
        string='Applied Date',
        readonly=True,
        copy=False,
    )

    # -------------------------------------------------------------------------
    # Computes & Constraints
    # -------------------------------------------------------------------------
    @api.depends('name', 'product_id', 'product_id.name', 'background_style')
    def _compute_display_name(self):
        for record in self:
            prod_name = record.product_id.name or _('Product')
            ref = record.name or _('Job')
            style = dict(ImagePipeline.BACKGROUND_STYLES).get(record.background_style, record.background_style or '')
            record.display_name = f"{ref} - {prod_name} ({style})"

    @api.constrains('padding_percent')
    def _check_padding_percent(self):
        for rec in self:
            if rec.padding_percent < 0.0 or rec.padding_percent > 40.0:
                raise ValidationError(_("Product padding must be between 0% and 40%."))

    @api.constrains('export_quality')
    def _check_export_quality(self):
        for rec in self:
            if rec.export_quality < 10 or rec.export_quality > 100:
                raise ValidationError(_("Export quality must be between 10 and 100."))

    # -------------------------------------------------------------------------
    # CRUD Overrides
    # -------------------------------------------------------------------------
    @api.model_create_multi
    def create(self, vals_list):
        for vals in vals_list:
            if not vals.get('name') or vals.get('name') == _('New'):
                vals['name'] = self.env['ir.sequence'].next_by_code('product.photo.editor') or _('PPE-%s') % fields.Datetime.now().strftime('%Y%m%d%H%M%S')
            # Compute original image dimensions
            if vals.get('image_original'):
                dims = self._read_image_dimensions(vals['image_original'])
                vals['original_width'] = dims.get('width', 0)
                vals['original_height'] = dims.get('height', 0)
                vals['original_file_size'] = dims.get('size', 0)
        return super().create(vals_list)

    def write(self, vals):
        if 'image_original' in vals and vals['image_original']:
            dims = self._read_image_dimensions(vals['image_original'])
            vals['original_width'] = dims.get('width', 0)
            vals['original_height'] = dims.get('height', 0)
            vals['original_file_size'] = dims.get('size', 0)
        return super().write(vals)

    # -------------------------------------------------------------------------
    # Actions & Business Logic
    # -------------------------------------------------------------------------
    def action_process(self):
        """Executes image editing pipeline for the current record(s)."""
        self.ensure_one()
        if not self.image_original:
            raise UserError(_("Please upload an original image to process."))

        self.write({
            'status': 'processing',
            'error_message': False,
        })

        try:
            image_bytes = _safe_b64decode(self.image_original)
            provider_config = self._get_provider_config()

            result = ImagePipeline.process_image(
                image_bytes=image_bytes,
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
            ext = result['format'].lower()
            if ext == 'jpeg':
                ext = 'jpg'
            filename = f"edited_{self.product_id.default_code or 'product'}_{self.name}.{ext}"

            self.write({
                'status': 'done',
                'image_processed': processed_b64,
                'image_processed_filename': filename,
                'processed_width': result['width'],
                'processed_height': result['height'],
                'processed_file_size': result['file_size'],
                'duration_sec': result['duration_sec'],
                'transformation_log': "\n".join(f"- {s}" for s in result['steps_applied']),
                'error_message': False,
            })

            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': _('Photo Processed Successfully'),
                    'message': _('Edited photo created in %s seconds (%sx%s).') % (result['duration_sec'], result['width'], result['height']),
                    'type': 'success',
                    'sticky': False,
                    'next': {'type': 'ir.actions.client', 'tag': 'reload'},
                }
            }

        except Exception as e:
            _logger.exception("Product photo editing failed for job %s: %s", self.id, e)
            self.write({
                'status': 'failed',
                'error_message': str(e),
            })
            raise UserError(_("Photo editing failed: %s") % str(e))

    def action_apply_to_product(self):
        """Replaces the product template's primary image_1920 with the processed photo."""
        self.ensure_one()
        if not self.image_processed:
            raise UserError(_("No edited image available to apply. Please process the photo first."))

        # Apply to product template
        self.product_id.write({
            'image_1920': self.image_processed,
        })

        # Apply to specific variant if specified
        if self.product_variant_id:
            self.product_variant_id.write({
                'image_1920': self.image_processed,
            })

        self.write({
            'applied_to_product': True,
            'applied_date': fields.Datetime.now(),
        })

        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': _('Product Image Updated'),
                'message': _('The edited photo was successfully applied to product "%s".') % self.product_id.display_name,
                'type': 'success',
                'sticky': False,
                'next': {'type': 'ir.actions.client', 'tag': 'reload'},
            }
        }

    def action_reset_draft(self):
        """Resets status back to draft."""
        self.write({
            'status': 'draft',
            'error_message': False,
        })

    def action_queue_pending(self):
        """Queues the record for background cron processing."""
        self.write({'status': 'pending'})

    def action_open_wizard(self):
        """Opens the interactive editing wizard populated with this job's parameters."""
        self.ensure_one()
        return {
            'type': 'ir.actions.act_window',
            'name': _('AI Photo Editor Wizard'),
            'res_model': 'product.photo.editor.wizard',
            'view_mode': 'form',
            'target': 'new',
            'context': {
                'default_product_id': self.product_id.id,
                'default_image_original': self.image_original,
                'default_background_style': self.background_style,
                'default_custom_bg_color': self.custom_bg_color,
                'default_dimensions': self.dimensions,
                'default_export_format': self.export_format,
                'default_export_quality': self.export_quality,
                'default_apply_perspective': self.apply_perspective,
                'default_apply_color_correction': self.apply_color_correction,
                'default_apply_auto_white_balance': self.apply_auto_white_balance,
                'default_apply_contrast_enhancement': self.apply_contrast_enhancement,
                'default_apply_sharpening': self.apply_sharpening,
                'default_padding_percent': self.padding_percent,
                'default_service_provider': self.service_provider,
                'default_existing_job_id': self.id,
            }
        }

    # -------------------------------------------------------------------------
    # Batch Processing Cron Job
    # -------------------------------------------------------------------------
    @api.model
    def _cron_batch_process(self, batch_limit=25):
        """
        Scheduled action for bulk product catalogs.
        Picks pending jobs and processes them sequentially.
        """
        pending_jobs = self.search([('status', '=', 'pending')], limit=batch_limit)
        _logger.info("Product Photo Editor cron started: found %d pending jobs.", len(pending_jobs))

        auto_apply = self.env['ir.config_parameter'].sudo().get_param('product_photo_editor.auto_apply_cron', 'False') == 'True'

        processed_count = 0
        error_count = 0

        for job in pending_jobs:
            try:
                job.action_process()
                if auto_apply:
                    job.action_apply_to_product()
                processed_count += 1
            except Exception as e:
                _logger.error("Failed to process job ID %d during cron: %s", job.id, e)
                error_count += 1

        _logger.info("Product Photo Editor cron completed. Processed: %d, Errors: %d", processed_count, error_count)

    # -------------------------------------------------------------------------
    # Private Helpers
    # -------------------------------------------------------------------------
    def _get_provider_config(self) -> Dict[str, Any]:
        """Fetches API keys and webhook settings from ir.config_parameter."""
        ICP = self.env['ir.config_parameter'].sudo()
        return {
            'remove_bg_api_key': ICP.get_param('product_photo_editor.remove_bg_api_key', ''),
            'clipdrop_api_key': ICP.get_param('product_photo_editor.clipdrop_api_key', ''),
            'photoroom_api_key': ICP.get_param('product_photo_editor.photoroom_api_key', ''),
            'custom_ai_endpoint_url': ICP.get_param('product_photo_editor.custom_ai_endpoint_url', ''),
            'custom_ai_api_key': ICP.get_param('product_photo_editor.custom_ai_api_key', ''),
        }

    @staticmethod
    def _read_image_dimensions(b64_data: str) -> Dict[str, int]:
        """Reads width, height, and raw byte length from base64 image data."""
        try:
            raw_bytes = _safe_b64decode(b64_data)
            img = Image.open(io.BytesIO(raw_bytes))
            return {
                'width': img.width,
                'height': img.height,
                'size': len(raw_bytes),
            }
        except Exception:
            return {'width': 0, 'height': 0, 'size': 0}
