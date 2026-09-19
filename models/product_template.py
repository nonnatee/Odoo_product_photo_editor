# -*- coding: utf-8 -*-
"""
Product Photo Editor - Product Template Extension
Adds one-click photo editing, job count smart button, and batch optimization actions.
"""

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
                action = act_rec.read()[0]
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

        message = _("Enqueued %d photo optimization jobs for background batch processing.") % len(created_jobs)
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
