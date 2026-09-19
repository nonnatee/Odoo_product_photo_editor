# -*- coding: utf-8 -*-
"""
Product Photo Editor - REST Controllers
Exposes API endpoints to trigger image editing pipelines, manage presets,
enqueue batch jobs, and check asynchronous job statuses.
"""

import base64
import json
import logging

try:
    from odoo import http, _
    from odoo.http import request, Response
except ImportError:
    # Standalone mock shim for testing without full Odoo runtime
    class _DummyHttp:
        @staticmethod
        def route(*args, **kwargs):
            def decorator(f):
                return f
            return decorator
        class Controller:
            pass
    http = _DummyHttp()
    _ = lambda s: s
    request = None
    class Response:
        def __init__(self, response=None, status=200, headers=None, mimetype=None, content_type=None):
            self.data = response.encode('utf-8') if isinstance(response, str) else response
            self.status_code = status
            self.headers = headers or {}
            self.mimetype = mimetype

try:
    from ..models.image_pipeline import ImagePipeline, ImagePipelineError
except (ImportError, ValueError):
    from models.image_pipeline import ImagePipeline, ImagePipelineError

_logger = logging.getLogger(__name__)


class ProductPhotoEditorController(http.Controller):

    # -------------------------------------------------------------------------
    # REST: Process Image (Upload -> Process -> Return)
    # -------------------------------------------------------------------------
    @http.route(
        ['/api/v1/photo_editor/process'],
        type='http',
        auth='user',
        methods=['POST'],
        csrf=False,
    )
    def api_process_photo(self, **kw):
        """
        REST endpoint to trigger image processing.
        Accepts multipart/form-data or application/json.
        """
        try:
            # 1. Parse parameters from JSON body or form-data
            if request.httprequest.content_type and 'application/json' in request.httprequest.content_type:
                try:
                    payload = json.loads(request.httprequest.get_data().decode('utf-8'))
                except Exception:
                    payload = {}
            else:
                payload = kw

            # Extract image bytes
            image_bytes = None
            if 'image' in request.httprequest.files:
                file_storage = request.httprequest.files['image']
                image_bytes = file_storage.read()
            elif payload.get('image'):
                b64_str = payload['image']
                # Strip data URL prefix if present
                if ',' in b64_str:
                    b64_str = b64_str.split(',', 1)[1]
                image_bytes = base64.b64decode(b64_str)

            if not image_bytes:
                return self._json_response({'error': 'Missing required parameter: image (file or base64)'}, status=400)

            # Extract options
            background_style = payload.get('background_style', 'white')
            custom_bg_color = payload.get('custom_bg_color', '#FFFFFF')
            dimensions = payload.get('dimensions', 'square_2000')
            target_width = int(payload.get('target_width', 0)) if payload.get('target_width') else None
            target_height = int(payload.get('target_height', 0)) if payload.get('target_height') else None
            export_format = payload.get('export_format', 'JPEG').upper()
            export_quality = int(payload.get('export_quality', 90))
            padding_percent = float(payload.get('padding_percent', 8.0))

            apply_perspective = str(payload.get('apply_perspective', 'true')).lower() in ('true', '1')
            apply_color_correction = str(payload.get('apply_color_correction', 'true')).lower() in ('true', '1')
            apply_auto_white_balance = str(payload.get('apply_auto_white_balance', 'true')).lower() in ('true', '1')
            apply_contrast_enhancement = str(payload.get('apply_contrast_enhancement', 'true')).lower() in ('true', '1')
            apply_sharpening = str(payload.get('apply_sharpening', 'true')).lower() in ('true', '1')
            service_provider = payload.get('service_provider', 'local')

            # Fetch provider configuration
            ICP = request.env['ir.config_parameter'].sudo()
            provider_config = {
                'remove_bg_api_key': ICP.get_param('product_photo_editor.remove_bg_api_key', ''),
                'clipdrop_api_key': ICP.get_param('product_photo_editor.clipdrop_api_key', ''),
                'photoroom_api_key': ICP.get_param('product_photo_editor.photoroom_api_key', ''),
                'custom_ai_endpoint_url': ICP.get_param('product_photo_editor.custom_ai_endpoint_url', ''),
                'custom_ai_api_key': ICP.get_param('product_photo_editor.custom_ai_api_key', ''),
            }

            # 2. Run Image Pipeline
            result = ImagePipeline.process_image(
                image_bytes=image_bytes,
                background_style=background_style,
                custom_bg_color=custom_bg_color,
                dimensions=dimensions,
                target_width=target_width,
                target_height=target_height,
                export_format=export_format,
                export_quality=export_quality,
                apply_perspective=apply_perspective,
                apply_color_correction=apply_color_correction,
                apply_auto_white_balance=apply_auto_white_balance,
                apply_contrast_enhancement=apply_contrast_enhancement,
                apply_sharpening=apply_sharpening,
                padding_percent=padding_percent,
                service_provider=service_provider,
                provider_config=provider_config,
            )

            processed_b64 = base64.b64encode(result['image_bytes']).decode('ascii')

            # 3. Optional: Associate with Product Template & save record
            product_id = int(payload.get('product_id', 0)) if payload.get('product_id') else None
            job_id = None
            if product_id:
                product = request.env['product.template'].browse(product_id)
                if product.exists():
                    job = request.env['product.photo.editor'].create({
                        'product_id': product.id,
                        'image_original': base64.b64encode(image_bytes),
                        'image_processed': processed_b64,
                        'status': 'done',
                        'background_style': background_style,
                        'custom_bg_color': custom_bg_color,
                        'dimensions': dimensions,
                        'export_format': result['format'],
                        'export_quality': export_quality,
                        'service_provider': service_provider,
                        'processed_width': result['width'],
                        'processed_height': result['height'],
                        'processed_file_size': result['file_size'],
                        'duration_sec': result['duration_sec'],
                        'transformation_log': "\n".join(f"- {s}" for s in result['steps_applied']),
                    })
                    job_id = job.id

                    if str(payload.get('apply_to_product', 'false')).lower() in ('true', '1'):
                        job.action_apply_to_product()

            return self._json_response({
                'success': True,
                'status': 'done',
                'image_processed': processed_b64,
                'format': result['format'],
                'width': result['width'],
                'height': result['height'],
                'file_size': result['file_size'],
                'duration_sec': result['duration_sec'],
                'steps_applied': result['steps_applied'],
                'job_id': job_id,
            })

        except ImagePipelineError as e:
            _logger.warning("ImagePipeline error in REST endpoint: %s", e)
            return self._json_response({'error': str(e), 'success': False}, status=422)
        except Exception as e:
            _logger.exception("Unexpected error in photo editor REST endpoint: %s", e)
            return self._json_response({'error': str(e), 'success': False}, status=500)

    # -------------------------------------------------------------------------
    # REST: Metadata & Presets Catalog
    # -------------------------------------------------------------------------
    @http.route(
        ['/api/v1/photo_editor/presets'],
        type='http',
        auth='user',
        methods=['GET'],
        csrf=False,
    )
    def api_get_presets(self, **kw):
        """Returns catalogue of supported styles, dimensions, and providers."""
        data = {
            'background_styles': [{'id': s[0], 'name': s[1]} for s in ImagePipeline.BACKGROUND_STYLES],
            'dimension_presets': [
                {'id': k, 'width': v[0] if v else None, 'height': v[1] if v else None}
                for k, v in ImagePipeline.DIMENSION_PRESETS.items()
            ],
            'providers': [{'id': p[0], 'name': p[1]} for p in ImagePipeline.PROVIDERS],
            'export_formats': ['JPEG', 'PNG', 'WEBP'],
        }
        return self._json_response(data)

    # -------------------------------------------------------------------------
    # REST: Batch Enqueue
    # -------------------------------------------------------------------------
    @http.route(
        ['/api/v1/photo_editor/batch'],
        type='json',
        auth='user',
        methods=['POST'],
        csrf=False,
    )
    def api_batch_enqueue(self, product_ids=None, **kwargs):
        """Enqueues batch processing for a list of product IDs."""
        if not product_ids or not isinstance(product_ids, list):
            return {'error': 'product_ids list is required'}

        products = request.env['product.template'].browse(product_ids).filtered(lambda p: p.exists() and p.image_1920)
        created_jobs = []

        ICP = request.env['ir.config_parameter'].sudo()
        def_provider = kwargs.get('service_provider') or ICP.get_param('product_photo_editor.default_provider', 'local')
        def_bg = kwargs.get('background_style') or ICP.get_param('product_photo_editor.default_background_style', 'white')
        def_dim = kwargs.get('dimensions') or ICP.get_param('product_photo_editor.default_dimensions', 'square_2000')

        for prod in products:
            job = request.env['product.photo.editor'].create({
                'product_id': prod.id,
                'image_original': prod.image_1920,
                'service_provider': def_provider,
                'background_style': def_bg,
                'dimensions': def_dim,
                'status': 'pending',
            })
            created_jobs.append({'job_id': job.id, 'product_id': prod.id, 'product_name': prod.name})

        return {
            'success': True,
            'queued_count': len(created_jobs),
            'jobs': created_jobs,
        }

    # -------------------------------------------------------------------------
    # REST: Job Status Inquiry
    # -------------------------------------------------------------------------
    @http.route(
        ['/api/v1/photo_editor/status/<int:job_id>'],
        type='http',
        auth='user',
        methods=['GET'],
        csrf=False,
    )
    def api_get_job_status(self, job_id, **kw):
        """Returns execution status and result of a specific job."""
        job = request.env['product.photo.editor'].browse(job_id)
        if not job.exists():
            return self._json_response({'error': f'Job ID {job_id} not found'}, status=404)

        data = {
            'job_id': job.id,
            'reference': job.name,
            'status': job.status,
            'product_id': job.product_id.id,
            'product_name': job.product_id.name,
            'width': job.processed_width,
            'height': job.processed_height,
            'file_size': job.processed_file_size,
            'duration_sec': job.duration_sec,
            'applied_to_product': job.applied_to_product,
            'error_message': job.error_message,
        }
        if kw.get('include_image') == '1' and job.image_processed:
            data['image_processed'] = job.image_processed.decode('ascii') if isinstance(job.image_processed, bytes) else job.image_processed

        return self._json_response(data)

    # -------------------------------------------------------------------------
    # Helper
    # -------------------------------------------------------------------------
    @staticmethod
    def _json_response(payload: dict, status: int = 200) -> Response:
        return Response(
            json.dumps(payload),
            status=status,
            content_type='application/json; charset=utf-8',
        )
