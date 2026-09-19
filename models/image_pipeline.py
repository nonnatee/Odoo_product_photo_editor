# -*- coding: utf-8 -*-
"""
Product Photo Editor - Image Processing Pipeline Engine
Supports:
- Background Removal: Local edge-aware GrabCut + alpha matting, and external APIs (remove.bg, Clipdrop, Photoroom, Custom Webhook)
- Perspective Correction & Auto-Deskewing via OpenCV
- Color & Lighting Correction: Gray World Auto White Balance + LAB CLAHE Exposure + Unsharp Mask Sharpening
- Dimension Standardization: 2000x2000, 1000x1000, 1600x2000, 1920x1080 with padding
- Background Composition: Transparent, Studio White, Studio Neutral, Gradient, Lifestyle (Wood, Marble), Custom Colors
- Contact & Soft Drop Shadow Generation for realistic e-commerce grounding
- Multi-format Export: PNG, JPEG, WEBP with sRGB color profile
"""

import base64
import io
import json
import logging
import math
import time
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps

try:
    import requests
except ImportError:
    requests = None

_logger = logging.getLogger(__name__)


class ImagePipelineError(Exception):
    """Custom exception for image pipeline errors."""
    pass


class ImagePipeline:
    """Core image processing pipeline for e-commerce product photos."""

    # Dimension preset definitions (width, height)
    DIMENSION_PRESETS = {
        'square_2000': (2000, 2000),      # Marketplace standard (Amazon, Shopify, eBay)
        'square_1000': (1000, 1000),      # Standard web catalog
        'portrait_4_5': (1600, 2000),     # Fashion / Instagram (4:5)
        'landscape_16_9': (1920, 1080),   # Hero banner (16:9)
        'original': None,                 # Preserve original dimensions
    }

    # Background styles
    BACKGROUND_STYLES = [
        ('transparent', 'Transparent PNG'),
        ('white', 'Pure White (#FFFFFF, Studio Standard)'),
        ('studio_neutral', 'Studio Neutral Light Grey (#F4F4F4)'),
        ('studio_soft_shadow', 'Studio White with Soft Grounding Shadow'),
        ('lifestyle_wood', 'Lifestyle Warm Wood Surface'),
        ('lifestyle_marble', 'Lifestyle White Marble Countertop'),
        ('lifestyle_gradient', 'Modern Clean Studio Gradient'),
        ('custom_color', 'Custom Hex Color'),
    ]

    # Service providers
    PROVIDERS = [
        ('local', 'Local Engine (OpenCV + Alpha Matting)'),
        ('remove_bg', 'Remove.bg API'),
        ('clipdrop', 'Clipdrop API (Stability AI)'),
        ('photoroom', 'Photoroom API'),
        ('custom_webhook', 'Custom AI Endpoint / Webhook'),
    ]

    # -------------------------------------------------------------------------
    # Public Entrypoint
    # -------------------------------------------------------------------------
    @classmethod
    def process_image(
        cls,
        image_bytes: bytes,
        background_style: str = 'white',
        custom_bg_color: str = '#FFFFFF',
        dimensions: str = 'square_2000',
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        export_format: str = 'JPEG',
        export_quality: int = 90,
        apply_perspective: bool = True,
        apply_color_correction: bool = True,
        apply_auto_white_balance: bool = True,
        apply_contrast_enhancement: bool = True,
        apply_sharpening: bool = True,
        padding_percent: float = 8.0,
        service_provider: str = 'local',
        provider_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute full image editing pipeline.
        Returns a dictionary containing:
            - 'image_bytes': bytes of processed image
            - 'format': export format ('JPEG', 'PNG', 'WEBP')
            - 'width': output width
            - 'height': output height
            - 'file_size': output byte length
            - 'duration_sec': execution duration in seconds
            - 'steps_applied': list of strings describing applied transformations
        """
        start_time = time.time()
        steps_applied = []
        provider_config = provider_config or {}

        if not image_bytes:
            raise ImagePipelineError("No image data provided for processing.")

        # 1. Decode input image
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            # Auto-orient based on EXIF
            pil_img = ImageOps.exif_transpose(pil_img)
            orig_w, orig_h = pil_img.size
        except Exception as e:
            raise ImagePipelineError(f"Failed to decode input image: {e}")

        steps_applied.append(f"Decoded input image ({orig_w}x{orig_h})")

        # Check if input already has an existing alpha mask/cutout
        has_existing_alpha = False
        existing_alpha_channel = None
        if pil_img.mode in ('RGBA', 'LA') or (pil_img.mode == 'P' and 'transparency' in pil_img.info):
            rgba_temp = pil_img.convert('RGBA')
            alpha_extrema = rgba_temp.getextrema()[3]
            if alpha_extrema[0] < 250:
                has_existing_alpha = True
                existing_alpha_channel = rgba_temp.split()[-1]

        # Convert to OpenCV BGR for computer vision operations
        # If image has alpha, composite onto white before color operations to avoid dark fringes
        if has_existing_alpha:
            white_bg = Image.new('RGB', pil_img.size, (255, 255, 255))
            white_bg.paste(pil_img.convert('RGBA'), mask=existing_alpha_channel)
            cv_img = cls._pil_to_cv2(white_bg)
        else:
            cv_img = cls._pil_to_cv2(pil_img.convert('RGB'))

        # 2. Perspective correction / Auto-deskewing
        if apply_perspective:
            try:
                cv_img, deskew_applied = cls._correct_perspective_or_deskew(cv_img)
                if deskew_applied:
                    steps_applied.append("Applied perspective correction & deskewing")
            except Exception as e:
                _logger.warning("Perspective correction failed, continuing: %s", e)

        # 3. Color and Lighting adjustments
        if apply_color_correction:
            if apply_auto_white_balance:
                cv_img = cls._auto_white_balance(cv_img)
                steps_applied.append("Applied Gray-World auto white balance")
            if apply_contrast_enhancement:
                cv_img = cls._enhance_contrast_clahe(cv_img)
                steps_applied.append("Applied CLAHE adaptive contrast & exposure")
            if apply_sharpening:
                cv_img = cls._sharpen_image(cv_img)
                steps_applied.append("Applied unsharp mask sharpening")

        # Convert back to PIL
        rgb_pil = cls._cv2_to_pil(cv_img)

        # 4. Background Removal / Foreground Cutout (RGBA)
        rgba_cutout, provider_msg = cls._remove_background(
            rgb_pil=rgb_pil,
            image_bytes_orig=image_bytes,
            provider=service_provider,
            config=provider_config,
            has_existing_alpha=has_existing_alpha,
            existing_alpha_channel=existing_alpha_channel,
        )
        steps_applied.append(f"Segmented foreground cutout using {provider_msg}")

        # 5. Determine target canvas dimensions
        if target_width and target_height and target_width > 0 and target_height > 0:
            final_w, final_h = target_width, target_height
        elif dimensions in cls.DIMENSION_PRESETS and cls.DIMENSION_PRESETS[dimensions]:
            final_w, final_h = cls.DIMENSION_PRESETS[dimensions]
        else:
            final_w, final_h = orig_w, orig_h

        # 6. Fit cutout onto standardized canvas with padding
        padding_ratio = max(0.0, min(0.4, padding_percent / 100.0))
        avail_w = int(final_w * (1.0 - 2 * padding_ratio))
        avail_h = int(final_h * (1.0 - 2 * padding_ratio))

        cutout_w, cutout_h = rgba_cutout.size
        scale = min(avail_w / max(cutout_w, 1), avail_h / max(cutout_h, 1))
        new_cutout_w = max(1, int(cutout_w * scale))
        new_cutout_h = max(1, int(cutout_w * scale))

        resized_cutout = rgba_cutout.resize((new_cutout_w, new_cutout_h), Image.Resampling.LANCZOS)
        steps_applied.append(f"Standardized cutout to {new_cutout_w}x{new_cutout_h} (Canvas: {final_w}x{final_h}, Padding: {padding_percent}%)")

        # Cutout positioning (centered horizontally, grounded vertically)
        pos_x = (final_w - new_cutout_w) // 2
        pos_y = (final_h - new_cutout_h) // 2

        # 7. Render background & composition
        final_canvas = cls._compose_background(
            cutout=resized_cutout,
            canvas_w=final_w,
            canvas_h=final_h,
            pos_x=pos_x,
            pos_y=pos_y,
            style=background_style,
            custom_hex=custom_bg_color,
        )
        steps_applied.append(f"Composed background style '{background_style}'")

        # 8. Multi-format export optimization
        fmt = export_format.upper()
        if fmt not in ('JPEG', 'JPG', 'PNG', 'WEBP'):
            fmt = 'JPEG'
        if background_style == 'transparent' and fmt in ('JPEG', 'JPG'):
            # JPEG cannot store transparency; automatically switch to PNG
            fmt = 'PNG'

        out_buffer = io.BytesIO()
        save_kwargs = {}

        if fmt in ('JPEG', 'JPG'):
            fmt = 'JPEG'
            save_img = final_canvas.convert('RGB')
            save_kwargs = {
                'quality': max(10, min(100, export_quality)),
                'optimize': True,
                'progressive': True,
            }
        elif fmt == 'PNG':
            save_img = final_canvas if final_canvas.mode == 'RGBA' else final_canvas.convert('RGBA')
            save_kwargs = {
                'optimize': True,
                'compress_level': 6,
            }
        elif fmt == 'WEBP':
            save_img = final_canvas
            save_kwargs = {
                'quality': max(10, min(100, export_quality)),
                'method': 6,
            }

        # Set 300 DPI metadata for crisp printing and e-commerce display
        save_kwargs['dpi'] = (300, 300)

        # Embed sRGB profile if available
        try:
            srgb_profile = cls._get_srgb_profile()
            if srgb_profile:
                save_kwargs['icc_profile'] = srgb_profile
        except Exception:
            pass

        save_img.save(out_buffer, format=fmt, **save_kwargs)
        processed_bytes = out_buffer.getvalue()
        duration = round(time.time() - start_time, 3)

        return {
            'image_bytes': processed_bytes,
            'format': fmt,
            'width': final_w,
            'height': final_h,
            'file_size': len(processed_bytes),
            'duration_sec': duration,
            'steps_applied': steps_applied,
        }

    # -------------------------------------------------------------------------
    # Perspective Correction & Auto-Deskew
    # -------------------------------------------------------------------------
    @classmethod
    def _correct_perspective_or_deskew(cls, img: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Detects quadrilateral contours for perspective wrap or falls back to
        minimum bounding box rotation for deskewing.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        total_area = h * w

        # Smooth to eliminate fine noise
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate edges to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img, False

        # Sort contours by area descending
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Attempt 4-corner perspective warp on largest significant contour
        min_perspective_area = max(500, int(total_area * 0.03))
        for c in contours[:3]:
            area = cv2.contourArea(c)
            if area < min_perspective_area:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                pts = approx.reshape(4, 2).astype(np.float32)
                warped = cls._four_point_transform(img, pts)
                wh, ww = warped.shape[:2]
                if wh >= 30 and ww >= 30 and (wh * ww) >= min_perspective_area:
                    return warped, True

        # Fallback: Check rotation tilt angle using minAreaRect on primary object
        primary_c = contours[0]
        if cv2.contourArea(primary_c) > max(400, int(total_area * 0.02)):
            rect = cv2.minAreaRect(primary_c)
            raw_angle = rect[-1]
            # OpenCV 4.5+ returns angle in [0, 90). Compute deviation from nearest orthogonal axis
            dev = raw_angle - 90.0 if raw_angle > 45.0 else raw_angle
            if abs(dev) > 0.5 and abs(dev) < 35.0:
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, dev, 1.0)
                rotated = cv2.warpAffine(
                    img, M, (w, h),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255),
                )
                return rotated, True

        return img, False

    @staticmethod
    def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Applies perspective transform given 4 quad points."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum
        rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right has smallest diff
        rect[3] = pts[np.argmax(diff)]  # Bottom-left has largest diff

        (tl, tr, br, bl) = rect
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_w = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_h = max(int(height_a), int(height_b))

        dst = np.array([
            [0, 0],
            [max_w - 1, 0],
            [max_w - 1, max_h - 1],
            [0, max_h - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (max_w, max_h), flags=cv2.INTER_LANCZOS4)

    # -------------------------------------------------------------------------
    # Color & Lighting Correction
    # -------------------------------------------------------------------------
    @classmethod
    def _auto_white_balance(cls, img: np.ndarray) -> np.ndarray:
        """Gray World Auto White Balance algorithm."""
        b, g, r = cv2.split(img)
        mean_b = np.mean(b)
        mean_g = np.mean(g)
        mean_r = np.mean(r)

        if mean_b == 0 or mean_g == 0 or mean_r == 0:
            return img

        mean_gray = (mean_b + mean_g + mean_r) / 3.0

        scale_b = min(mean_gray / mean_b, 2.0)
        scale_g = min(mean_gray / mean_g, 2.0)
        scale_r = min(mean_gray / mean_r, 2.0)

        b = np.clip(b.astype(np.float32) * scale_b, 0, 255).astype(np.uint8)
        g = np.clip(g.astype(np.float32) * scale_g, 0, 255).astype(np.uint8)
        r = np.clip(r.astype(np.float32) * scale_r, 0, 255).astype(np.uint8)

        return cv2.merge([b, g, r])

    @classmethod
    def _enhance_contrast_clahe(cls, img: np.ndarray) -> np.ndarray:
        """Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB space."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    @classmethod
    def _sharpen_image(cls, img: np.ndarray) -> np.ndarray:
        """Crisp unsharp mask sharpening for e-commerce catalog detail."""
        gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
        sharpened = cv2.addWeighted(img, 1.35, gaussian, -0.35, 0)
        return sharpened

    # -------------------------------------------------------------------------
    # Background Removal Engine
    # -------------------------------------------------------------------------
    @classmethod
    def _remove_background(
        cls,
        rgb_pil: Image.Image,
        image_bytes_orig: bytes,
        provider: str,
        config: Dict[str, Any],
        has_existing_alpha: bool = False,
        existing_alpha_channel: Optional[Image.Image] = None,
    ) -> Tuple[Image.Image, str]:
        """Dispatches foreground segmentation to requested provider or local fallback."""
        if has_existing_alpha and existing_alpha_channel and provider == 'local':
            if existing_alpha_channel.size != rgb_pil.size:
                existing_alpha_channel = existing_alpha_channel.resize(rgb_pil.size, Image.Resampling.LANCZOS)
            r, g, b = rgb_pil.split()
            rgba = Image.merge('RGBA', (r, g, b, existing_alpha_channel))
            bbox = existing_alpha_channel.getbbox()
            if bbox:
                rgba = rgba.crop(bbox)
            return rgba, "pre-existing transparent cutout"

        if provider == 'remove_bg':
            api_key = config.get('remove_bg_api_key')
            if api_key:
                try:
                    return cls._call_remove_bg_api(image_bytes_orig, api_key), 'remove.bg API'
                except Exception as e:
                    _logger.warning("remove.bg API failed (%s), falling back to local engine", e)
                    return cls._local_grabcut_segmentation(rgb_pil), 'local engine (fallback from remove.bg)'

        elif provider == 'clipdrop':
            api_key = config.get('clipdrop_api_key')
            if api_key:
                try:
                    return cls._call_clipdrop_api(image_bytes_orig, api_key), 'Clipdrop API'
                except Exception as e:
                    _logger.warning("Clipdrop API failed (%s), falling back to local engine", e)
                    return cls._local_grabcut_segmentation(rgb_pil), 'local engine (fallback from Clipdrop)'

        elif provider == 'photoroom':
            api_key = config.get('photoroom_api_key')
            if api_key:
                try:
                    return cls._call_photoroom_api(image_bytes_orig, api_key), 'Photoroom API'
                except Exception as e:
                    _logger.warning("Photoroom API failed (%s), falling back to local engine", e)
                    return cls._local_grabcut_segmentation(rgb_pil), 'local engine (fallback from Photoroom)'

        elif provider == 'custom_webhook':
            endpoint = config.get('custom_ai_endpoint_url')
            auth_token = config.get('custom_ai_api_key')
            if endpoint:
                try:
                    return cls._call_custom_ai_endpoint(image_bytes_orig, endpoint, auth_token), 'Custom AI webhook'
                except Exception as e:
                    _logger.warning("Custom AI webhook failed (%s), falling back to local engine", e)
                    return cls._local_grabcut_segmentation(rgb_pil), 'local engine (fallback from Custom AI)'

        # Local segmentation engine (OpenCV GrabCut + adaptive edge matting)
        return cls._local_grabcut_segmentation(rgb_pil), f"provider '{provider}' (local engine)"

    @classmethod
    def _local_grabcut_segmentation(cls, rgb_pil: Image.Image) -> Image.Image:
        """
        High-quality offline segmentation using OpenCV GrabCut + morphological
        refinement + edge alpha feathering. Produces clean RGBA cutout.
        """
        cv_rgb = np.array(rgb_pil)
        cv_bgr = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2BGR)
        h, w = cv_bgr.shape[:2]

        # Guard against extremely small images where GrabCut GMM initialization fails
        if w < 20 or h < 20:
            r, g, b = rgb_pil.split()
            full_alpha = Image.new('L', (w, h), 255)
            return Image.merge('RGBA', (r, g, b, full_alpha))

        # 1. Edge and saliency estimation to locate central product bounding box
        gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        norm_grad = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        _, thresh = cv2.threshold(norm_grad, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        margin_x = max(1, min(int(w * 0.05), w // 4))
        margin_y = max(1, min(int(h * 0.05), h // 4))
        rect_w = max(1, w - 2 * margin_x)
        rect_h = max(1, h - 2 * margin_y)
        rect = (margin_x, margin_y, rect_w, rect_h)

        if contours:
            significant = [c for c in contours if cv2.contourArea(c) > (w * h * 0.01)]
            if significant:
                all_pts = np.vstack(significant)
                bx, by, bw, bh = cv2.boundingRect(all_pts)
                exp_x = max(2, int(bw * 0.03))
                exp_y = max(2, int(bh * 0.03))
                nx = max(1, bx - exp_x)
                ny = max(1, by - exp_y)
                nw = min(w - nx - 1, bw + 2 * exp_x)
                nh = min(h - ny - 1, bh + 2 * exp_y)
                if nw > 15 and nh > 15 and (nx + nw <= w) and (ny + nh <= h):
                    rect = (nx, ny, nw, nh)

        # 2. GrabCut initialization
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            if rect[2] >= 10 and rect[3] >= 10 and rect[0] + rect[2] <= w and rect[1] + rect[3] <= h:
                cv2.grabCut(cv_bgr, mask, rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_RECT)
                bin_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
            else:
                _, bin_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        except Exception as e:
            _logger.warning("GrabCut failed, using threshold fallback: %s", e)
            _, bin_mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
            if np.sum(bin_mask) == 0:
                # If pure white / no contrast, keep full object
                bin_mask = np.full((h, w), 255, dtype=np.uint8)

        # 3. Morphological cleanup: remove small noise islands and close internal gaps
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel_clean, iterations=2)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel_clean, iterations=1)

        # 4. Alpha edge feathering (soft antialiasing)
        feathered_alpha = cv2.GaussianBlur(bin_mask, (5, 5), 1.5)

        # 5. Assemble RGBA PIL image
        r, g, b = rgb_pil.split()
        alpha = Image.fromarray(feathered_alpha, mode='L')
        rgba = Image.merge('RGBA', (r, g, b, alpha))

        # Crop to tight bounding box of cutout object if significant
        bbox = alpha.getbbox()
        if bbox:
            rgba = rgba.crop(bbox)

        return rgba

    # -------------------------------------------------------------------------
    # External AI API Integrations
    # -------------------------------------------------------------------------
    @classmethod
    def _call_remove_bg_api(cls, image_bytes: bytes, api_key: str) -> Image.Image:
        """Invokes remove.bg API v1.0."""
        if not requests:
            raise ImagePipelineError("requests library is not installed.")

        b64_img = base64.b64encode(image_bytes).decode('utf-8')
        response = requests.post(
            'https://api.remove.bg/v1.0/removebg',
            data={'image_file_b64': b64_img, 'size': 'auto', 'format': 'png'},
            headers={'X-Api-Key': api_key},
            timeout=30,
        )
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content)).convert('RGBA')
        raise ImagePipelineError(f"remove.bg API failed [{response.status_code}]: {response.text}")

    @classmethod
    def _call_clipdrop_api(cls, image_bytes: bytes, api_key: str) -> Image.Image:
        """Invokes Clipdrop remove-background API."""
        if not requests:
            raise ImagePipelineError("requests library is not installed.")

        files = {'image_file': ('image.png', image_bytes, 'image/png')}
        response = requests.post(
            'https://clipdrop-api.co/remove-background/v1',
            files=files,
            headers={'x-api-key': api_key},
            timeout=30,
        )
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content)).convert('RGBA')
        raise ImagePipelineError(f"Clipdrop API failed [{response.status_code}]: {response.text}")

    @classmethod
    def _call_photoroom_api(cls, image_bytes: bytes, api_key: str) -> Image.Image:
        """Invokes Photoroom segment API."""
        if not requests:
            raise ImagePipelineError("requests library is not installed.")

        files = {'image_file': ('image.png', image_bytes, 'image/png')}
        response = requests.post(
            'https://sdk.photoroom.com/v1/segment',
            files=files,
            headers={'x-api-key': api_key},
            timeout=30,
        )
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content)).convert('RGBA')
        raise ImagePipelineError(f"Photoroom API failed [{response.status_code}]: {response.text}")

    @classmethod
    def _call_custom_ai_endpoint(cls, image_bytes: bytes, url: str, token: Optional[str]) -> Image.Image:
        """Invokes custom external AI segmentation endpoint."""
        if not requests:
            raise ImagePipelineError("requests library is not installed.")

        headers = {}
        if token:
            headers['Authorization'] = f"Bearer {token}"
            headers['x-api-key'] = token

        files = {'file': ('product.png', image_bytes, 'image/png')}
        response = requests.post(url, files=files, headers=headers, timeout=45)
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            if 'application/json' in content_type:
                data = response.json()
                b64_str = data.get('image') or data.get('image_base64') or data.get('result')
                if b64_str:
                    img_data = base64.b64decode(b64_str)
                    return Image.open(io.BytesIO(img_data)).convert('RGBA')
                raise ImagePipelineError("Custom AI response JSON missing image base64 field.")
            return Image.open(io.BytesIO(response.content)).convert('RGBA')
        raise ImagePipelineError(f"Custom AI endpoint returned [{response.status_code}]: {response.text}")

    # -------------------------------------------------------------------------
    # Background Composition & Shadow Styling
    # -------------------------------------------------------------------------
    @classmethod
    def _compose_background(
        cls,
        cutout: Image.Image,
        canvas_w: int,
        canvas_h: int,
        pos_x: int,
        pos_y: int,
        style: str,
        custom_hex: str,
    ) -> Image.Image:
        """Composes cutout onto requested background with optional realistic drop shadows."""
        cutout_w, cutout_h = cutout.size

        # 1. Transparent
        if style == 'transparent':
            canvas = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
            canvas.alpha_composite(cutout, (pos_x, pos_y))
            return canvas

        # 2. Pure White (Studio Standard)
        if style == 'white':
            canvas = Image.new('RGBA', (canvas_w, canvas_h), (255, 255, 255, 255))
            canvas.alpha_composite(cutout, (pos_x, pos_y))
            return canvas

        # 3. Studio Neutral Grey
        if style == 'studio_neutral':
            canvas = Image.new('RGBA', (canvas_w, canvas_h), (244, 244, 244, 255))
            canvas.alpha_composite(cutout, (pos_x, pos_y))
            return canvas

        # 4. Studio Soft Grounding Shadow
        if style == 'studio_soft_shadow':
            canvas = Image.new('RGBA', (canvas_w, canvas_h), (255, 255, 255, 255))
            shadow_layer = cls._generate_realistic_shadow(
                cutout=cutout,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                pos_x=pos_x,
                pos_y=pos_y,
            )
            canvas.alpha_composite(shadow_layer, (0, 0))
            canvas.alpha_composite(cutout, (pos_x, pos_y))
            return canvas

        # 5. Lifestyle Warm Wood Surface
        if style == 'lifestyle_wood':
            bg = cls._render_lifestyle_wood(canvas_w, canvas_h)
            shadow_layer = cls._generate_realistic_shadow(
                cutout=cutout,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                pos_x=pos_x,
                pos_y=pos_y,
            )
            bg.alpha_composite(shadow_layer, (0, 0))
            bg.alpha_composite(cutout, (pos_x, pos_y))
            return bg

        # 6. Lifestyle Marble Countertop
        if style == 'lifestyle_marble':
            bg = cls._render_lifestyle_marble(canvas_w, canvas_h)
            shadow_layer = cls._generate_realistic_shadow(
                cutout=cutout,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                pos_x=pos_x,
                pos_y=pos_y,
            )
            bg.alpha_composite(shadow_layer, (0, 0))
            bg.alpha_composite(cutout, (pos_x, pos_y))
            return bg

        # 7. Modern Clean Studio Gradient
        if style == 'lifestyle_gradient':
            bg = cls._render_studio_gradient(canvas_w, canvas_h)
            shadow_layer = cls._generate_realistic_shadow(
                cutout=cutout,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                pos_x=pos_x,
                pos_y=pos_y,
            )
            bg.alpha_composite(shadow_layer, (0, 0))
            bg.alpha_composite(cutout, (pos_x, pos_y))
            return bg

        # 8. Custom Hex Color
        if style == 'custom_color':
            rgb = cls._hex_to_rgb(custom_hex)
            canvas = Image.new('RGBA', (canvas_w, canvas_h), (rgb[0], rgb[1], rgb[2], 255))
            canvas.alpha_composite(cutout, (pos_x, pos_y))
            return canvas

        # Default fallback
        canvas = Image.new('RGBA', (canvas_w, canvas_h), (255, 255, 255, 255))
        canvas.alpha_composite(cutout, (pos_x, pos_y))
        return canvas

    @classmethod
    def _generate_realistic_shadow(
        cls,
        cutout: Image.Image,
        canvas_w: int,
        canvas_h: int,
        pos_x: int,
        pos_y: int,
    ) -> Image.Image:
        """
        Generates dual shadow:
        1. Dark contact shadow underneath product base
        2. Soft ambient drop shadow for depth
        """
        cutout_w, cutout_h = cutout.size
        shadow_layer = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))

        # 1. Contact shadow (flattened dark ellipse at bottom contact line)
        contact_w = int(cutout_w * 0.85)
        contact_h = max(8, int(cutout_h * 0.08))
        contact_x0 = pos_x + (cutout_w - contact_w) // 2
        contact_y0 = pos_y + cutout_h - (contact_h // 2)

        contact_img = Image.new('RGBA', (contact_w + 40, contact_h + 40), (0, 0, 0, 0))
        c_draw = ImageDraw.Draw(contact_img)
        c_draw.ellipse(
            [(20, 20), (20 + contact_w, 20 + contact_h)],
            fill=(20, 20, 25, 160)
        )
        contact_blurred = contact_img.filter(ImageFilter.GaussianBlur(radius=max(3, contact_h // 3)))
        shadow_layer.alpha_composite(contact_blurred, (contact_x0 - 20, contact_y0 - 20))

        # 2. Ambient drop shadow (downward blurred projection of cutout alpha)
        alpha = cutout.split()[-1]
        ambient_shadow = Image.new('RGBA', cutout.size, (30, 30, 35, 0))
        alpha_faint = alpha.point(lambda p: int(p * 0.28))
        ambient_shadow.putalpha(alpha_faint)
        ambient_blurred = ambient_shadow.filter(ImageFilter.GaussianBlur(radius=max(8, int(cutout_w * 0.03))))

        offset_x = pos_x + int(cutout_w * 0.015)
        offset_y = pos_y + int(cutout_h * 0.025)
        shadow_layer.alpha_composite(ambient_blurred, (offset_x, offset_y))

        return shadow_layer

    # -------------------------------------------------------------------------
    # Procedural Background Renderers
    # -------------------------------------------------------------------------
    @classmethod
    def _render_lifestyle_wood(cls, width: int, height: int) -> Image.Image:
        """Renders warm natural wooden studio desk surface."""
        base_color = (205, 170, 135)
        bg = Image.new('RGBA', (width, height), (*base_color, 255))
        draw = ImageDraw.Draw(bg)

        np.random.seed(42)
        plank_h = max(60, height // 10)
        for y in range(0, height, plank_h):
            draw.line([(0, y), (width, y)], fill=(150, 115, 85, 140), width=2)
            for _ in range(12):
                streak_y = y + np.random.randint(2, max(3, plank_h - 2))
                alpha = np.random.randint(15, 45)
                shade = int(np.random.choice([160, 225]))
                draw.line([(0, streak_y), (width, streak_y)], fill=(shade, shade - 30, shade - 60, alpha), width=1)

        return bg.filter(ImageFilter.GaussianBlur(radius=1.5))

    @classmethod
    def _render_lifestyle_marble(cls, width: int, height: int) -> Image.Image:
        """Renders elegant white marble tabletop with subtle grey veining."""
        bg = Image.new('RGBA', (width, height), (246, 246, 248, 255))
        draw = ImageDraw.Draw(bg)

        np.random.seed(101)
        for _ in range(6):
            cur_x = np.random.randint(0, width)
            cur_y = 0
            points = [(cur_x, cur_y)]
            while cur_y < height:
                cur_x += np.random.randint(-40, 50)
                cur_y += np.random.randint(30, 80)
                points.append((cur_x, cur_y))
            draw.line(points, fill=(200, 205, 212, 60), width=int(np.random.randint(2, 6)))

        return bg.filter(ImageFilter.GaussianBlur(radius=8.0))

    @classmethod
    def _render_studio_gradient(cls, width: int, height: int) -> Image.Image:
        """Renders modern subtle vertical studio spotlight gradient."""
        top_rgb = (255, 255, 255)
        bot_rgb = (215, 222, 228)

        gradient_arr = np.zeros((height, width, 4), dtype=np.uint8)
        for y in range(height):
            ratio = y / max(height - 1, 1)
            r = int(top_rgb[0] * (1 - ratio) + bot_rgb[0] * ratio)
            g = int(top_rgb[1] * (1 - ratio) + bot_rgb[1] * ratio)
            b = int(top_rgb[2] * (1 - ratio) + bot_rgb[2] * ratio)
            gradient_arr[y, :, 0] = r
            gradient_arr[y, :, 1] = g
            gradient_arr[y, :, 2] = b
            gradient_arr[y, :, 3] = 255

        return Image.fromarray(gradient_arr, mode='RGBA')

    # -------------------------------------------------------------------------
    # Helper Utilities
    # -------------------------------------------------------------------------
    @staticmethod
    def _pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
        """Converts PIL RGB image to OpenCV BGR numpy array."""
        rgb = np.array(pil_img)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _cv2_to_pil(cv_img: np.ndarray) -> Image.Image:
        """Converts OpenCV BGR numpy array to PIL RGB image."""
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    @staticmethod
    def _hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
        """Converts hex color string to RGB tuple."""
        clean = hex_code.lstrip('#')
        if len(clean) == 3:
            clean = ''.join(c * 2 for c in clean)
        if len(clean) != 6:
            return 255, 255, 255
        try:
            return tuple(int(clean[i:i+2], 16) for i in (0, 2, 4))
        except ValueError:
            return 255, 255, 255

    @staticmethod
    def _get_srgb_profile() -> Optional[bytes]:
        """Returns standard sRGB ICC profile bytes if Pillow has it."""
        try:
            from PIL import ImageCms
            profile = ImageCms.createProfile('sRGB')
            return ImageCms.ImageCmsProfile(profile).tobytes()
        except Exception:
            return None
