# -*- coding: utf-8 -*-
{
    'name': 'Product Photo Editor',
    'version': '19.0.1.0.0',
    'summary': 'AI-Powered Product Photo Editing & Optimization Connector and Workflow',
    'sequence': 10,
    'description': """
Product Photo Editor for Odoo
=============================
Treats product photo editing as a high-performance connector + workflow application inside Odoo.

Key Features:
-------------
* **Background Removal**: Intelligent offline local edge-aware GrabCut segmentation and external AI API connectors (remove.bg, Stability AI Clipdrop, Photoroom, Custom Webhooks).
* **Perspective Correction**: OpenCV quad detection and automatic leveling / deskewing.
* **Color & Lighting Enhancement**: Gray-World auto white balance, LAB-space CLAHE adaptive exposure, and unsharp masking.
* **Dimension Standardization**: Resize and center product cutouts onto marketplace-compliant canvases (e.g. 2000x2000 Amazon standard, 1000x1000 web catalog, 1600x2000 fashion 4:5, 1920x1080 hero banner).
* **Realistic Grounding Shadows**: Contact shadow and soft ambient drop shadows to eliminate floating cutouts.
* **Procedural Backgrounds**: Transparent PNG, Pure White, Studio Neutral Grey, Lifestyle Warm Wood, Elegant Marble, Studio Spotlight Gradients, or Custom Hex Color.
* **Interactive UI Wizard**: Upload raw photos, test presets with live side-by-side preview, and apply directly to the product catalog with one click.
* **Automated Batch Processing**: Background cron job and list-view multi-record action for bulk product catalogs.
* **REST API Endpoints**: Full API integration for automated external pipelines.
    """,
    'category': 'Website/eCommerce',
    'author': 'Nonnatee Kanjana',
    'website': 'https://github.com/nonnatee/Odoo_product_photo_editor',
    'license': 'LGPL-3',
    'depends': [
        'base',
        'product',
        'web',
    ],
    'data': [
        'security/photo_editor_security.xml',
        'security/ir.model.access.csv',
        'data/photo_editor_data.xml',
        'data/ir_cron_data.xml',
        'views/photo_editor_views.xml',
        'views/photo_editor_wizard_views.xml',
        'views/product_template_views.xml',
        'views/res_config_settings_views.xml',
        'views/photo_editor_menus.xml',
    ],
    'external_dependencies': {
        'python': ['cv2', 'PIL', 'numpy', 'requests', 'rembg'],
    },
    'installable': True,
    'application': True,
    'auto_install': False,
}
