# Product Photo Editor for Odoo (v18.0 / v19.0+)

**AI-Powered Product Photo Editing & Optimization Connector and Workflow Inside Odoo**

---

## 🌟 Overview

The **Product Photo Editor** (`product_editor`) module bridges raw product photography and marketplace-ready e-commerce imagery. Designed as a high-performance connector + workflow application inside Odoo, it allows merchants to transform raw, unedited supplier or smartphone product photos into standardized, high-converting catalog assets with a single click.

---

## 🚀 Key Features

### ✂️ 1. Multi-Provider Background Removal
- **Local Engine (Offline & Free)**: Powered by OpenCV GrabCut contour segmentation, edge-gradient saliency detection, morphological filtering, and Gaussian alpha edge feathering. Runs 100% offline without external API keys or subscriptions.
- **External AI Connectors**:
  - [remove.bg](https://www.remove.bg) API v1.0
  - [Stability AI Clipdrop](https://clipdrop.co) API
  - [Photoroom](https://www.photoroom.com) Segment API
  - **Custom AI Endpoint**: Self-hosted ONNX / Triton inference service or proprietary webhook.

### 📐 2. OpenCV Perspective Correction & Auto-Deskewing
- Automatic 4-corner polygon detection using bilateral filtering, Canny edge detection, and contour approximation (`cv2.approxPolyDP`).
- Euclidean distance calculation and 4-point homography perspective warp (`cv2.warpPerspective`).
- Automatic tilt angle detection (`cv2.minAreaRect`) and affine rotation deskewing to ensure products stand straight.

### 🎨 3. Color, Lighting & Sharpness Enhancement
- **Gray-World Auto White Balance**: Neutralizes artificial tungsten or fluorescent color casts.
- **CLAHE Adaptive Exposure**: Contrast Limited Adaptive Histogram Equalization in LAB color space to lift dark product shadows while preserving highlights.
- **Unsharp Mask Sharpening**: Crisp edge accentuation optimized for marketplace thumbnails.

### 🛒 4. Dimension Standardization & Background Composition
- **Dimension Presets**:
  - `2000 x 2000 px`: Amazon, Shopify, eBay, and Google Shopping standard.
  - `1000 x 1000 px`: Compact web catalog.
  - `1600 x 2000 px`: Fashion / Instagram (4:5).
  - `1920 x 1080 px`: Hero banners (16:9).
  - `Original`: Keep input aspect ratio.
- **Padding Control**: Configurable safety margins (default 8%) so products fill ~85% of the frame.
- **Background Styles**:
  - `Transparent PNG`: Preserves alpha channel.
  - `Pure White (#FFFFFF)`: Standard studio white.
  - `Studio Neutral Grey (#F4F4F4)`: Modern minimal aesthetic.
  - `Studio Soft Shadow`: Procedural dual-shadow generation (dark base contact ellipse + ambient blurred drop shadow).
  - `Lifestyle Wood`: Warm natural wooden tabletop surface.
  - `Lifestyle Marble`: White marble countertop with realistic veining.
  - `Lifestyle Gradient`: Subtle vertical studio spotlight gradient.
  - `Custom Hex Color`: Any brand color code.

### ⚡ 5. Seamless Odoo UI Integration
- **Product Form Smart Button**: Shows count of photo edits for the product.
- **"✨ AI Photo Editor" Header Action**: Instantly launches the interactive wizard from any product form.
- **Interactive Wizard with Live Preview**: Test presets and see the before/after preview before applying.
- **Bulk Batch Processing**:
  - Select multiple products in the Product List View and execute **"✨ Bulk AI Photo Optimization"**.
  - Background cron job (`product.photo.editor._cron_batch_process`) processes pending queues during off-peak hours.

### 🌐 6. REST API Endpoints
- `POST /api/v1/photo_editor/process`: Upload raw image (file or base64) and receive processed result + metadata.
- `GET /api/v1/photo_editor/presets`: Retrieve supported backgrounds, dimensions, and providers.
- `POST /api/v1/photo_editor/batch`: Enqueue a batch of product IDs for processing.
- `GET /api/v1/photo_editor/status/<job_id>`: Check job status and fetch resulting image.

---

## 📦 Directory Structure

```text
product_editor/
├── __init__.py
├── __manifest__.py
├── README.md
├── controllers/
│   ├── __init__.py
│   └── main.py                     # REST endpoints
├── models/
│   ├── __init__.py
│   ├── image_pipeline.py           # Core image engine (OpenCV, PIL, external APIs)
│   ├── photo_editor.py             # product.photo.editor model
│   ├── product_template.py         # product.template extension
│   └── res_config_settings.py      # System settings & API keys
├── wizard/
│   ├── __init__.py
│   └── photo_editor_wizard.py      # Interactive modal with live preview
├── views/
│   ├── photo_editor_views.xml      # List, Form, Kanban & Search views (<list> tag)
│   ├── photo_editor_wizard_views.xml # Wizard split-layout form
│   ├── product_template_views.xml  # Smart button & header action
│   ├── res_config_settings_views.xml # Settings UI
│   └── photo_editor_menus.xml      # Navigation menus
├── data/
│   ├── photo_editor_data.xml       # Sequences, parameters, and bulk server actions
│   └── ir_cron_data.xml            # Scheduled cron batch job
├── security/
│   ├── photo_editor_security.xml   # User & Manager groups, multi-company rules
│   └── ir.model.access.csv         # Access control permissions
├── static/
│   └── description/
│       └── index.html              # App store documentation
└── tests/
    ├── __init__.py
    ├── test_image_pipeline.py      # Deep pipeline unit & edge-case tests
    ├── test_photo_editor.py        # Odoo model & workflow test cases
    └── test_controllers.py         # REST controller integration tests
```

---

## ⚙️ Installation & Configuration

1. Copy the `product_editor` folder to your Odoo custom addons directory.
2. Install Python dependencies:
   ```bash
   pip install opencv-python-headless pillow numpy requests
   ```
3. Restart Odoo server and update Apps list (`Apps` -> `Update Apps List`).
4. Search for **Product Photo Editor** and click **Install**.
5. Navigate to **Photo Editor** -> **Configuration** -> **Settings**:
   - Choose default AI provider (e.g. `Local Engine`, `remove.bg`, etc.).
   - Enter API keys if using external providers.
   - Set default background style and canvas dimensions.

---

## 🧪 Testing

Run standalone pipeline unit tests using pytest:
```bash
pytest tests/test_image_pipeline.py -v
```
Run within Odoo test suite:
```bash
odoo-bin -c odoo.conf -u product_editor --test-enable --stop-after-init
```

---

## 📄 License
LGPL-3.0
