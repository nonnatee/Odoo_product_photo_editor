import sys
import types

if 'odoo' not in sys.modules:
    try:
        import odoo
    except ImportError:
        class _Dummy:
            def __init__(self, *a, **k):
                pass
            def __call__(self, *a, **k):
                return self
            def __getattr__(self, name):
                return _Dummy()
            def __getitem__(self, name):
                return _Dummy()

        _odoo = types.ModuleType('odoo')
        _odoo.models = types.ModuleType('odoo.models')
        _odoo.models.Model = _Dummy
        _odoo.models.TransientModel = _Dummy
        _odoo.models.AbstractModel = _Dummy

        _odoo.fields = types.ModuleType('odoo.fields')
        for fld in [
            'Char', 'Text', 'Integer', 'Float', 'Boolean', 'Binary',
            'Selection', 'Many2one', 'One2many', 'Many2many', 'Datetime', 'Date', 'Image'
        ]:
            setattr(_odoo.fields, fld, _Dummy)

        _odoo.api = types.ModuleType('odoo.api')
        _odoo.api.depends = lambda *a, **k: lambda f: f
        _odoo.api.constrains = lambda *a, **k: lambda f: f
        _odoo.api.model_create_multi = lambda f: f
        _odoo.api.model = lambda f: f
        _odoo.api.onchange = lambda *a, **k: lambda f: f
        _odoo._ = lambda s: s

        _odoo.exceptions = types.ModuleType('odoo.exceptions')
        class UserError(Exception):
            pass
        class ValidationError(Exception):
            pass
        class AccessError(Exception):
            pass
        _odoo.exceptions.UserError = UserError
        _odoo.exceptions.ValidationError = ValidationError
        _odoo.exceptions.AccessError = AccessError

        _odoo_http = types.ModuleType('odoo.http')
        _odoo_http.request = None
        _odoo_http.Controller = object
        _odoo_http.route = lambda *a, **k: lambda f: f

        class MockResponse:
            def __init__(self, response=None, status=200, headers=None, mimetype=None, content_type=None):
                self.data = response.encode('utf-8') if isinstance(response, str) else (response or b'')
                self.status_code = status
                self.headers = headers or {}
                self.mimetype = mimetype or 'application/json'

        _odoo_http.Response = MockResponse
        _odoo.http = _odoo_http

        sys.modules['odoo'] = _odoo
        sys.modules['odoo.models'] = _odoo.models
        sys.modules['odoo.fields'] = _odoo.fields
        sys.modules['odoo.api'] = _odoo.api
        sys.modules['odoo.exceptions'] = _odoo.exceptions
        sys.modules['odoo.http'] = _odoo_http

from . import test_image_pipeline
from . import test_photo_editor
from . import test_controllers
