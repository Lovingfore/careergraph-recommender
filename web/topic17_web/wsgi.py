"""WSGI 应用入口，供生产级 WSGI 服务器加载 Django 应用对象。"""

import os

from django.core.wsgi import get_wsgi_application


# 外部服务器若未提供设置模块，则默认使用本项目配置。
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web.topic17_web.settings")
application = get_wsgi_application()
