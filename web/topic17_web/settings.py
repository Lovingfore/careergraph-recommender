"""CareerGraph 本地演示站点的 Django 配置。

本文件只负责 Web 框架配置；CSV、SQLite 和推荐数据的实际读取逻辑集中在
``src`` 业务模块中，避免把业务计算散落到 Django 设置层。
"""

from pathlib import Path
import os


# 基础路径：所有数据、模型产物和模板均从项目根目录解析，避免依赖启动目录。
BASE_DIR = Path(__file__).resolve().parents[2]

# 安全与网络：默认只允许本机访问。局域网课堂演示可显式设置
# CAREERGRAPH_ALLOW_NETWORK=1；该开关不应直接用于公网生产部署。
SECRET_KEY = "topic17-local-demo-key"
DEBUG = True
ALLOWED_HOSTS = ["127.0.0.1", "localhost", "testserver"]
if os.environ.get("CAREERGRAPH_ALLOW_NETWORK", "").strip().lower() in {"1", "true", "yes"}:
    ALLOWED_HOSTS = ["*"]
ROOT_URLCONF = "web.topic17_web.urls"

# 中间件只保留安全、通用请求处理和 CSRF 防护所需的最小集合。
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
]

# 应用：contenttypes/staticfiles 为 Django 基础能力，topic17_app 提供全部页面与 API。
INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
    "web.topic17_app",
]

# 模板：既支持显式模板目录，也允许 Django 从已安装应用中查找模板。
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "web" / "topic17_app" / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": []},
    }
]
WSGI_APPLICATION = "web.topic17_web.wsgi.application"

# SQLite 是 Web 查询层；离线流水线仍以 data/clean 下的 CSV 作为可审计交换层。
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "artifacts" / "topic17.sqlite3",
    }
}

# 静态资源与模型默认主键配置。
STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# 上传限制：单个简历最多 1 MB，整个请求体最多 2 MB；视图层还会二次校验文件大小。
FILE_UPLOAD_MAX_MEMORY_SIZE = 1024 * 1024
DATA_UPLOAD_MAX_MEMORY_SIZE = 2 * 1024 * 1024
