"""CareerGraph Web 应用的 Django 元数据配置。"""

from django.apps import AppConfig


class Topic17AppConfig(AppConfig):
    """声明应用导入路径、默认主键类型和后台显示名称。"""

    default_auto_field = "django.db.models.BigAutoField"
    name = "web.topic17_app"
    verbose_name = "Topic 17 demo"
