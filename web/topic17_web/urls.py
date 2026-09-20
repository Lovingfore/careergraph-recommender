"""项目级 URL 入口：根路径下的所有请求统一交给 topic17_app 路由。"""

from django.urls import include, path


urlpatterns = [
    # 应用路由包含首页、JSON API 以及两个 SVG 图谱资源。
    path("", include("web.topic17_app.urls")),
]
