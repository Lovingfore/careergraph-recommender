from django.urls import path

from .views import index, summary_api


urlpatterns = [
    path("", index, name="index"),
    path("api/summary/", summary_api, name="summary_api"),
]
