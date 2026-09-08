from django.urls import include, path


urlpatterns = [
    path("", include("web.topic17_app.urls")),
]
