from django.urls import include, path

urlpatterns = [
    path("", include("deepfake_images.urls")),
]
