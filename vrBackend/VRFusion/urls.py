"""VRFusion URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
import os

from django.conf.urls import url
from django.urls import re_path
from django.views.static import serve
from . import settings
from panorama import views

urlpatterns = [
    url('imageSynthesis', views.image_synthesis),
    url('save_img', views.save_img),
    url('fusion_image', views.fusion_image),
    url('test', views.test_interface),
    re_path(r'media/(?P<path>.*)$', serve, {'document_root': os.path.join(settings.BASE_DIR, 'static/media')}),
]
