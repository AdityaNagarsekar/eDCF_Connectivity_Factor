"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
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
from django.contrib import admin
from django.urls import path
from DriverInterface.views import process_data, index, download_zip, delete_zip
from django.http import HttpResponse

def empty_favicon(request):
    return HttpResponse(status=204)  # No Content

urlpatterns = [
    path('admin/', admin.site.urls),
    path('process_data/', process_data, name='process_data'), 
    path('', index, name='index'), 
    path('favicon.ico', empty_favicon),
    path('download_zip/', download_zip, name='download_zip'), 
    path('delete_zip/', delete_zip, name='delete_zip'), 
]
