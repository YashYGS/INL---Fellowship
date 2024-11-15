"""INLwebapp2 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
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
from app2 import views

urlpatterns = [
    path('admin/', admin.site.urls),

    path('protected/', views.protected_view, name='protected_view'),
    path('logout/', views.logout_view, name='logout_view'),

    path('welcome/', views.welcome, name='welcome'),
    path('choose_task/', views.choose_task, name='choose_task'),
    path('plot/<str:csv_file_path>/', views.plot, name='plot'),
    path('mask_options/', views.mask_options, name='mask_options'),
    path('train/<str:csv_file_path>/', views.train, name='train'),
    path('download_default_file/', views.download_default_file, name='download_default_file'),
    path('download_file/', views.download_file, name='download_file'),
    path('download_label/', views.download_label, name='download_label'),

    path('about_us/', views.about_us, name='about_us'),
    path('protected/', views.protected_view, name='protected_view'),
]
