"""
URL configuration for Detection App

Project Code: GP_IT_16
Image Forgery Detection System
"""

from django.urls import path
from . import views

urlpatterns = [
    # Main pages
    path('', views.home_view, name='home'),
    path('about/', views.about_view, name='about'),
    path('how-it-works/', views.how_it_works_view, name='how_it_works'),
    
    # Image upload and analysis
    path('upload/', views.upload_image_view, name='upload_image'),
    path('analyze/<int:image_id>/', views.analyze_image_view, name='analyze_image'),
    path('result/<int:result_id>/', views.analysis_result_view, name='analysis_result'),
    
    # Gallery and comparison
    path('gallery/', views.gallery_view, name='gallery'),
    
    # API endpoints
    path('api/analyze/', views.api_analyze_view, name='api_analyze'),
    
    # Reports
    path('report/<int:result_id>/download/', views.download_report_view, name='download_report'),
    
    # Management
    path('analysis/<int:result_id>/delete/', views.delete_analysis_view, name='delete_analysis'),
]