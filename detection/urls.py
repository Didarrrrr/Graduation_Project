from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('about/', views.about_view, name='about'),
    path('how-it-works/', views.how_it_works_view, name='how_it_works'),
    
    path('upload/', views.upload_image_view, name='upload_image'),
    path('analyze/<int:image_id>/', views.analyze_image_view, name='analyze_image'),
    path('result/<int:result_id>/', views.analysis_result_view, name='analysis_result'),
    
    path('gallery/', views.gallery_view, name='gallery'),
    
    
    path('report/<int:result_id>/download/', views.download_report_view, name='download_report'),
    
    path('analysis/<int:result_id>/delete/', views.delete_analysis_view, name='delete_analysis'),
]