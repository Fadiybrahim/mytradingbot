from django.urls import path
from . import views

urlpatterns = [
    path('', views.portfolio_view, name='portfolio_detail'),
    path('generate-analysis/', views.generate_analysis_view, name='generate_analysis'),
]
