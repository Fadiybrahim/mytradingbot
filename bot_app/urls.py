# bot_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.main_view, name='main'),
    path('stock-search/', views.stock_search_view, name='stock_search'),
    path('configure/', views.configure_bot_view, name='configure_bot_new'),
    path('configure/<int:config_id>/', views.configure_bot_view, name='configure_bot'),
    path('api/bot-status/<int:config_id>/', views.get_bot_status_json, name='get_bot_status_json'),
    path('bots/', views.list_configured_bots, name='list_configured_bots'),
    path('toggle_trading/<int:config_id>/', views.toggle_trading_status, name='toggle_trading_status'),
    path('manual_trade/<int:config_id>/<str:order_type>/', views.manual_trade_view, name='manual_trade'),
]
