from django.urls import path
from . import views

urlpatterns = [
    # Home page showing CKD awareness and navigation
    path('', views.home, name='home'),

    # Prediction page (manual + CSV + model selection)
    path('predict/', views.predict_ckd, name='predict'),
]
