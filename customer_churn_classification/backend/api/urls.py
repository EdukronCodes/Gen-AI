from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import CustomerViewSet, TransactionViewSet, ProductCategoryViewSet

router = DefaultRouter()
router.register(r'customers', CustomerViewSet)
router.register(r'transactions', TransactionViewSet)
router.register(r'product-categories', ProductCategoryViewSet)

urlpatterns = [
    path('', include(router.urls)),
] 