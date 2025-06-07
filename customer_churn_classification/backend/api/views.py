from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.db.models import Avg, Count
from ..models.customer import Customer
from ..models.transaction import Transaction, ProductCategory
from ..services.churn_predictor import ChurnPredictor, update_customer_churn_probabilities
from .serializers import (
    CustomerSerializer,
    TransactionSerializer,
    ProductCategorySerializer,
    ChurnPredictionSerializer
)

class CustomerViewSet(viewsets.ModelViewSet):
    queryset = Customer.objects.all()
    serializer_class = CustomerSerializer
    permission_classes = [IsAuthenticated]
    
    @action(detail=True, methods=['post'])
    def predict_churn(self, request, pk=None):
        """Predict churn probability for a specific customer"""
        customer = self.get_object()
        predictor = ChurnPredictor()
        churn_probability = predictor.predict_churn(customer)
        
        # Update customer's churn probability
        customer.churn_probability = churn_probability
        customer.is_churned = churn_probability > 0.5
        customer.save()
        
        return Response({
            'customer_id': customer.customer_id,
            'churn_probability': churn_probability,
            'is_churned': customer.is_churned
        })
    
    @action(detail=False, methods=['get'])
    def churn_stats(self, request):
        """Get churn statistics for all customers"""
        total_customers = Customer.objects.count()
        churned_customers = Customer.objects.filter(is_churned=True).count()
        avg_churn_probability = Customer.objects.aggregate(
            avg_prob=Avg('churn_probability')
        )['avg_prob'] or 0
        
        # Get customers at risk (high churn probability but not churned)
        at_risk_customers = Customer.objects.filter(
            churn_probability__gt=0.7,
            is_churned=False
        ).count()
        
        return Response({
            'total_customers': total_customers,
            'churned_customers': churned_customers,
            'churn_rate': (churned_customers / total_customers) if total_customers > 0 else 0,
            'average_churn_probability': avg_churn_probability,
            'at_risk_customers': at_risk_customers
        })
    
    @action(detail=False, methods=['post'])
    def update_all_churn_probabilities(self, request):
        """Update churn probabilities for all customers"""
        update_customer_churn_probabilities()
        return Response({'status': 'Churn probabilities updated successfully'})
    
    @action(detail=False, methods=['get'])
    def feature_importance(self, request):
        """Get feature importance from the churn prediction model"""
        predictor = ChurnPredictor()
        importance = predictor.get_feature_importance()
        return Response(importance)

class TransactionViewSet(viewsets.ModelViewSet):
    queryset = Transaction.objects.all()
    serializer_class = TransactionSerializer
    permission_classes = [IsAuthenticated]
    
    @action(detail=False, methods=['get'])
    def customer_transactions(self, request):
        """Get transactions for a specific customer"""
        customer_id = request.query_params.get('customer_id')
        if not customer_id:
            return Response(
                {'error': 'customer_id parameter is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        transactions = Transaction.objects.filter(
            customer__customer_id=customer_id
        ).order_by('-transaction_date')
        
        serializer = self.get_serializer(transactions, many=True)
        return Response(serializer.data)

class ProductCategoryViewSet(viewsets.ModelViewSet):
    queryset = ProductCategory.objects.all()
    serializer_class = ProductCategorySerializer
    permission_classes = [IsAuthenticated]
    
    @action(detail=False, methods=['get'])
    def customer_preferences(self, request):
        """Get product category preferences for a specific customer"""
        customer_id = request.query_params.get('customer_id')
        if not customer_id:
            return Response(
                {'error': 'customer_id parameter is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        preferences = ProductCategory.objects.filter(
            customercategorypreference__customer__customer_id=customer_id
        ).annotate(
            purchase_count=Count('customercategorypreference')
        ).order_by('-purchase_count')
        
        serializer = self.get_serializer(preferences, many=True)
        return Response(serializer.data) 