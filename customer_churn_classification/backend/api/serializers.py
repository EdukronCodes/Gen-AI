from rest_framework import serializers
from ..models.customer import Customer
from ..models.transaction import Transaction, ProductCategory, CustomerCategoryPreference

class CustomerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Customer
        fields = [
            'id', 'customer_id', 'first_name', 'last_name', 'email',
            'phone', 'date_joined', 'last_purchase_date', 'total_purchases',
            'total_spent', 'average_order_value', 'days_since_last_purchase',
            'churn_probability', 'is_churned', 'purchase_frequency',
            'customer_lifetime_value', 'website_visits', 'support_tickets',
            'product_returns', 'age', 'gender', 'location'
        ]
        read_only_fields = [
            'churn_probability', 'is_churned', 'average_order_value',
            'customer_lifetime_value', 'days_since_last_purchase'
        ]

class TransactionSerializer(serializers.ModelSerializer):
    customer_name = serializers.SerializerMethodField()
    
    class Meta:
        model = Transaction
        fields = [
            'id', 'transaction_id', 'customer', 'customer_name',
            'amount', 'status', 'transaction_date', 'payment_method',
            'product_categories'
        ]
    
    def get_customer_name(self, obj):
        return f"{obj.customer.first_name} {obj.customer.last_name}"

class ProductCategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = ProductCategory
        fields = ['id', 'name', 'description']

class CustomerCategoryPreferenceSerializer(serializers.ModelSerializer):
    category_name = serializers.CharField(source='category.name')
    
    class Meta:
        model = CustomerCategoryPreference
        fields = [
            'id', 'customer', 'category', 'category_name',
            'purchase_count', 'last_purchase_date'
        ]

class ChurnPredictionSerializer(serializers.Serializer):
    customer_id = serializers.CharField()
    churn_probability = serializers.FloatField()
    is_churned = serializers.BooleanField()
    risk_level = serializers.SerializerMethodField()
    
    def get_risk_level(self, obj):
        probability = obj['churn_probability']
        if probability >= 0.7:
            return 'High'
        elif probability >= 0.4:
            return 'Medium'
        else:
            return 'Low'

class ChurnStatsSerializer(serializers.Serializer):
    total_customers = serializers.IntegerField()
    churned_customers = serializers.IntegerField()
    churn_rate = serializers.FloatField()
    average_churn_probability = serializers.FloatField()
    at_risk_customers = serializers.IntegerField()

class FeatureImportanceSerializer(serializers.Serializer):
    feature = serializers.CharField()
    importance = serializers.FloatField() 