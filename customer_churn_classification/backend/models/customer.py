from django.db import models
from django.contrib.auth.models import User

class Customer(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, null=True, blank=True)
    customer_id = models.CharField(max_length=50, unique=True)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=20, null=True, blank=True)
    date_joined = models.DateTimeField(auto_now_add=True)
    last_purchase_date = models.DateTimeField(null=True, blank=True)
    total_purchases = models.IntegerField(default=0)
    total_spent = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    average_order_value = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    days_since_last_purchase = models.IntegerField(default=0)
    churn_probability = models.FloatField(default=0)
    is_churned = models.BooleanField(default=False)
    
    # Customer behavior metrics
    purchase_frequency = models.FloatField(default=0)  # Purchases per month
    customer_lifetime_value = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    website_visits = models.IntegerField(default=0)
    support_tickets = models.IntegerField(default=0)
    product_returns = models.IntegerField(default=0)
    
    # Demographic information
    age = models.IntegerField(null=True, blank=True)
    gender = models.CharField(max_length=10, null=True, blank=True)
    location = models.CharField(max_length=100, null=True, blank=True)
    
    class Meta:
        ordering = ['-date_joined']
        
    def __str__(self):
        return f"{self.first_name} {self.last_name} ({self.customer_id})"
    
    def update_churn_metrics(self):
        """Update customer metrics for churn prediction"""
        from django.utils import timezone
        if self.last_purchase_date:
            self.days_since_last_purchase = (timezone.now() - self.last_purchase_date).days
        if self.total_purchases > 0:
            self.average_order_value = self.total_spent / self.total_purchases
        self.save()
    
    def calculate_lifetime_value(self):
        """Calculate customer lifetime value"""
        if self.total_purchases > 0:
            self.customer_lifetime_value = self.total_spent / (self.total_purchases / 12)  # Annual value
            self.save() 