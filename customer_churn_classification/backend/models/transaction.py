from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
from .customer import Customer

class Transaction(models.Model):
    TRANSACTION_STATUS = (
        ('completed', 'Completed'),
        ('pending', 'Pending'),
        ('failed', 'Failed'),
        ('refunded', 'Refunded'),
    )
    
    transaction_id = models.CharField(max_length=50, unique=True)
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name='transactions')
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    status = models.CharField(max_length=20, choices=TRANSACTION_STATUS, default='pending')
    transaction_date = models.DateTimeField(auto_now_add=True)
    payment_method = models.CharField(max_length=50)
    product_categories = models.JSONField(default=list)  # List of product categories in the transaction
    
    class Meta:
        ordering = ['-transaction_date']
        
    def __str__(self):
        return f"Transaction {self.transaction_id} - {self.customer} - {self.amount}"

@receiver(post_save, sender=Transaction)
def update_customer_metrics(sender, instance, created, **kwargs):
    """Update customer metrics when a new transaction is created"""
    if created and instance.status == 'completed':
        customer = instance.customer
        
        # Update total purchases and amount
        customer.total_purchases += 1
        customer.total_spent += instance.amount
        
        # Update last purchase date
        customer.last_purchase_date = instance.transaction_date
        
        # Calculate purchase frequency (purchases per month)
        months_since_joined = (instance.transaction_date - customer.date_joined).days / 30
        if months_since_joined > 0:
            customer.purchase_frequency = customer.total_purchases / months_since_joined
        
        # Update average order value
        customer.average_order_value = customer.total_spent / customer.total_purchases
        
        # Calculate customer lifetime value
        customer.calculate_lifetime_value()
        
        # Update churn metrics
        customer.update_churn_metrics()
        
        customer.save()

class ProductCategory(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    
    def __str__(self):
        return self.name

class CustomerCategoryPreference(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name='category_preferences')
    category = models.ForeignKey(ProductCategory, on_delete=models.CASCADE)
    purchase_count = models.IntegerField(default=0)
    last_purchase_date = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        unique_together = ('customer', 'category')
        
    def __str__(self):
        return f"{self.customer} - {self.category} ({self.purchase_count})" 