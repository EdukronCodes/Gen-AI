from django.db import models
from django.utils import timezone

class Product(models.Model):
    name = models.CharField(max_length=200)
    category = models.CharField(max_length=100)
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    class Meta:
        ordering = ['-created_at']

class Review(models.Model):
    SENTIMENT_CHOICES = [
        ('positive', 'Positive'),
        ('negative', 'Negative'),
        ('neutral', 'Neutral'),
    ]

    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='reviews')
    title = models.CharField(max_length=200)
    content = models.TextField()
    rating = models.IntegerField(choices=[(i, i) for i in range(1, 6)])
    sentiment = models.CharField(max_length=10, choices=SENTIMENT_CHOICES)
    sentiment_score = models.FloatField()  # Score between -1 and 1
    keywords = models.JSONField(default=list)  # Store extracted keywords
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.title} - {self.product.name}"

    class Meta:
        ordering = ['-created_at']

class SentimentAnalysis(models.Model):
    review = models.OneToOneField(Review, on_delete=models.CASCADE, related_name='analysis')
    positive_score = models.FloatField()
    negative_score = models.FloatField()
    neutral_score = models.FloatField()
    compound_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Analysis for {self.review.title}"

    class Meta:
        ordering = ['-created_at'] 