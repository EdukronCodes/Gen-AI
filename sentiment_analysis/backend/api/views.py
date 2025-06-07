from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.db.models import Avg
from models.review import Product, Review
from .serializers import (
    ProductSerializer,
    ReviewSerializer,
    ReviewCreateSerializer,
    SentimentStatsSerializer,
    CategoryStatsSerializer
)
from services.sentiment_analyzer import SentimentAnalyzer

class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer

class ReviewViewSet(viewsets.ModelViewSet):
    queryset = Review.objects.all()
    
    def get_serializer_class(self):
        if self.action == 'create':
            return ReviewCreateSerializer
        return ReviewSerializer
    
    def get_queryset(self):
        queryset = Review.objects.all()
        
        # Filter by product
        product_id = self.request.query_params.get('product_id')
        if product_id:
            queryset = queryset.filter(product_id=product_id)
        
        # Filter by sentiment
        sentiment = self.request.query_params.get('sentiment')
        if sentiment:
            queryset = queryset.filter(sentiment=sentiment)
        
        # Filter by rating
        min_rating = self.request.query_params.get('min_rating')
        if min_rating:
            queryset = queryset.filter(rating__gte=min_rating)
        
        # Filter by date range
        start_date = self.request.query_params.get('start_date')
        if start_date:
            queryset = queryset.filter(created_at__gte=start_date)
        
        end_date = self.request.query_params.get('end_date')
        if end_date:
            queryset = queryset.filter(created_at__lte=end_date)
        
        return queryset.select_related('product', 'analysis')
    
    @action(detail=False, methods=['get'])
    def sentiment_stats(self, request):
        reviews = self.get_queryset()
        analyzer = SentimentAnalyzer()
        stats = analyzer.get_sentiment_stats(reviews)
        serializer = SentimentStatsSerializer(stats)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def category_stats(self, request):
        reviews = self.get_queryset()
        analyzer = SentimentAnalyzer()
        stats = analyzer.get_category_stats(reviews)
        return Response(stats)
    
    @action(detail=False, methods=['get'])
    def keywords(self, request):
        reviews = self.get_queryset()
        all_keywords = []
        for review in reviews:
            all_keywords.extend(review.keywords)
        
        # Count keyword frequency
        keyword_freq = {}
        for keyword in all_keywords:
            keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
        
        # Sort by frequency
        sorted_keywords = sorted(
            keyword_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return Response({
            'keywords': [{'word': word, 'frequency': freq} for word, freq in sorted_keywords[:20]]
        })
    
    @action(detail=True, methods=['post'])
    def reanalyze(self, request, pk=None):
        review = self.get_object()
        analyzer = SentimentAnalyzer()
        
        # Reanalyze the review
        analysis_results = analyzer.analyze_review(
            review.title,
            review.content
        )
        
        # Update review
        review.sentiment = analysis_results['sentiment']
        review.sentiment_score = analysis_results['sentiment_score']
        review.keywords = analysis_results['keywords']
        review.save()
        
        # Update sentiment analysis
        review.analysis.positive_score = analysis_results['positive_score']
        review.analysis.negative_score = analysis_results['negative_score']
        review.analysis.neutral_score = analysis_results['neutral_score']
        review.analysis.compound_score = analysis_results['compound_score']
        review.analysis.save()
        
        serializer = self.get_serializer(review)
        return Response(serializer.data) 