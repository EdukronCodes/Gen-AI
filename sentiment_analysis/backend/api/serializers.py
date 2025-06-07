from rest_framework import serializers
from models.review import Product, Review, SentimentAnalysis

class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = ['id', 'name', 'category', 'description', 'created_at', 'updated_at']

class SentimentAnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = SentimentAnalysis
        fields = ['positive_score', 'negative_score', 'neutral_score', 'compound_score', 'created_at']

class ReviewSerializer(serializers.ModelSerializer):
    product = ProductSerializer(read_only=True)
    analysis = SentimentAnalysisSerializer(read_only=True)
    
    class Meta:
        model = Review
        fields = [
            'id', 'product', 'title', 'content', 'rating',
            'sentiment', 'sentiment_score', 'keywords',
            'analysis', 'created_at', 'updated_at'
        ]
        read_only_fields = ['sentiment', 'sentiment_score', 'keywords', 'analysis']

class ReviewCreateSerializer(serializers.ModelSerializer):
    product_id = serializers.IntegerField(write_only=True)
    
    class Meta:
        model = Review
        fields = ['product_id', 'title', 'content', 'rating']

    def create(self, validated_data):
        product_id = validated_data.pop('product_id')
        try:
            product = Product.objects.get(id=product_id)
        except Product.DoesNotExist:
            raise serializers.ValidationError({'product_id': 'Product not found'})
        
        # Get sentiment analyzer
        from services.sentiment_analyzer import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        
        # Analyze review
        analysis_results = analyzer.analyze_review(
            validated_data['title'],
            validated_data['content']
        )
        
        # Create review with sentiment analysis
        review = Review.objects.create(
            product=product,
            **validated_data,
            sentiment=analysis_results['sentiment'],
            sentiment_score=analysis_results['sentiment_score'],
            keywords=analysis_results['keywords']
        )
        
        # Create sentiment analysis
        SentimentAnalysis.objects.create(
            review=review,
            positive_score=analysis_results['positive_score'],
            negative_score=analysis_results['negative_score'],
            neutral_score=analysis_results['neutral_score'],
            compound_score=analysis_results['compound_score']
        )
        
        return review

class SentimentStatsSerializer(serializers.Serializer):
    total_reviews = serializers.IntegerField()
    sentiment_distribution = serializers.DictField(
        child=serializers.FloatField()
    )
    average_sentiment_score = serializers.FloatField()

class CategoryStatsSerializer(serializers.Serializer):
    total_reviews = serializers.IntegerField()
    sentiment_distribution = serializers.DictField(
        child=serializers.FloatField()
    )
    average_sentiment_score = serializers.FloatField() 