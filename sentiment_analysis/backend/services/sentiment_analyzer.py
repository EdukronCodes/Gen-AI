import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import string

class SentimentAnalyzer:
    def __init__(self):
        # Download required NLTK data
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        self.analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))

    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of a given text using VADER.
        Returns a dictionary with sentiment scores and classification.
        """
        # Get sentiment scores
        scores = self.analyzer.polarity_scores(text)
        
        # Determine sentiment category
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'sentiment_score': scores['compound'],
            'positive_score': scores['pos'],
            'negative_score': scores['neg'],
            'neutral_score': scores['neu'],
            'compound_score': scores['compound']
        }

    def extract_keywords(self, text, num_keywords=5):
        """
        Extract the most important keywords from the text.
        """
        # Tokenize and clean the text
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word not in self.stop_words and word not in string.punctuation]
        
        # Get frequency distribution
        fdist = FreqDist(tokens)
        
        # Return top keywords
        return [word for word, freq in fdist.most_common(num_keywords)]

    def analyze_review(self, title, content):
        """
        Perform complete analysis of a review including sentiment and keywords.
        """
        # Combine title and content for analysis
        full_text = f"{title} {content}"
        
        # Get sentiment analysis
        sentiment_results = self.analyze_sentiment(full_text)
        
        # Extract keywords
        keywords = self.extract_keywords(full_text)
        
        return {
            **sentiment_results,
            'keywords': keywords
        }

    def get_sentiment_stats(self, reviews):
        """
        Calculate sentiment statistics for a collection of reviews.
        """
        total_reviews = len(reviews)
        if total_reviews == 0:
            return {
                'total_reviews': 0,
                'sentiment_distribution': {
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0
                },
                'average_sentiment_score': 0
            }

        sentiment_counts = {
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }
        
        total_sentiment_score = 0
        
        for review in reviews:
            sentiment_counts[review.sentiment] += 1
            total_sentiment_score += review.sentiment_score
        
        return {
            'total_reviews': total_reviews,
            'sentiment_distribution': {
                sentiment: (count / total_reviews) * 100
                for sentiment, count in sentiment_counts.items()
            },
            'average_sentiment_score': total_sentiment_score / total_reviews
        }

    def get_category_stats(self, reviews):
        """
        Calculate sentiment statistics by product category.
        """
        category_stats = {}
        
        for review in reviews:
            category = review.product.category
            if category not in category_stats:
                category_stats[category] = {
                    'total_reviews': 0,
                    'sentiment_counts': {
                        'positive': 0,
                        'negative': 0,
                        'neutral': 0
                    },
                    'total_sentiment_score': 0
                }
            
            stats = category_stats[category]
            stats['total_reviews'] += 1
            stats['sentiment_counts'][review.sentiment] += 1
            stats['total_sentiment_score'] += review.sentiment_score
        
        # Calculate percentages and averages
        for category, stats in category_stats.items():
            total = stats['total_reviews']
            stats['sentiment_distribution'] = {
                sentiment: (count / total) * 100
                for sentiment, count in stats['sentiment_counts'].items()
            }
            stats['average_sentiment_score'] = stats['total_sentiment_score'] / total
            del stats['sentiment_counts']
            del stats['total_sentiment_score']
        
        return category_stats 