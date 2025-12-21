import re
import string
import nltk
import spacy

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords if not already
nltk.download("stopwords", quiet=True)

class TextPreprocessor:
    def __init__(self, 
                 lowercase=True, 
                 remove_punct=True, 
                 remove_numbers=True,
                 remove_extra_spaces=True,
                 remove_stopwords=True,
                 lemmatize=True,
                 stemming=False,
                 remove_urls=True,
                 remove_emails=True,
                 remove_mentions=True,
                 remove_hashtags=True,
                 remove_nonascii=True,
                 language="english"):
        """
        Initialize preprocessing options.
        """
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        self.remove_numbers = remove_numbers
        self.remove_extra_spaces = remove_extra_spaces
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stemming = stemming
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.remove_nonascii = remove_nonascii

        # Setup tools
        self.stop_words = set(stopwords.words(language)) if remove_stopwords else set()
        self.stemmer = PorterStemmer() if stemming else None
        self.nlp = spacy.load("en_core_web_sm") if lemmatize else None

    def to_lower(self, text):
        return text.lower()

    def remove_punctuation(self, text):
        return text.translate(str.maketrans("", "", string.punctuation))

    def remove_digits(self, text):
        return re.sub(r"\d+", "", text)

    def remove_spaces(self, text):
        return " ".join(text.split())

    def remove_urls_emails_mentions_hashtags(self, text):
        if self.remove_urls:
            text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        if self.remove_emails:
            text = re.sub(r"\S+@\S+", "", text)
        if self.remove_mentions:
            text = re.sub(r"@\w+", "", text)
        if self.remove_hashtags:
            text = re.sub(r"#\w+", "", text)
        return text

    def remove_non_ascii(self, text):
        return text.encode("ascii", "ignore").decode()

    def remove_stop_words(self, text):
        return " ".join([w for w in text.split() if w not in self.stop_words])

    def stem_text(self, text):
        return " ".join([self.stemmer.stem(w) for w in text.split()])

    def lemmatize_text(self, text):
        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc])

    def clean_text(self, text):
        """
        Apply all preprocessing steps in sequence.
        """
        if self.lowercase:
            text = self.to_lower(text)
        if self.remove_urls or self.remove_emails or self.remove_mentions or self.remove_hashtags:
            text = self.remove_urls_emails_mentions_hashtags(text)
        if self.remove_punct:
            text = self.remove_punctuation(text)
        if self.remove_numbers:
            text = self.remove_digits(text)
        if self.remove_nonascii:
            text = self.remove_non_ascii(text)
        if self.remove_stopwords:
            text = self.remove_stop_words(text)
        if self.stemming:
            text = self.stem_text(text)
        if self.lemmatize:
            text = self.lemmatize_text(text)
        if self.remove_extra_spaces:
            text = self.remove_spaces(text)
        return text

    def clean_list(self, texts):
        """
        Apply cleaning to a list of strings.
        """
        return [self.clean_text(t) for t in texts]


# ---------- Usage Example ----------
if __name__ == "__main__":
    raw_text = "Hello!!! 😊 Visit https://example.com for more info. Email me at test@example.com #NLP @AI 2025"
    
    processor = TextPreprocessor()
    cleaned = processor.clean_text(raw_text)
    
    print("Before:", raw_text)
    print("After :", cleaned)

    texts = ["Running runners ran!!!", "Email: abc@xyz.com #ML NLP!!! 123"]
    print("List Cleaned:", processor.clean_list(texts))
