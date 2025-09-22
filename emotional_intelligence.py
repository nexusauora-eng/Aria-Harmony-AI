
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import random
import re

class EmotionalIntelligence:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.emotion_keywords = {
            'happy': ['joy', 'excited', 'wonderful', 'great', 'amazing', 'happy'],
            'sad': ['sad', 'depressed', 'unhappy', 'miserable', 'heartbroken'],
            'angry': ['angry', 'mad', 'furious', 'annoyed', 'frustrated'],
            'anxious': ['anxious', 'worried', 'nervous', 'stressed', 'tense']
        }
        
    def analyze_sentiment(self, text):
        scores = self.sia.polarity_scores(text)
        if scores['compound'] >= 0.05:
            return 'positive'
        elif scores['compound'] <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def detect_emotion(self, text):
        text_lower = text.lower()
        for emotion, keywords in self.emotion_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return emotion
        return 'neutral'
    
    def get_empathic_response(self, emotion):
        responses = {
            'happy': "It's wonderful to hear you're feeling positive! ",
            'sad': "I'm sorry you're feeling this way. I'm here to listen. ",
            'angry': "I understand this is frustrating. Let's work through it together. ",
            'anxious': "It sounds like you're dealing with some worries. Take a deep breath. ",
            'neutral': "I appreciate you sharing this with me. "
        }
        return responses.get(emotion, "")






"""
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class EmotionalIntelligence:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze_emotion(self, text):
        sentiment = self.sia.polarity_scores(text)
        return sentiment

    def resonate_empathy(self, emotion):
        if emotion['compound'] >= 0.05:
            return "You seem happy, what's bringing you joy?"
        elif emotion['compound'] <= -0.05:
            return "You seem sad, would you like to talk about it?"
        else:
            return "You seem neutral, how can I help?"
"""
