from transformers import pipeline
import re
import logging
import json
from functools import lru_cache
import emoji
# Load external data files with proper encoding
try:
    with open("emoji_data.json", encoding='utf-8') as f:
        EMOJI_MAP = json.load(f)
    
    with open("hinglish_lexicon.json", encoding='utf-8') as f:
        LEXICON = json.load(f)
    
    with open("neutral_lexicon.json", encoding='utf-8') as f:
        NEUTRAL_TERMS = json.load(f)

except FileNotFoundError as e:
    logging.error(f"Missing required file: {str(e)}")
    raise
except json.JSONDecodeError as e:
    logging.error(f"Invalid JSON format: {str(e)}")
    raise

# Initialize mBERT with custom config
mbert_analyzer = pipeline(
    "text-classification",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device="cpu",
    return_all_scores=True
)

def detect_language(text):
    """Enhanced language detection with mixed script support"""
    devanagari = re.search(r'[\u0900-\u097F]', text)
    latin = re.search(r'[a-zA-Z]', text)
    numeric = re.search(r'\d', text)
    
    # Code-mixing detection
    if devanagari and latin:
        return "hinglish"
    if devanagari:
        return "hindi"
    if latin:
        if numeric and len(text) < 20:  # Handle short numeric/text mixes
            return "mixed"
        return "english"
    return "unknown"

def preprocess_text(text):
    """Advanced text normalization"""
    # Convert emojis to text and preserve
    text = emoji.demojize(text, delimiters=(" [", "] "))
    
    # Remove URLs but preserve domain mentions
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Normalize repeated characters with context awareness
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)  # Allow 2 repeats max
    
    # Handle mixed script normalization
    text = re.sub(r'[॰ॐ।]+', ' ', text)  # Hindi punctuation
    return text.strip()

def adjust_scores(text, scores):
    """Balanced score adjustment with enhanced neutral support"""
    lang = detect_language(text)
    text_lower = text.lower()
    
    # Emoji sentiment balancing
    emoji_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for emj, sentiment in EMOJI_MAP.items():
        count = text.count(emj)
        if count > 0:
            scores[sentiment] *= 1 + (0.15 * count)
            emoji_counts[sentiment] += count
    
    # Neutral emoji compensation
    if emoji_counts["neutral"] > 0:
        scores["neutral"] *= 1 + (0.1 * emoji_counts["neutral"])
    
    # Lexicon adjustments with intensity scaling
    neutral_boost = sum(1 for term in NEUTRAL_TERMS 
                     if re.search(r'\b' + re.escape(term) + r'\b', text_lower))
    scores["neutral"] *= 1 + (0.25 * neutral_boost)
    
    # Reduced positive/negative boosts
    boost_factors = {
        "positive": 1 + (0.12 * sum(1 for w in LEXICON["positive"] 
                                  if re.search(r'\b' + re.escape(w) + r'\b', text_lower))),
        "negative": 1 + (0.12 * sum(1 for w in LEXICON["negative"] 
                                  if re.search(r'\b' + re.escape(w) + r'\b', text_lower)))
    }
    
    # Apply boosts
    scores = {
        "positive": scores["positive"] * boost_factors["positive"],
        "negative": scores["negative"] * boost_factors["negative"],
        "neutral": scores["neutral"]
    }
    
    # Enhanced normalization
    total = sum(scores.values())
    min_score = min(scores.values())
    return {k: (v + min_score/2) / (total + min_score/2) for k, v in scores.items()}

@lru_cache(maxsize=1000)
def analyze_sentiment(text):
    """Precision-focused sentiment analysis with neutral boost"""
    try:
        cleaned_text = preprocess_text(text[:512])
        
        # Get raw scores from model
        raw_result = mbert_analyzer(cleaned_text)[0]
        star_ratings = {label['label']: label['score'] for label in raw_result}
        
        # Enhanced sentiment mapping
        sentiment_map = {
            "1 star": {"sentiment": "negative", "weight": 1.1},
            "2 stars": {"sentiment": "negative", "weight": 0.7},
            "3 stars": {"sentiment": "neutral", "weight": 1.4},
            "4 stars": {"sentiment": "positive", "weight": 0.7},
            "5 stars": {"sentiment": "positive", "weight": 1.1}
        }
        
        # Calculate weighted scores
        scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for label, data in sentiment_map.items():
            scores[data["sentiment"]] += star_ratings[label] * data["weight"]
        
        # Apply contextual adjustments
        scores = adjust_scores(cleaned_text, scores)
        
        # Confidence calculation with neutral priority
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        primary_score = sorted_scores[0][1]
        secondary_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
        
        confidence = int(((primary_score - secondary_score) * 125) + 25)
        confidence = max(1, min(100, confidence))
        
        return {
            "sentiment": sorted_scores[0][0],
            "confidence": confidence,
            "language": detect_language(cleaned_text),
            "breakdown": {
                "raw_scores": {k: round(v, 4) for k, v in scores.items()},
                "dominant_class": sorted_scores[0][0],
                "secondary_class": sorted_scores[1][0] if len(sorted_scores) > 1 else None
            },
            "source": "enhanced-mbert"
        }
        
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}", exc_info=True)
        return {
            "sentiment": "neutral",
            "confidence": 1,
            "language": "unknown",
            "source": "error"
        }
