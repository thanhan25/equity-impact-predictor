"""
NLP Zero-Shot Classification Engine.
Extracts thematic drivers from unstructured corporate and macroeconomic news.
"""
import logging
from typing import List

import torch
from transformers import pipeline
from .database import init_db, MarketEvent

logger = logging.getLogger(__name__)

class NewsClassifier:
    def __init__(self, db_session):
        self.db = db_session
        logger.info("Initializing Zero-Shot NLP Pipeline (PyTorch backend)...")
        
        # Check for GPU acceleration
        device = 0 if torch.cuda.is_available() else -1
        
        # Using a robust zero-shot model
        self.classifier = pipeline(
            "zero-shot-classification", 
            model="facebook/bart-large-mnli", 
            device=device
        )
        
        # Define the financial and macro themes we want to track
        self.themes = [
            "Earnings Beat", 
            "Earnings Miss", 
            "Forward Guidance Cut", 
            "Macroeconomic Headwinds", 
            "Supply Chain Disruption", 
            "C-Suite Turnover",
            "Mergers and Acquisitions"
        ]

    def process_unclassified_events(self):
        """Fetches events from the DB without NLP scores and classifies them."""
        unprocessed_events = self.db.query(MarketEvent).filter(MarketEvent.nlp_theme == None).all()
        
        if not unprocessed_events:
            logger.info("No unclassified events found in the database.")
            return

        logger.info(f"Processing {len(unprocessed_events)} unclassified headlines...")
        
        for event in unprocessed_events:
            try:
                # Run the zero-shot classification
                result = self.classifier(event.headline, self.themes)
                
                # Extract the top theme and its probability/confidence score
                top_theme = result['labels'][0]
                confidence = result['scores'][0]
                
                # Update the SQL record
                event.nlp_theme = top_theme
                event.nlp_sentiment_score = float(confidence)  # Using confidence as a proxy for signal strength
                
            except Exception as e:
                logger.error(f"Failed to classify event ID {event.id}: {e}")
                
        self.db.commit()
        logger.info("NLP Classification complete and committed to database.")

if __name__ == "__main__":
    session = init_db("sqlite:///data/processed/market_data.db")
    nlp_engine = NewsClassifier(session)
    nlp_engine.process_unclassified_events()