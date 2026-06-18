"""
Master Execution Pipeline (CLI)
Trains an XGBoost model on historical DB events and outputs SHAP explainability.
Usage:
  python main.py --popular
"""
import logging
import argparse
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

from src.impact_monitor.database import init_db, MarketEvent, PriceHistory
from src.impact_monitor.etl_pipeline import MarketDataIngestor
from src.impact_monitor.nlp_classifier import NewsClassifier
from src.impact_monitor.predictive_model import ImpactPredictor

os.makedirs("outputs", exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

POPULAR_COMPANIES = {
    "Technology": {
        "AAPL": "Apple Inc.", "MSFT": "Microsoft", "NVDA": "NVIDIA"
    },
    "Communication Services": {
        "META": "Meta Platforms", "GOOGL": "Alphabet Inc."
    },
    "Consumer Discretionary": {
        "AMZN": "Amazon", "TSLA": "Tesla"
    },
    "Financials": {
        "JPM": "JPMorgan Chase", "GS": "Goldman Sachs"
    }
}

def inject_dynamic_mock_events(ingestor, ticker, name):
    """Generates realistic mock headlines to ensure we have recent test data."""
    mock_events = [
        {"date": datetime.now() - timedelta(days=2), "type": "Macro", "headline": f"{name} faces macroeconomic headwinds as global supply chains tighten."},
        {"date": datetime.now() - timedelta(days=20), "type": "Earnings", "headline": f"{name} smashes quarterly earnings expectations, raising forward guidance."},
        {"date": datetime.now() - timedelta(days=45), "type": "PR", "headline": f"{name} announces strategic restructuring and executive turnover."}
    ]
    for event in mock_events:
        exists = ingestor.db.query(MarketEvent).filter_by(ticker=ticker, event_date=event["date"]).first()
        if not exists:
            new_event = MarketEvent(ticker=ticker, event_date=event["date"], event_type=event["type"], headline=event["headline"])
            ingestor.db.add(new_event)
    ingestor.db.commit()

def build_historical_feature_matrix(session):
    """Extracts real historical events and calculates their pre-event quantitative features."""
    logger.info("Extracting historical event features from database...")
    events = pd.read_sql("SELECT * FROM market_events WHERE nlp_sentiment_score IS NOT NULL", session.bind)
    prices = pd.read_sql("SELECT * FROM price_history", session.bind)
    
    if events.empty or prices.empty:
        raise ValueError("Insufficient data to build feature matrix.")

    events['event_date'] = pd.to_datetime(events['event_date'])
    prices['trade_date'] = pd.to_datetime(prices['trade_date'])
    
    features = []
    targets = []
    
    for _, event in events.iterrows():
        tkr = event['ticker']
        ev_date = event['event_date']
        
        # Slicing price data before the event
        hist_prices = prices[(prices['ticker'] == tkr) & (prices['trade_date'] < ev_date)].sort_values('trade_date').tail(30)
        
        if len(hist_prices) < 5:
            continue # Skip if not enough history
            
        # Calculate features
        returns = hist_prices['close_price'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100 if not returns.empty else 0.0
        momentum = (hist_prices['close_price'].iloc[-1] / hist_prices['close_price'].iloc[0]) - 1 if hist_prices['close_price'].iloc[0] != 0 else 0.0
        
        # Calculate Target (Simplified 5-Day Forward Return proxy for CAR)
        fwd_prices = prices[(prices['ticker'] == tkr) & (prices['trade_date'] >= ev_date)].sort_values('trade_date').head(5)
        if len(fwd_prices) > 0:
            target_car = (fwd_prices['close_price'].iloc[-1] / hist_prices['close_price'].iloc[-1]) - 1
        else:
            continue
            
        features.append({
            'nlp_sentiment_score': event['nlp_sentiment_score'],
            'pre_event_volatility': volatility,
            'momentum_30d': momentum * 100
        })
        targets.append(target_car)
        
    return pd.DataFrame(features), pd.Series(targets)

def run_pipeline(tickers_to_run):
    logger.info(f"Initializing Pipeline for {len(tickers_to_run)} assets...")
    session = init_db("sqlite:///data/processed/market_data.db")
    
    # 1. ETL INGESTION
    ingestor = MarketDataIngestor(session)
    for ticker, data in tickers_to_run.items():
        ingestor.add_company(ticker, data["name"], data["sector"])
        ingestor.ingest_price_history(ticker, lookback_days=200)
        inject_dynamic_mock_events(ingestor, ticker, data["name"])
    
    # 2. NLP CLASSIFICATION
    nlp = NewsClassifier(session)
    nlp.process_unclassified_events()
    
    # 3. REAL ML TRAINING
    logger.info("Training Predictive ML Engine on historical data...")
    X_train, y_train = build_historical_feature_matrix(session)
    
    if not X_train.empty:
        predictor = ImpactPredictor()
        predictor.train(X_train, y_train)
        logger.info("✅ ML Training Complete. Model and SHAP explainer ready.")
    else:
        logger.warning("Could not build feature matrix. Insufficient historical overlap.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--popular", action="store_true", help="Run pipeline for popular mega-caps")
    args = parser.parse_args()

    target_assets = {}
    if args.popular:
        for sector, comps in POPULAR_COMPANIES.items():
            for tk, name in comps.items():
                target_assets[tk] = {"name": name, "sector": sector}
        run_pipeline(target_assets)
    else:
        logger.warning("Use --popular to run the ingestion pipeline.")