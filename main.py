"""
Master Execution Pipeline (CLI)
Usage:
  python main.py --popular
  python main.py --tickers AAPL MSFT PLTR
"""
import logging
import argparse
import pandas as pd
import os
from datetime import datetime, timedelta

from src.impact_monitor.database import init_db, MarketEvent
from src.impact_monitor.etl_pipeline import MarketDataIngestor
from src.impact_monitor.nlp_classifier import NewsClassifier
from src.impact_monitor.predictive_model import ImpactPredictor

os.makedirs("outputs", exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Pre-categorized Mega-Caps
POPULAR_COMPANIES = {
    "Technology": {
        "AAPL": "Apple Inc.", "MSFT": "Microsoft", "NVDA": "NVIDIA", "PLTR": "Palantir Technologies"
    },
    "Communication Services": {
        "META": "Meta Platforms", "GOOGL": "Alphabet Inc.", "NFLX": "Netflix"
    },
    "Consumer Discretionary": {
        "AMZN": "Amazon", "TSLA": "Tesla"
    },
    "Financials": {
        "JPM": "JPMorgan Chase", "GS": "Goldman Sachs"
    },
    "Aerospace & Defense": {
        "LMT": "Lockheed Martin", "BA": "Boeing"
    }
}

def inject_dynamic_mock_events(ingestor, ticker, name):
    """Generates realistic mock headlines dynamically based on the company name."""
    mock_events = [
        {"date": datetime.now() - timedelta(days=5), "type": "Macro", "headline": f"{name} faces macroeconomic headwinds as global supply chains tighten."},
        {"date": datetime.now() - timedelta(days=20), "type": "Earnings", "headline": f"{name} smashes quarterly earnings expectations, raising forward guidance."},
        {"date": datetime.now() - timedelta(days=45), "type": "PR", "headline": f"{name} announces strategic restructuring and new product pipeline."}
    ]
    for event in mock_events:
        exists = ingestor.db.query(MarketEvent).filter_by(ticker=ticker, event_date=event["date"]).first()
        if not exists:
            new_event = MarketEvent(ticker=ticker, event_date=event["date"], event_type=event["type"], headline=event["headline"])
            ingestor.db.add(new_event)
    ingestor.db.commit()

def run_pipeline(tickers_to_run):
    logger.info(f"Initializing Pipeline for {len(tickers_to_run)} assets...")
    session = init_db("sqlite:///data/processed/market_data.db")
    
    # 1. BATCH ETL INGESTION
    ingestor = MarketDataIngestor(session)
    for ticker, data in tickers_to_run.items():
        name, sector = data["name"], data["sector"]
        ingestor.add_company(ticker, name, sector)
        ingestor.ingest_price_history(ticker, lookback_days=180)
        inject_dynamic_mock_events(ingestor, ticker, name)
    
    # 2. BATCH NLP CLASSIFICATION
    nlp = NewsClassifier(session)
    nlp.process_unclassified_events()
    
    # 3. ML TRAINING (Mocked historical feature matrix for XAI initialization)
    logger.info("Initializing Predictive ML Engine & SHAP Explainer...")
    mock_X_train = pd.DataFrame({
        'nlp_sentiment_score': [0.95, -0.80, 0.10, -0.92, 0.88],
        'pre_event_vol_annualized': [0.15, 0.40, 0.12, 0.50, 0.20],
        'is_earnings': [1, 1, 0, 0, 1]
    })
    mock_y_train = pd.Series([0.045, -0.072, 0.012, -0.105, 0.051])
    predictor = ImpactPredictor()
    predictor.train(mock_X_train, mock_y_train)
    
    logger.info("✅ Pipeline Execution Complete. Data is ready for the Dashboard.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Earnings & News Impact Pipeline")
    parser.add_argument("--popular", action="store_true", help="Run pipeline for categorized popular mega-cap companies")
    parser.add_argument("--tickers", nargs="+", help="Run pipeline for specific tickers (e.g., --tickers AAPL MSFT)")
    args = parser.parse_args()

    target_assets = {}

    if args.popular:
        for sector, comps in POPULAR_COMPANIES.items():
            for tk, name in comps.items():
                target_assets[tk] = {"name": name, "sector": sector}
    
    if args.tickers:
        for tk in args.tickers:
            # If manually specified, we assign it a custom sector
            target_assets[tk.upper()] = {"name": f"{tk.upper()} Corp", "sector": "Custom/Watchlist"}

    if not target_assets:
        logger.warning("No assets selected. Use --popular or --tickers <list>")
    else:
        run_pipeline(target_assets)