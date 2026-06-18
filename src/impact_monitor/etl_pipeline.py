"""
ETL Ingestion Engine.
Pulls OHLCV market data and unstructured corporate events, mapping them to the SQL database.
"""
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from .database import init_db, Company, PriceHistory, MarketEvent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class MarketDataIngestor:
    def __init__(self, db_session):
        self.db = db_session

    def add_company(self, ticker: str, name: str, sector: str):
        """Safely inserts a company into the database if it doesn't exist."""
        company = self.db.query(Company).filter(Company.ticker == ticker).first()
        if not company:
            company = Company(ticker=ticker, name=name, sector=sector)
            self.db.add(company)
            self.db.commit()
            logger.info(f"Added company: {ticker}")
        return company

    def ingest_price_history(self, ticker: str, lookback_days: int = 180):
        """Fetches historical price data via yfinance and loads into SQL."""
        logger.info(f"Fetching price history for {ticker}...")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        # FIX: Use yf.Ticker().history() to prevent MultiIndex column nesting errors
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date)
        
        if df.empty:
            logger.warning(f"No price data found for {ticker}.")
            return

        price_records = []
        for index, row in df.iterrows():
            # Convert to standard Python datetime
            trade_dt = index.to_pydatetime()
            # FIX: Strip timezone awareness for safe SQLite insertion
            if trade_dt.tzinfo is not None:
                trade_dt = trade_dt.replace(tzinfo=None)

            # Check if record already exists to prevent duplication
            exists = self.db.query(PriceHistory).filter_by(
                ticker=ticker, trade_date=trade_dt
            ).first()
            
            if not exists:
                record = PriceHistory(
                    ticker=ticker,
                    trade_date=trade_dt,
                    close_price=float(row['Close']),
                    volume=int(row['Volume'])
                )
                price_records.append(record)
        
        if price_records:
            self.db.bulk_save_objects(price_records)
            self.db.commit()
            logger.info(f"Inserted {len(price_records)} daily price rows for {ticker}.")
            
    def ingest_mock_events(self, ticker: str):
        """Simulates ingestion of unstructured news API data."""
        mock_events = [
            {"date": datetime.now() - timedelta(days=10), "type": "Earnings", "headline": f"{ticker} slashes forward guidance due to Eurozone inflation pressures."},
            {"date": datetime.now() - timedelta(days=45), "type": "PR", "headline": f"{ticker} announces departure of Chief Financial Officer effective immediately."},
            {"date": datetime.now() - timedelta(days=90), "type": "Earnings", "headline": f"{ticker} reports record Q2 margins, overcoming supply chain bottlenecks."}
        ]
        
        for event in mock_events:
            exists = self.db.query(MarketEvent).filter_by(
                ticker=ticker, event_date=event["date"]
            ).first()
            
            if not exists:
                new_event = MarketEvent(
                    ticker=ticker,
                    event_date=event["date"],
                    event_type=event["type"],
                    headline=event["headline"]
                )
                self.db.add(new_event)
        
        self.db.commit()
        logger.info(f"Ingested corporate events for {ticker}.")

if __name__ == "__main__":
    # Test execution block
    session = init_db("sqlite:///data/processed/market_data.db")
    ingestor = MarketDataIngestor(session)
    
    # European Mid-Cap Proxy Example (e.g., ASML or Rheinmetall, using standard tickers here)
    target_ticker = "RHM.DE" 
    
    ingestor.add_company(target_ticker, "Rheinmetall AG", "Industrials")
    ingestor.ingest_price_history(target_ticker, lookback_days=120)
    ingestor.ingest_mock_events(target_ticker)