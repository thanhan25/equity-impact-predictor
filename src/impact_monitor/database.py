"""
SQLAlchemy 2.0 Relational Database Schema.
Enforces 3NF normalization for Companies, Events, and Price Histories.
"""
import logging
from datetime import date, datetime
from typing import List, Optional

from sqlalchemy import String, Float, Integer, ForeignKey, DateTime, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    """Base class for SQLAlchemy 2.0 Declarative Mapping."""
    pass

class Company(Base):
    __tablename__ = "companies"
    
    ticker: Mapped[str] = mapped_column(String(10), primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    sector: Mapped[str] = mapped_column(String(50))
    
    # Relationships
    events: Mapped[List["MarketEvent"]] = relationship(back_populates="company", cascade="all, delete-orphan")
    prices: Mapped[List["PriceHistory"]] = relationship(back_populates="company", cascade="all, delete-orphan")

class MarketEvent(Base):
    __tablename__ = "market_events"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(ForeignKey("companies.ticker"))
    event_date: Mapped[datetime] = mapped_column(DateTime, index=True)
    event_type: Mapped[str] = mapped_column(String(50))  # e.g., 'Earnings', 'Macro News'
    headline: Mapped[str] = mapped_column(String(500))
    
    # NLP Extracted Features (Nullable until processed)
    nlp_theme: Mapped[Optional[str]] = mapped_column(String(50))
    nlp_sentiment_score: Mapped[Optional[float]] = mapped_column(Float)
    
    company: Mapped["Company"] = relationship(back_populates="events")

class PriceHistory(Base):
    __tablename__ = "price_history"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(ForeignKey("companies.ticker"))
    trade_date: Mapped[date] = mapped_column(DateTime, index=True)
    close_price: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(Integer)
    
    company: Mapped["Company"] = relationship(back_populates="prices")


def init_db(db_path: str = "sqlite:///data/processed/market_data.db"):
    """Initializes the database engine and creates tables if they don't exist."""
    engine = create_engine(db_path, echo=False)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info(f"Database initialized at {db_path}")
    return SessionLocal()