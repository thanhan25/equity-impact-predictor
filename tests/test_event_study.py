"""
Unit tests for the Event Study Engine.
Validates the OLS Market Model and Cumulative Abnormal Return (CAR) calculations.
"""
import pytest
import pandas as pd
import numpy as np
from src.impact_monitor.event_study import EventStudyEngine

@pytest.fixture
def mock_market_data():
    """
    Generates 150 days of perfectly correlated mock data (Beta = 1.0, Alpha = 0.0),
    with an artificial -5% crash on the asset on day 130 (the event day).
    """
    dates = pd.date_range(start="2025-01-01", periods=150, freq="B")
    
    # Base benchmark returns (random normal)
    np.random.seed(42)
    bench_returns = np.random.normal(0.0005, 0.01, 150)
    
    # Asset returns perfectly match benchmark, except on day 130
    asset_returns = bench_returns.copy()
    asset_returns[130] -= 0.05  # Artificial 5% crash independent of the market
    
    # Convert returns to price series (starting at 100)
    bench_prices = 100 * np.exp(np.cumsum(bench_returns))
    asset_prices = 100 * np.exp(np.cumsum(asset_returns))
    
    bench_series = pd.Series(bench_prices, index=dates)
    asset_series = pd.Series(asset_prices, index=dates)
    
    return asset_series, bench_series, dates[130]

def test_market_model_and_car(mock_market_data):
    """
    Tests if the OLS regression accurately finds Beta ~ 1.0 and 
    if the CAR correctly identifies the isolated -5% event crash.
    """
    asset_series, bench_series, event_date = mock_market_data
    
    # Initialize engine: 120 day estimation window, [-1, +2] event window
    engine = EventStudyEngine(estimation_window=120, pre_event=1, post_event=2)
    
    result = engine.calculate_car(asset_series, bench_series, event_date)
    
    # 1. Test OLS Regression Accuracy (Beta should be almost exactly 1.0)
    assert 0.95 < result['beta'] < 1.05
    assert -0.005 < result['alpha'] < 0.005
    
    # 2. Test Cumulative Abnormal Return (Should capture the ~ -5% crash)
    # CAR is in log-returns, so -5% is approximately -0.05
    assert result['cumulative_abnormal_return'] < -0.04
    assert result['cumulative_abnormal_return'] > -0.06