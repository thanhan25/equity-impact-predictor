"""
Quantitative Event Study Engine.
Calculates Cumulative Abnormal Returns (CAR) using the Market Model (OLS Regression).
"""
import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

class EventStudyEngine:
    def __init__(self, estimation_window: int = 120, pre_event: int = 1, post_event: int = 3):
        """
        Parameters:
        - estimation_window: Days prior to the event used to calculate Alpha and Beta.
        - pre_event: Days before the event to start measuring impact (captures information leaks).
        - post_event: Days after the event to measure impact.
        """
        self.estimation_window = estimation_window
        self.pre_event = pre_event
        self.post_event = post_event

    def calculate_car(self, asset_prices: pd.Series, benchmark_prices: pd.Series, event_idx: int) -> dict:
        """
        Executes the Market Model to calculate Cumulative Abnormal Returns (CAR) and Volatility.
        Requires price series aligned by date index.
        """
        # 1. Calculate daily log returns
        asset_returns = np.log(asset_prices / asset_prices.shift(1)).dropna()
        bench_returns = np.log(benchmark_prices / benchmark_prices.shift(1)).dropna()
        
        # Align series to ensure matching indices
        df = pd.concat([asset_returns, bench_returns], axis=1).dropna()
        df.columns = ['asset', 'benchmark']
        
        if event_idx not in df.index:
            raise ValueError("Event date not found in price index.")

        # Locate integer position of the event
        loc = df.index.get_loc(event_idx)
        
        # 2. Define Windows
        est_start = max(0, loc - self.pre_event - self.estimation_window)
        est_end = max(0, loc - self.pre_event - 1)
        
        event_start = max(0, loc - self.pre_event)
        event_end = min(len(df) - 1, loc + self.post_event)
        
        if est_end - est_start < 30:
            raise ValueError("Insufficient estimation window data (minimum 30 days required).")

        # 3. Fit the Market Model (OLS Regression) on Estimation Window
        X_est = df.iloc[est_start:est_end]['benchmark'].values.reshape(-1, 1)
        y_est = df.iloc[est_start:est_end]['asset'].values
        
        model = LinearRegression()
        model.fit(X_est, y_est)
        
        alpha, beta = model.intercept_, model.coef_[0]
        
        # 4. Calculate Abnormal Returns over the Event Window
        X_event = df.iloc[event_start:event_end]['benchmark'].values
        y_event = df.iloc[event_start:event_end]['asset'].values
        
        expected_returns = alpha + (beta * X_event)
        abnormal_returns = y_event - expected_returns
        
        # 5. Calculate Outputs
        car = np.sum(abnormal_returns)
        event_volatility = np.std(abnormal_returns) * np.sqrt(252) # Annualized
        
        return {
            "alpha": round(alpha, 6),
            "beta": round(beta, 4),
            "cumulative_abnormal_return": round(car, 4),
            "annualized_abnormal_volatility": round(event_volatility, 4),
            "event_window_days": len(abnormal_returns)
        }