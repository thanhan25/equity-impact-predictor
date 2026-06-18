"""
Predictive Engine using XGBoost and SHAP.
Forecasts Cumulative Abnormal Returns (CAR) based on NLP Sentiment and Market context,
and provides mathematical explainability for institutional traders.
"""
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
import shap

logger = logging.getLogger(__name__)

class ImpactPredictor:
    def __init__(self):
        # Using XGBoost Regressor for tabular financial data
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        self.is_trained = False
        self.explainer = None

    def train(self, features_df: pd.DataFrame, target_series: pd.Series):
        """
        Trains the XGBoost model on historical NLP and market features.
        Expected features: ['nlp_sentiment_score', 'pre_event_volatility', 'theme_encoded_...']
        """
        logger.info(f"Training XGBoost model on {len(features_df)} historical events...")
        self.model.fit(features_df, target_series)
        self.is_trained = True
        
        # Initialize SHAP TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)
        logger.info("Model training complete. SHAP explainer initialized.")

    def predict_and_explain(self, current_event_features: pd.DataFrame) -> dict:
        """
        Predicts the CAR for a new news event and calculates SHAP values 
        to explain the prediction to a trader or portfolio manager.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before generating predictions.")

        # 1. Generate Quantitative Prediction
        prediction = self.model.predict(current_event_features)[0]
        
        # 2. Generate SHAP Explainability Values
        shap_values = self.explainer.shap_values(current_event_features)
        
        # 3. Format the actionable insight
        feature_contributions = dict(zip(current_event_features.columns, shap_values[0]))
        
        # Sort features by their absolute impact on the prediction
        sorted_contributions = sorted(
            feature_contributions.items(), 
            key=lambda item: abs(item[1]), 
            reverse=True
        )

        return {
            "predicted_cumulative_abnormal_return": round(prediction, 4),
            "base_value": round(self.explainer.expected_value, 4),
            "top_driving_factors": sorted_contributions[:3] # Top 3 reasons for the prediction
        }

if __name__ == "__main__":
    # Quick sanity test of the XGBoost/SHAP architecture
    predictor = ImpactPredictor()
    
    # Mock historical training data
    mock_X = pd.DataFrame({
        'nlp_sentiment_score': [0.9, -0.8, 0.1, -0.9, 0.8],
        'pre_event_vol_annualized': [0.15, 0.40, 0.12, 0.50, 0.20],
        'is_earnings': [1, 1, 0, 0, 1]
    })
    # Mock Target (Historical CAR)
    mock_y = pd.Series([0.05, -0.08, 0.01, -0.12, 0.04])
    
    predictor.train(mock_X, mock_y)
    
    # New Event: Highly negative sentiment, high market volatility, not earnings (e.g. Macro News)
    new_event = pd.DataFrame({
        'nlp_sentiment_score': [-0.85],
        'pre_event_vol_annualized': [0.45],
        'is_earnings': [0]
    })
    
    insight = predictor.predict_and_explain(new_event)
    print("\n--- ACTIONABLE TRADING INSIGHT ---")
    print(f"Predicted CAR: {insight['predicted_cumulative_abnormal_return'] * 100:.2f}%")
    print("Why? (SHAP Feature Contributions):")
    for feature, impact in insight['top_driving_factors']:
        direction = "up" if impact > 0 else "down"
        print(f" - {feature} drove the prediction {direction} by {abs(impact)*100:.2f}%")