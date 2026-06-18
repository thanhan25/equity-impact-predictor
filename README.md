
# ⚡ Alpha Signal Terminal: Equity Impact Predictor

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Real_Time-green)

An institutional-grade Quantitative Event Monitor. This full-stack pipeline ingests asynchronous market news and OHLCV price data, utilizing Zero-Shot NLP and Machine Learning to forecast short-term Cumulative Abnormal Returns (CAR).

## 🏗️ Architecture & Core Components

1. **ETL Data Ingestion (`etl_pipeline.py`):** Asynchronously fetches historical price/volume data and unstructured corporate news events, storing them in a strictly typed 3NF database via **SQLAlchemy**.
2. **AI News Classification (`nlp_classifier.py`):** Utilizes zero-shot NLP classification (`BART-Large-MNLI`) to categorize unstructured text into macroeconomic themes (e.g., *Forward Guidance Cut*, *M&A*) and assign sentiment probabilities.
3. **Machine Learning Backend (`predictive_model.py`):** Trains an **XGBoost** regressor on historical feature matrices (pre-event volatility, 30D momentum, NLP sentiment) to predict post-event asset trajectories.
4. **XAI Decision Terminal (`app.py`):** A front-end Streamlit application featuring **SHAP** (SHapley Additive exPlanations) value breakdowns, equal-weighted market regime tracking, and dynamic event-study trajectory visualization.

## 🚀 Local Quickstart

```bash
# Clone the repository
git clone [https://github.com/thanhan25/equity-impact-predictor.git](https://github.com/thanhan25/equity-impact-predictor.git)
cd equity-impact-predictor

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -e .
pip install streamlit plotly xgboost

# Run the backend ETL and ML training pipeline
python main.py --popular

# Launch the Alpha Signal Terminal
streamlit run app.py
```

## 📊 Dashboard Capabilities

* **Algorithmic Signal Generation:** Synthesizes NLP sentiment and market regimes into Bullish/Bearish/Neutral desk recommendations.
* **Feature Drift & Governance:** Live monitoring of Z-Score volatility drift to ensure real-time inputs remain within the ML model's historical training distribution.
* **Glass-Box Explainability:** Horizontal SHAP rendering immediately quantifies the directional drag of specific market conditions on the predicted output.
