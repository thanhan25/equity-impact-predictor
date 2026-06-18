# Event-Driven Equity Impact Predictor 📊

[![CI Pipeline](https://github.com/thanhan25/equity-impact-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/thanhan25/equity-impact-predictor/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-grade Python ETL and Machine Learning pipeline designed to track earnings events, classify market news via Natural Language Processing (NLP), and forecast short-term Cumulative Abnormal Returns (CAR) on listed equities.

Built to support institutional Equity Research desks by synthesizing complex unstructured market data into actionable, mathematically explainable investment alerts.

## 🏗️ Architecture & Core Pipeline

1. **Ingestion Engine (`etl_pipeline.py`):** Asynchronously fetches historical price/volume data (via APIs) and unstructured corporate news events, storing them in a strictly typed 3NF database built on **SQLAlchemy 2.0**.
2. **AI News Classification (`nlp_classifier.py`):** Utilizes zero-shot classification (HuggingFace Transformers/PyTorch) to automatically categorize volatile text into macroeconomic themes (e.g., *Forward Guidance Cut*, *Macro Headwinds*).
3. **Quantitative Event Study (`event_study.py`):** Executes the OLS Market Model over pre/post-event windows to isolate the exact Cumulative Abnormal Return (CAR) and volatility spikes, stripping out general market noise.
4. **Predictive XAI Engine (`predictive_model.py`):** A Gradient Boosting model (XGBoost) trained on historical event data. Utilizes **SHAP (SHapley Additive exPlanations)** to break down feature contributions, providing researchers with explainable "glass-box" predictions rather than "black-box" guesses.

## 🚀 Quickstart & Reproducibility

This project is packaged using standard PEP 517 distribution architecture.

```bash
# Clone the repository
git clone [https://github.com/thanhan25/equity-impact-predictor.git](https://github.com/thanhan25/equity-impact-predictor.git)
cd equity-impact-predictor

# Create virtual environment & install package defensively
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .[dev]

# Execute the master pipeline and generate the impact dashboard
python main.py

# Run the PyTest suite
pytest -v
```
