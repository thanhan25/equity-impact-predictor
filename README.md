
# ⚡ Alpha Signal Terminal: Equity Impact Predictor

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Predictive_Model-00C853)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Zero--Shot_NLP-FFD21E)

An institutional-grade Quantitative Event Monitor. This full-stack pipeline ingests asynchronous market news and OHLCV price data, utilizing Zero-Shot Natural Language Processing (NLP) and Machine Learning to forecast short-term Cumulative Abnormal Returns (CAR).

### 🌐 Live Application

**[Click here to view the live Alpha Signal Terminal](https://[Insert-Your-Streamlit-Cloud-Link-Here])**

---

## 📸 Platform Interface & Explainable AI (XAI)

> *The terminal converts complex algorithmic outputs into a highly scannable, decision-ready interface for trading desks and quantitative researchers.*

*(**Note to self:** Drag and drop your main dashboard screenshot right here to let GitHub generate the image link)*

<div align="center">
  <img src="[Drag and drop your dashboard screenshot here]" alt="Alpha Signal Terminal Main Dashboard" width="800"/>
</div>

### Explainable AI & Event Study Trajectories

Instead of a "black box" prediction, the terminal leverages **SHAP (SHapley Additive exPlanations)** to mathematically isolate and visualize the exact market drivers pushing the algorithmic signal, plotted directly alongside the expected 5-day event trajectory.

*(**Note to self:** Take a cropped screenshot of just the SHAP and Trajectory charts and drag it here)*

<div align="center">
  <img src="[Drag and drop your SHAP/Trajectory screenshot here]" alt="SHAP Feature Importance and Event Trajectory" width="800"/>
</div>

---

## 🏗️ Architecture & Core Components

1. **ETL Data Ingestion (`etl_pipeline.py`):** Asynchronously fetches historical price/volume data and unstructured corporate news events, mapping them into a strictly typed 3NF database via **SQLAlchemy**.
2. **AI News Classification (`nlp_classifier.py`):** Utilizes zero-shot NLP classification (`BART-Large-MNLI`) to categorize unstructured text into macroeconomic themes (e.g., *Forward Guidance Cut*, *Regulatory Risk*) and assign sentiment probabilities.
3. **Machine Learning Backend (`predictive_model.py`):** Trains a **LightGBM** regressor on historical feature matrices (pre-event volatility drag, 30D momentum, NLP sentiment) to predict post-event asset trajectories relative to a market pseudo-index.
4. **XAI Decision Terminal (`app.py`):** A front-end Streamlit application featuring live feature-drift governance metrics, interactive Plotly visualizations, and logic-driven desk recommendations.

## 🚀 Local Quickstart

```bash
# Clone the repository
git clone [https://github.com/thanhan25/equity-impact-predictor.git](https://github.com/thanhan25/equity-impact-predictor.git)
cd equity-impact-predictor

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -e .
pip install streamlit plotly lightgbm

# Run the backend ETL and ML training pipeline
python main.py --popular

# Launch the Alpha Signal Terminal
streamlit run app.py
```
