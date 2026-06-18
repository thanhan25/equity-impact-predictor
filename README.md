
# ⚡ Alpha Signal Terminal: Equity Impact Predictor

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Model-00C853)
![HuggingFace](https://img.shields.io/badge/HuggingFace-NLP-FFD21E)

Alpha Signal Terminal is an event-driven equity research project that ingests market news and price data, classifies catalysts with NLP, engineers event features, and predicts short-term cumulative abnormal return (CAR) using a tabular machine-learning pipeline. The project is designed as a portfolio-grade research terminal rather than a notebook demo: it combines ETL, modeling, explainability, and a decision-facing Streamlit interface.

## Live application

[Open the live Streamlit app](https://equity-impact-predictor-mjjwp8oarqbeenecyzk3pw.streamlit.app/)

## Platform preview

The dashboard is built to make the model output scannable for an equity-research or desk workflow: recommendation, predicted 5-day CAR, catalyst summary, SHAP-style feature contribution view, event trajectory, and governance metrics are all shown on one screen.

<div align="center">
  <img src="assets/dashboard-preview.png" alt="Alpha Signal Terminal dashboard preview" width="900" />
</div>

## What the project does

- Ingests historical OHLCV price data and event/news data into a local SQL-backed workflow.
- Classifies event text into market-relevant themes and derives sentiment-style features.
- Builds event-level features such as pre-event volatility, recent momentum, and event-type priors.
- Predicts 5-day cumulative abnormal return after a catalyst.
- Visualizes model logic with explainability-oriented charts and decision-ready monitoring panels.

## Architecture

### 1. Data ingestion

`etl_pipeline.py` loads and normalizes historical price data plus event/news records into the working database.

### 2. NLP classification

`nlp_classifier.py` transforms raw headlines or event text into structured labels such as macro headwinds, earnings beat, regulatory risk, or guidance cut.

### 3. Predictive modeling

`predictive_model.py` trains a LightGBM model on engineered event features to estimate post-event short-horizon CAR.

### 4. Decision terminal

`app.py` serves the Streamlit interface with recommendation panels, event trajectory visualization, model governance indicators, and the intelligence log.

## Repository structure

```text
.
├── app.py
├── main.py
├── etl_pipeline.py
├── nlp_classifier.py
├── predictive_model.py
├── data/
│   └── processed/
├── assets/
├── requirements.txt / pyproject.toml
└── README.md
```

Adjust the tree above if your actual filenames differ.

## Modeling logic

The current application combines four main drivers at the event level:

- NLP theme sentiment
- 30-day price momentum
- pre-event volatility drag
- event-type prior

These are aggregated into a directional signal and a predicted 5-day CAR, then surfaced in the interface through feature contribution and trajectory charts.

## Local quickstart

### 1. Clone the repository

```bash
git clone https://github.com/thanhan25/equity-impact-predictor.git
cd equity-impact-predictor
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

If you are using a package-based setup:

```bash
pip install -e .
```

If you are using a requirements file:

```bash
pip install -r requirements.txt
```

If some UI or modeling packages are not already captured in your environment file, install them explicitly:

```bash
pip install streamlit plotly lightgbm sqlalchemy pandas numpy
```

### 4. Run the ingestion and modeling pipeline

```bash
python main.py --popular
```

This step is expected to populate the processed database used by the app.

### 5. Launch the terminal

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually `http://localhost:8501`.

## Core features

| Feature                    | What it does                                                                    |
| -------------------------- | ------------------------------------------------------------------------------- |
| Event-driven signal engine | Scores likely short-term post-event equity impact                               |
| NLP catalyst labeling      | Converts unstructured news into structured themes                               |
| 5-day CAR view             | Frames prediction in event-study terms rather than generic price direction      |
| XAI panel                  | Makes feature contribution visible instead of treating the model as a black box |
| Governance panel           | Surfaces validation and regime-awareness metrics                                |
| Intelligence log           | Preserves the underlying catalyst history behind the recommendation             |

## Why this project is strong for a finance / data role

This project demonstrates several things at once:

- Python-based ETL and data preparation
- SQL-backed analytical workflow
- feature engineering for event-driven modeling
- machine-learning application in a financial context
- explainability and governance thinking
- product-minded presentation through Streamlit

That combination makes it more compelling than a single notebook or static dashboard screenshot.

## Recommended screenshots

To make the repository look polished on GitHub, add these two images under `assets/`:

- `assets/dashboard-preview.png` — the full dashboard view
- `assets/xai-trajectory-preview.png` — a cropped SHAP + trajectory panel

If you add the second image, you can place it here:

```html
<div align="center">
  <img src="assets/xai-trajectory-preview.png" alt="Feature contribution and event trajectory view" width="900" />
</div>
```

## Notes

- Keep the README consistent with the real implementation. If you use zero-shot classification in one version and FinBERT or another model in another version, standardize the wording.
- Keep the model name consistent across the README, app UI, and codebase.
- If the app uses heuristic signal logic for demonstration, state that clearly or connect it to saved model artifacts before presenting it as a production model.

## Next improvements

- Replace heuristic front-end signal generation with saved model inference.
- Persist actual SHAP outputs from the trained model.
- Add backtest documentation and metric methodology.
- Add tests and GitHub Actions badges once CI is configured.

## License

This project is licensed under the MIT License
