# ⚡ Alpha Signal Terminal: Equity Impact Predictor

Alpha Signal Terminal is an event-driven equity research application that combines market data engineering, NLP-based catalyst classification, predictive modeling, and an interactive Streamlit interface. It is designed as a portfolio project that demonstrates how unstructured news and structured price data can be turned into a short-horizon signal workflow for listed equities.

## Live app

[Open the live application](https://equity-impact-predictor-mjjwp8oarqbeenecyzk3pw.streamlit.app/)

<div align="center">
  <img src="assets\dashboard-preview.PNG" alt="Dashboard Preview" width="100%" />
  <p><i>Generated interactive dashboard featuring Desk Recommendation and Key Catalyst Summary</i></p>
</div>

## Overview

The application ingests market events and historical price data, transforms raw text into structured catalyst signals, engineers event-level features, and estimates short-term cumulative abnormal return after a new event. The front end then presents the output in a research-terminal format with recommendation, predicted CAR, feature contribution, trajectory view, and a historical event log.

## Main capabilities

* Ingests OHLCV and event/news data into a local database workflow.
* Classifies event text into market-relevant themes and sentiment signals.
* Builds features such as volatility, recent momentum, and event-type priors.
* Estimates 5-day cumulative abnormal return after an event.
* Presents the result in an interactive dashboard built for research-style interpretation.

## Project components

## Data ingestion

`etl_pipeline.py` collects and standardizes historical price and event data for downstream use.

## NLP classification

`nlp_classifier.py` maps raw headlines or event text into structured categories and sentiment-oriented fields.

## Predictive model

`predictive_model.py` trains the event-level model used to estimate post-catalyst return direction and magnitude.

## Streamlit terminal

`app.py` provides the user interface for signal review, event inspection, and visual analytics.

## Repository structure

<pre class="not-prose w-full rounded font-mono text-sm font-extralight"><div class="codeWrapper bg-subtle text-light selection:text-super selection:bg-super/10 my-md relative flex flex-col rounded-lg font-mono text-sm font-medium"><div class="translate-y-xs -translate-x-xs bottom-xl mb-xl flex h-0 items-start justify-end sm:sticky sm:top-xs"><div class="overflow-hidden border-subtlest ring-subtlest divide-subtlest bg-base rounded-full"><div class="border-subtlest ring-subtlest divide-subtlest bg-subtle"></div></div></div><div class="-mt-xl"><div><div data-testid="code-language-indicator" class="text-quiet bg-quiet py-xs px-sm inline-block rounded-br rounded-tl-lg text-xs font-thin">text</div></div><div><span><code><span><span>.
</span></span><span>├── app.py
</span><span>├── main.py
</span><span>├── etl_pipeline.py
</span><span>├── nlp_classifier.py
</span><span>├── predictive_model.py
</span><span>├── data/
</span><span>│   └── processed/
</span><span>├── assets/
</span><span>└── README.md</span></code></span></div></div></div></pre>

## Quickstart

## 1. Clone the repository

```bash
git clone https://github.com/thanhan25/equity-impact-predictor.git
cd equity-impact-predictor
```

## 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv venv
.env\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

If your project uses editable installation:

```bash
pip install -e .
```

Or, if you use a requirements file:

```bash
pip install -r requirements.txt
```

If needed, install the key packages directly:

```bash
pip install streamlit plotly lightgbm sqlalchemy pandas numpy
```

## 4. Run the pipeline

```bash
python main.py --popular
```

This step should populate the processed database used by the Streamlit app.

## 5. Launch the app

```bash
streamlit run app.py
```

After launch, open the local address shown in the terminal, usually `http://localhost:8501`.

## Signal logic

The current workflow uses four core event-level drivers:

* NLP theme sentiment
* 30-day momentum
* pre-event volatility drag
* event-type prior

These inputs are aggregated into a directional recommendation and a predicted 5-day CAR, then visualized through feature contribution and trajectory panels.

## Core features

| Feature                    | Description                                                |
| -------------------------- | ---------------------------------------------------------- |
| Event-driven signal engine | Estimates likely short-term equity impact after a catalyst |
| NLP catalyst labeling      | Turns raw text into structured themes                      |
| 5-day CAR framework        | Evaluates the signal in event-study terms                  |
| Explainability view        | Shows feature contribution instead of only output          |
| Governance panel           | Displays model monitoring and validation-style metrics     |
| Intelligence log           | Keeps the underlying event history visible                 |

## License

This project is licensed under the MIT License.
