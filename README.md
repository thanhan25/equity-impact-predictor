
# ⚡ Alpha Signal Terminal: Equity Impact Predictor

Alpha Signal Terminal is an event-driven equity research application that combines market data engineering, NLP-based catalyst classification, predictive modeling, and an interactive Streamlit interface. It is designed as a portfolio project that demonstrates how unstructured news and structured price data can be turned into a short-horizon signal workflow for listed equities.

## Live app

[Open the live application](https://equity-impact-predictor-mjjwp8oarqbeenecyzk3pw.streamlit.app/)

![Alpha Signal Terminal dashboard](assets/dashboard-preview.png)

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

Update the structure section if your repository contains additional modules or different filenames.

## Quickstart

## 1. Clone the repository

<pre class="not-prose w-full rounded font-mono text-sm font-extralight"><div class="codeWrapper bg-subtle text-light selection:text-super selection:bg-super/10 my-md relative flex flex-col rounded-lg font-mono text-sm font-medium"><div class="translate-y-xs -translate-x-xs bottom-xl mb-xl flex h-0 items-start justify-end sm:sticky sm:top-xs"><div class="overflow-hidden border-subtlest ring-subtlest divide-subtlest bg-base rounded-full"><div class="border-subtlest ring-subtlest divide-subtlest bg-subtle"></div></div></div><div class="-mt-xl"><div><div data-testid="code-language-indicator" class="text-quiet bg-quiet py-xs px-sm inline-block rounded-br rounded-tl-lg text-xs font-thin">bash</div></div><div><span><code><span><span class="token token">git</span><span> clone https://github.com/thanhan25/equity-impact-predictor.git
</span></span><span><span></span><span class="token token">cd</span><span> equity-impact-predictor</span></span></code></span></div></div></div></pre>

## 2. Create a virtual environment

**Windows PowerShell**

<pre class="not-prose w-full rounded font-mono text-sm font-extralight"><div class="codeWrapper bg-subtle text-light selection:text-super selection:bg-super/10 my-md relative flex flex-col rounded-lg font-mono text-sm font-medium"><div class="translate-y-xs -translate-x-xs bottom-xl mb-xl flex h-0 items-start justify-end sm:sticky sm:top-xs"><div class="overflow-hidden border-subtlest ring-subtlest divide-subtlest bg-base rounded-full"><div class="border-subtlest ring-subtlest divide-subtlest bg-subtle"></div></div></div><div class="-mt-xl"><div><div data-testid="code-language-indicator" class="text-quiet bg-quiet py-xs px-sm inline-block rounded-br rounded-tl-lg text-xs font-thin">powershell</div></div><div><span><code><span><span>python </span><span class="token token operator">-</span><span>m venv venv
</span></span><span><span></span><span class="token token punctuation">.</span><span>env\Scripts\Activate</span><span class="token token punctuation">.</span><span>ps1</span></span></code></span></div></div></div></pre>

**macOS / Linux**

<pre class="not-prose w-full rounded font-mono text-sm font-extralight"><div class="codeWrapper bg-subtle text-light selection:text-super selection:bg-super/10 my-md relative flex flex-col rounded-lg font-mono text-sm font-medium"><div class="translate-y-xs -translate-x-xs bottom-xl mb-xl flex h-0 items-start justify-end sm:sticky sm:top-xs"><div class="overflow-hidden border-subtlest ring-subtlest divide-subtlest bg-base rounded-full"><div class="border-subtlest ring-subtlest divide-subtlest bg-subtle"></div></div></div><div class="-mt-xl"><div><div data-testid="code-language-indicator" class="text-quiet bg-quiet py-xs px-sm inline-block rounded-br rounded-tl-lg text-xs font-thin">bash</div></div><div><span><code><span><span>python -m venv venv
</span></span><span><span></span><span class="token token">source</span><span> venv/bin/activate</span></span></code></span></div></div></div></pre>

## 3. Install dependencies

If your project uses editable installation:

<pre class="not-prose w-full rounded font-mono text-sm font-extralight"><div class="codeWrapper bg-subtle text-light selection:text-super selection:bg-super/10 my-md relative flex flex-col rounded-lg font-mono text-sm font-medium"><div class="translate-y-xs -translate-x-xs bottom-xl mb-xl flex h-0 items-start justify-end sm:sticky sm:top-xs"><div class="overflow-hidden border-subtlest ring-subtlest divide-subtlest bg-base rounded-full"><div class="border-subtlest ring-subtlest divide-subtlest bg-subtle"></div></div></div><div class="-mt-xl"><div><div data-testid="code-language-indicator" class="text-quiet bg-quiet py-xs px-sm inline-block rounded-br rounded-tl-lg text-xs font-thin">bash</div></div><div><span><code><span><span>pip </span><span class="token token">install</span><span> -e </span><span class="token token">.</span></span></code></span></div></div></div></pre>

Or, if you use a requirements file:

<pre class="not-prose w-full rounded font-mono text-sm font-extralight"><div class="codeWrapper bg-subtle text-light selection:text-super selection:bg-super/10 my-md relative flex flex-col rounded-lg font-mono text-sm font-medium"><div class="translate-y-xs -translate-x-xs bottom-xl mb-xl flex h-0 items-start justify-end sm:sticky sm:top-xs"><div class="overflow-hidden border-subtlest ring-subtlest divide-subtlest bg-base rounded-full"><div class="border-subtlest ring-subtlest divide-subtlest bg-subtle"></div></div></div><div class="-mt-xl"><div><div data-testid="code-language-indicator" class="text-quiet bg-quiet py-xs px-sm inline-block rounded-br rounded-tl-lg text-xs font-thin">bash</div></div><div><span><code><span><span>pip </span><span class="token token">install</span><span> -r requirements.txt</span></span></code></span></div></div></div></pre>

If needed, install the key packages directly:

<pre class="not-prose w-full rounded font-mono text-sm font-extralight"><div class="codeWrapper bg-subtle text-light selection:text-super selection:bg-super/10 my-md relative flex flex-col rounded-lg font-mono text-sm font-medium"><div class="translate-y-xs -translate-x-xs bottom-xl mb-xl flex h-0 items-start justify-end sm:sticky sm:top-xs"><div class="overflow-hidden border-subtlest ring-subtlest divide-subtlest bg-base rounded-full"><div class="border-subtlest ring-subtlest divide-subtlest bg-subtle"></div></div></div><div class="-mt-xl"><div><div data-testid="code-language-indicator" class="text-quiet bg-quiet py-xs px-sm inline-block rounded-br rounded-tl-lg text-xs font-thin">bash</div></div><div><span><code><span><span>pip </span><span class="token token">install</span><span> streamlit plotly lightgbm sqlalchemy pandas numpy</span></span></code></span></div></div></div></pre>

## 4. Run the pipeline

<pre class="not-prose w-full rounded font-mono text-sm font-extralight"><div class="codeWrapper bg-subtle text-light selection:text-super selection:bg-super/10 my-md relative flex flex-col rounded-lg font-mono text-sm font-medium"><div class="translate-y-xs -translate-x-xs bottom-xl mb-xl flex h-0 items-start justify-end sm:sticky sm:top-xs"><div class="overflow-hidden border-subtlest ring-subtlest divide-subtlest bg-base rounded-full"><div class="border-subtlest ring-subtlest divide-subtlest bg-subtle"></div></div></div><div class="-mt-xl"><div><div data-testid="code-language-indicator" class="text-quiet bg-quiet py-xs px-sm inline-block rounded-br rounded-tl-lg text-xs font-thin">bash</div></div><div><span><code><span><span>python main.py --popular</span></span></code></span></div></div></div></pre>

This step should populate the processed database used by the Streamlit app.

## 5. Launch the app

<pre class="not-prose w-full rounded font-mono text-sm font-extralight"><div class="codeWrapper bg-subtle text-light selection:text-super selection:bg-super/10 my-md relative flex flex-col rounded-lg font-mono text-sm font-medium"><div class="translate-y-xs -translate-x-xs bottom-xl mb-xl flex h-0 items-start justify-end sm:sticky sm:top-xs"><div class="overflow-hidden border-subtlest ring-subtlest divide-subtlest bg-base rounded-full"><div class="border-subtlest ring-subtlest divide-subtlest bg-subtle"></div></div></div><div class="-mt-xl"><div><div data-testid="code-language-indicator" class="text-quiet bg-quiet py-xs px-sm inline-block rounded-br rounded-tl-lg text-xs font-thin">bash</div></div><div><span><code><span><span>streamlit run app.py</span></span></code></span></div></div></div></pre>

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

## Why this project is valuable

This repository demonstrates a combination of technical and analytical skills that are useful in finance, analytics, and applied machine-learning roles:

* Python ETL and data preparation
* SQL-oriented data workflow
* event-driven feature engineering
* applied machine learning in a finance context
* explainability and model-governance thinking
* dashboard presentation for non-technical end users

## Important note on screenshots

Your current GitHub README is showing broken images because the referenced files are not present at the paths used by the README. GitHub will only render images correctly if the files actually exist in the repository, for example:

* `assets/dashboard-preview.png`
* `assets/xai-trajectory-preview.png`

If those files are not in the repo yet, do not include `<img>` tags. It is better to have no screenshot than a broken image placeholder.

## Recommended next improvements

* Save real dashboard screenshots into the `assets/` folder.
* Connect the front-end signal output to persisted model artifacts.
* Add backtest documentation and metric methodology.
* Add tests, CI, and release workflow badges.

## License

This project is licensed under the MIT License.
