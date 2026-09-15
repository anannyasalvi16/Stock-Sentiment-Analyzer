# 📊 Stock Sentiment Analyzer

[![Live App](https://img.shields.io/badge/Live_Demo-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://anannya-7efnjoycasbyhuwdafentm.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![NLTK VADER](https://img.shields.io/badge/NLP-NLTK_VADER-green?style=for-the-badge)](https://www.nltk.org/)
[![Plotly](https://img.shields.io/badge/Visualization-Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Author](https://img.shields.io/badge/Author-Anannya_Salvi-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anannya-salvi-a317623ab/)

An end-to-end, portfolio-grade financial NLP web application built with **Python**, **Streamlit**, **yfinance**, **NLTK VADER**, and **Plotly**. 

The **Stock Sentiment Analyzer** fetches real-time market news headlines for any stock ticker symbol (e.g. `AAPL`, `TSLA`, `NVDA`, `MSFT`), processes each headline using Natural Language Processing (NLP) sentiment intensity scoring, and presents market sentiment trends through a clean, fintech dashboard.

🔗 **Live Web Application:** [https://anannya-7efnjoycasbyhuwdafentm.streamlit.app/](https://anannya-7efnjoycasbyhuwdafentm.streamlit.app/)

---

## ✨ Key Features

- 🔍 **Live Ticker Search**: Instantly pull up to 15 real-time news headlines for any NYSE/NASDAQ stock.
- 🧠 **NLP Sentiment Scoring**: Uses **NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner)** to compute individual sentiment polarity scores (*Compound*, *Positive*, *Neutral*, *Negative*) tailored for social & news content.
- 📈 **Market Verdict Metric**: Generates a high-level overall sentiment verdict (**Bullish**, **Bearish**, or **Neutral**) alongside compound score averages.
- 📊 **Interactive Plotly Visualizations**:
  - **Distribution Pie Chart**: Breakdown of Positive vs. Neutral vs. Negative headlines.
  - **Time-Series Sentiment Scatter**: Chronological compound score trajectory across recent news releases.
- 📑 **Raw Data & Granular Breakdown**: Sortable, interactive data table listing every headline, publisher source, compound score, and sentiment category.
- 🎨 **Modern Fintech UI/UX**: White minimalist aesthetic featuring DM Sans typography, custom SVG trend indicators, and responsive layout design.

---

## 🛠️ Tech Stack

| Domain | Technology / Library | Description |
| :--- | :--- | :--- |
| **Frontend & UI** | [Streamlit](https://streamlit.io/) | Interactive Web Application Framework |
| **Data Fetching** | [yfinance](https://github.com/ranaroussi/yfinance) | Real-time Yahoo Finance API Data Provider |
| **Natural Language Processing** | [NLTK VADER](https://www.nltk.org/_modules/nltk/sentiment/vader.html) | Sentiment Polarity Intensity Analyzer |
| **Data Manipulation** | [Pandas](https://pandas.pydata.org/) | DataFrame structuring, sorting, and aggregations |
| **Data Visualization** | [Plotly Express](https://plotly.com/python/) | Interactive charts (Pie Distribution & Scatter Plot) |
| **Styling** | Custom Vanilla CSS + SVG | Clean light theme, micro-animations, and SVG icons |

---

## 🏗️ Architecture & Data Flow

```
[ User Ticker Input ] ──► [ yfinance API Fetch ] ──► [ NLTK VADER Lexicon ]
                                                              │
                                                              ▼
[ Interactive Dashboard ] ◄── [ Plotly & Dataframe ] ◄── [ Sentiment Scores ]
  • Overall Market Verdict
  • Sentiment Distribution
  • Chronological Scatter Plot
  • Headline Source Table
```

---

## 🚀 Quick Start (Local Setup)

Follow these steps to run the application locally on your machine:

### 1. Prerequisites
Ensure you have **Python 3.10+** installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/anannyasalvi16/Stock-Sentiment-Analyzer.git
cd Stock-Sentiment-Analyzer
```

### 3. Create a Virtual Environment (Optional but recommended)
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```
The application will open automatically in your web browser at `http://localhost:8501`.

---

## 🌐 Deployment

This repository is configured for automatic deployment via **Streamlit Community Cloud**.

- **Branch**: `main`
- **Entry point**: `app.py`
- **Dependencies**: Auto-installed from `requirements.txt`

Any changes pushed to the `main` branch trigger an automatic build and deployment pipeline.

---

## 👤 Author

**Anannya Salvi**

- 💼 **LinkedIn:** [linkedin.com/in/anannya-salvi-a317623ab](https://www.linkedin.com/in/anannya-salvi-a317623ab/)
- 🐙 **GitHub:** [@anannyasalvi16](https://github.com/anannyasalvi16)

---

## 📝 Disclaimer

*Sentiment analysis provided by this application is automated using natural language processing algorithms and is intended solely for portfolio, educational, and informational purposes. It does **not** constitute financial advice or investment recommendations.*
