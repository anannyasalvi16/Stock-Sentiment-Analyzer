"""
Stock Sentiment Analyzer
========================
A portfolio-grade Streamlit dashboard that fetches live news headlines
for any stock ticker and scores their sentiment using VADER NLP.

Stack: Streamlit · yfinance · NLTK VADER · Plotly Express · Pandas
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POSITIVE_THRESHOLD: float = 0.05
NEGATIVE_THRESHOLD: float = -0.05
MAX_HEADLINES: int = 15

SENTIMENT_COLORS: dict[str, str] = {
    "Positive": "#00D4AA",
    "Negative": "#FF4B4B",
    "Neutral":  "#FFD700",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1 · NLTK Initialisation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_resource
def load_analyzer() -> SentimentIntensityAnalyzer:
    """Download VADER lexicon once (cached) and return an analyzer instance."""
    nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2 · Data Fetching
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_news(ticker: str) -> Optional[list[dict]]:
    """
    Fetch the latest news headlines for *ticker* via yfinance.

    Handles structural differences across yfinance versions by trying
    both flat (``item["title"]``) and nested (``item["content"]["title"]``)
    layouts.  Returns a normalised list of dicts with ``title`` and
    ``publisher`` keys, capped at :pydata:`MAX_HEADLINES`, or ``None``
    on failure.
    """
    try:
        stock = yf.Ticker(ticker)

        # Quick validity check – if info comes back near-empty the ticker
        # is almost certainly invalid.
        info = stock.info or {}
        if info.get("quoteType") is None and info.get("shortName") is None:
            return None

        raw_news = stock.news
        if not raw_news:
            return None

        normalised: list[dict] = []
        for item in raw_news[:MAX_HEADLINES]:
            title: Optional[str] = None
            publisher: str = "Unknown"

            # Layout A – flat dict  (yfinance < 0.2.36)
            if "title" in item:
                title = item["title"]
                publisher = item.get("publisher", "Unknown")

            # Layout B – nested under "content"  (yfinance ≥ 0.2.36)
            elif "content" in item and isinstance(item["content"], dict):
                content = item["content"]
                title = content.get("title")
                provider = content.get("provider", {})
                if isinstance(provider, dict):
                    publisher = provider.get("displayName", "Unknown")

            if title:
                normalised.append({"title": title, "publisher": publisher})

        return normalised if normalised else None

    except Exception as exc:
        st.error(f"⚠️  Network / API error: {exc}")
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3 · Sentiment Processing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def analyze_sentiment(
    headlines: list[dict],
    analyzer: SentimentIntensityAnalyzer,
) -> pd.DataFrame:
    """
    Score each headline with VADER and return a tidy DataFrame.

    Columns: Status (emoji) · Headline · Publisher · Score · Sentiment
    """
    records: list[dict] = []
    for item in headlines:
        title: str = item["title"]
        compound: float = analyzer.polarity_scores(title)["compound"]

        if compound >= POSITIVE_THRESHOLD:
            label, indicator = "Positive", "🟢"
        elif compound <= NEGATIVE_THRESHOLD:
            label, indicator = "Negative", "🔴"
        else:
            label, indicator = "Neutral", "🟡"

        records.append(
            {
                "Status":    indicator,
                "Headline":  title,
                "Publisher":  item["publisher"],
                "Score":      round(compound, 4),
                "Sentiment":  label,
            }
        )

    return pd.DataFrame(records)


def compute_verdict(df: pd.DataFrame) -> tuple[str, str, str]:
    """
    Derive the overall market verdict from the average compound score.

    Returns ``(label, emoji, hex_colour)``.
    """
    avg: float = df["Score"].mean()
    if avg >= POSITIVE_THRESHOLD:
        return "BULLISH", "🟢", "#00D4AA"
    if avg <= NEGATIVE_THRESHOLD:
        return "BEARISH", "🔴", "#FF4B4B"
    return "NEUTRAL", "🟡", "#FFD700"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4 · UI Rendering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── 4a.  Custom CSS ──────────────────────────
def inject_custom_css() -> None:
    """Inject the fintech-inspired stylesheet."""
    st.markdown(
        """
        <style>
        /* ── Typography ─────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        .stApp, .stMarkdown, .stText { font-family: 'Inter', sans-serif; }

        /* ── Header ─────────────────────────── */
        .app-header {
            text-align: center;
            padding: 2rem 0 0.5rem;
        }
        .app-header h1 {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #00D4AA 0%, #00A3FF 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.25rem;
            letter-spacing: -0.5px;
        }
        .app-header p {
            color: #6B7A90;
            font-size: 1.05rem;
            font-weight: 300;
        }

        /* ── Divider ────────────────────────── */
        .divider {
            height: 1px;
            background: linear-gradient(90deg, transparent 0%, #2a3050 50%, transparent 100%);
            margin: 1.25rem 0;
        }

        /* ── Verdict Card ───────────────────── */
        .verdict-card {
            background: linear-gradient(145deg, #1a1f36 0%, #151928 100%);
            border: 1px solid #2a3050;
            border-radius: 20px;
            padding: 2.25rem 1rem;
            text-align: center;
            margin: 0.75rem 0 1.25rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
        }
        .verdict-card .label {
            font-size: 0.8rem;
            font-weight: 600;
            color: #6B7A90;
            text-transform: uppercase;
            letter-spacing: 2.5px;
            margin-bottom: 0.6rem;
        }
        .verdict-card .value {
            font-size: 2.6rem;
            font-weight: 800;
            letter-spacing: 2px;
        }
        .verdict-card .sub {
            font-size: 0.9rem;
            color: #6B7A90;
            margin-top: 0.4rem;
        }

        /* ── Metric Cards ───────────────────── */
        .metric-card {
            background: #1a1f36;
            border: 1px solid #2a3050;
            border-radius: 14px;
            padding: 1.3rem 1rem;
            text-align: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        }
        .metric-card .label {
            font-size: 0.7rem;
            font-weight: 600;
            color: #6B7A90;
            text-transform: uppercase;
            letter-spacing: 1.8px;
            margin-bottom: 0.45rem;
        }
        .metric-card .value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #FAFAFA;
        }

        /* ── Section Headers ────────────────── */
        .section-title {
            font-size: 1.05rem;
            font-weight: 600;
            color: #FAFAFA;
            margin: 1.5rem 0 0.8rem;
            padding-bottom: 0.45rem;
            border-bottom: 1px solid #2a3050;
            letter-spacing: 0.3px;
        }

        /* ── Footer ─────────────────────────── */
        .app-footer {
            text-align: center;
            padding: 2.5rem 0 1rem;
            color: #4A5568;
            font-size: 0.8rem;
            letter-spacing: 0.5px;
        }

        /* ── Hide Streamlit chrome ───────────── */
        #MainMenu, footer, header { visibility: hidden; }

        /* ── Primary button override ─────────  */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #00D4AA, #00A3FF);
            border: none;
            font-weight: 600;
            letter-spacing: 0.5px;
            padding: 0.6rem 1.5rem;
            border-radius: 10px;
            transition: opacity 0.2s ease;
        }
        .stButton > button[kind="primary"]:hover {
            opacity: 0.85;
        }

        /* ── Input field ─────────────────────  */
        .stTextInput > div > div > input {
            border-radius: 10px;
            border: 1px solid #2a3050;
            background: #151928;
            font-size: 1.05rem;
            padding: 0.7rem 1rem;
            text-transform: uppercase;
        }
        .stTextInput > div > div > input:focus {
            border-color: #00D4AA;
            box-shadow: 0 0 0 2px rgba(0, 212, 170, 0.15);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── 4b.  Component renderers ────────────────
def render_header() -> None:
    """Top-of-page branding block."""
    st.markdown(
        """
        <div class="app-header">
            <h1>📊 Stock Sentiment Analyzer</h1>
            <p>Real-time NLP-powered market sentiment from the latest news headlines</p>
        </div>
        <div class="divider"></div>
        """,
        unsafe_allow_html=True,
    )


def render_verdict(
    verdict: str, emoji: str, colour: str, avg_score: float, ticker: str
) -> None:
    """Full-width verdict banner."""
    st.markdown(
        f"""
        <div class="verdict-card">
            <div class="label">Market Sentiment Verdict · {ticker}</div>
            <div class="value" style="color:{colour};">{emoji}  {verdict}</div>
            <div class="sub">Average Compound Score: {avg_score:+.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(
    total: int, avg_score: float, pos_pct: float, neg_pct: float
) -> None:
    """Row of four KPI cards."""
    definitions = [
        ("Headlines Analyzed", str(total)),
        ("Avg. Compound Score", f"{avg_score:+.4f}"),
        ("Positive",           f"{pos_pct:.0f}%"),
        ("Negative",           f"{neg_pct:.0f}%"),
    ]
    cols = st.columns(4, gap="medium")
    for col, (label, value) in zip(cols, definitions):
        col.markdown(
            f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_donut_chart(df: pd.DataFrame) -> None:
    """Plotly donut chart of sentiment distribution."""
    counts = df["Sentiment"].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]

    fig = px.pie(
        counts,
        values="Count",
        names="Sentiment",
        hole=0.58,
        color="Sentiment",
        color_discrete_map=SENTIMENT_COLORS,
    )
    fig.update_traces(
        textposition="outside",
        textinfo="label+percent",
        textfont_size=13,
        marker=dict(line=dict(color="#0E1117", width=2.5)),
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Count: %{value}<br>"
            "Share: %{percent}<extra></extra>"
        ),
        pull=[0.03] * len(counts),
    )
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#FAFAFA"),
        margin=dict(t=30, b=30, l=10, r=10),
        height=370,
        annotations=[
            dict(
                text=f"<b>{len(df)}</b><br><span style='font-size:12px'>Headlines</span>",
                x=0.5,
                y=0.5,
                font_size=18,
                font_color="#FAFAFA",
                showarrow=False,
            )
        ],
    )
    st.plotly_chart(fig, use_container_width=True)


def render_data_table(df: pd.DataFrame) -> None:
    """Styled data-table with colour-coded sentiment labels."""
    display = df[["Status", "Headline", "Publisher", "Score", "Sentiment"]].copy()
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        height=400,
        column_config={
            "Status":    st.column_config.TextColumn("",          width="small"),
            "Headline":  st.column_config.TextColumn("Headline",  width="large"),
            "Publisher":  st.column_config.TextColumn("Source",    width="medium"),
            "Score":      st.column_config.NumberColumn("Score",   format="%.4f", width="small"),
            "Sentiment":  st.column_config.TextColumn("Sentiment", width="small"),
        },
    )


def render_footer() -> None:
    """Minimal footer with tech-stack attribution."""
    st.markdown(
        """
        <div class="app-footer">
            Built with Streamlit · yfinance · NLTK VADER · Plotly<br/>
            Sentiment analysis is automated and does not constitute financial advice.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main() -> None:
    """Application entry-point."""

    # ── Page config (must be first Streamlit call) ──
    st.set_page_config(
        page_title="Stock Sentiment Analyzer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    inject_custom_css()
    render_header()

    # ── Load NLP engine ──
    analyzer = load_analyzer()

    # ── Ticker search bar (centred) ──
    _pad_l, centre, _pad_r = st.columns([1, 2, 1])
    with centre:
        ticker_input: str = st.text_input(
            "Enter Stock Ticker",
            placeholder="e.g.  AAPL,  TSLA,  NVDA,  MSFT …",
            label_visibility="collapsed",
        )
        analyse_clicked: bool = st.button(
            "🔍  Analyze Sentiment",
            use_container_width=True,
            type="primary",
        )

    # ── Guard clauses ──
    if analyse_clicked and not ticker_input.strip():
        st.warning("Please enter a valid stock ticker symbol.")
        return

    if not analyse_clicked:
        # Show a subtle call-to-action on first load
        st.markdown(
            "<p style='text-align:center;color:#4A5568;margin-top:3rem;'>"
            "Enter a ticker above and press <strong>Analyze Sentiment</strong> to get started.</p>",
            unsafe_allow_html=True,
        )
        render_footer()
        return

    ticker: str = ticker_input.strip().upper()

    # ── Fetch ──
    with st.spinner(f"Fetching the latest headlines for **{ticker}** …"):
        headlines = fetch_news(ticker)

    if headlines is None:
        st.error(
            f"⚠️  Could not retrieve news for **{ticker}**. "
            "Double-check the symbol and ensure you have an internet connection."
        )
        return

    # ── Analyse ──
    df = analyze_sentiment(headlines, analyzer)
    if df.empty:
        st.warning("No headlines available to analyse for this ticker.")
        return

    # ── Derived metrics ──
    verdict_label, verdict_emoji, verdict_colour = compute_verdict(df)
    avg_score: float = df["Score"].mean()
    total: int = len(df)
    counts = df["Sentiment"].value_counts()
    pos_pct: float = counts.get("Positive", 0) / total * 100
    neg_pct: float = counts.get("Negative", 0) / total * 100

    # ── Render dashboard ──
    render_verdict(verdict_label, verdict_emoji, verdict_colour, avg_score, ticker)
    st.markdown("")  # spacing
    render_metrics(total, avg_score, pos_pct, neg_pct)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    chart_col, table_col = st.columns([2, 3], gap="large")
    with chart_col:
        st.markdown(
            '<div class="section-title">📊 Sentiment Distribution</div>',
            unsafe_allow_html=True,
        )
        render_donut_chart(df)
    with table_col:
        st.markdown(
            '<div class="section-title">📋 Headline Breakdown</div>',
            unsafe_allow_html=True,
        )
        render_data_table(df)

    render_footer()


# ── Entrypoint ──
if __name__ == "__main__":
    main()
