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
    "Positive": "#10B981",
    "Negative": "#EF4444",
    "Neutral":  "#F59E0B",
}

# ── Inline SVG icons ────────────────────────
SVG_LOGO = """<svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect x="2" y="18" width="6" height="12" rx="1.5" fill="#10B981"/>
  <rect x="13" y="10" width="6" height="20" rx="1.5" fill="#3B82F6"/>
  <rect x="24" y="2" width="6" height="28" rx="1.5" fill="#0F172A"/>
</svg>"""

SVG_POSITIVE = """<svg width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="9" cy="9" r="9" fill="#D1FAE5"/>
  <path d="M5.5 9.5L7.5 11.5L12.5 6.5" stroke="#059669" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

SVG_NEGATIVE = """<svg width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="9" cy="9" r="9" fill="#FEE2E2"/>
  <path d="M6.5 6.5L11.5 11.5M11.5 6.5L6.5 11.5" stroke="#DC2626" stroke-width="2" stroke-linecap="round"/>
</svg>"""

SVG_NEUTRAL = """<svg width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="9" cy="9" r="9" fill="#FEF3C7"/>
  <path d="M6 9H12" stroke="#D97706" stroke-width="2" stroke-linecap="round"/>
</svg>"""

SVG_CHART = """<svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M10 2C5.58 2 2 5.58 2 10C2 14.42 5.58 18 10 18C14.42 18 18 14.42 18 10C18 5.58 14.42 2 10 2ZM10 3.5V10L14.5 12.5" stroke="#64748B" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
</svg>"""

SVG_TABLE = """<svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect x="2" y="3" width="16" height="14" rx="2" stroke="#64748B" stroke-width="1.5" fill="none"/>
  <line x1="2" y1="7.5" x2="18" y2="7.5" stroke="#64748B" stroke-width="1.5"/>
  <line x1="7.5" y1="3" x2="7.5" y2="17" stroke="#64748B" stroke-width="1.5"/>
</svg>"""

SVG_SEARCH = """<svg width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="7.5" cy="7.5" r="5.5" stroke="white" stroke-width="2" fill="none"/>
  <line x1="12" y1="12" x2="16" y2="16" stroke="white" stroke-width="2" stroke-linecap="round"/>
</svg>"""

SVG_BULLISH = """<svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M4 20L10 14L14 18L24 8" stroke="#059669" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M17 8H24V15" stroke="#059669" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

SVG_BEARISH = """<svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M4 8L10 14L14 10L24 20" stroke="#DC2626" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M17 20H24V13" stroke="#DC2626" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

SVG_NEUTRAL_TREND = """<svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M4 14H24" stroke="#D97706" stroke-width="2.5" stroke-linecap="round"/>
  <path d="M19 9L24 14L19 19" stroke="#D97706" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""


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
    both flat and nested layouts.  Returns a normalised list of dicts
    with ``title`` and ``publisher`` keys, capped at MAX_HEADLINES,
    or ``None`` on failure.
    """
    try:
        stock = yf.Ticker(ticker)

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

            if "title" in item:
                title = item["title"]
                publisher = item.get("publisher", "Unknown")
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
        st.error(f"Network / API error: {exc}")
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

    Columns: Status · Headline · Publisher · Score · Sentiment
    """
    records: list[dict] = []
    for item in headlines:
        title: str = item["title"]
        compound: float = analyzer.polarity_scores(title)["compound"]

        if compound >= POSITIVE_THRESHOLD:
            label, indicator = "Positive", SVG_POSITIVE
        elif compound <= NEGATIVE_THRESHOLD:
            label, indicator = "Negative", SVG_NEGATIVE
        else:
            label, indicator = "Neutral", SVG_NEUTRAL

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


def compute_verdict(df: pd.DataFrame) -> tuple[str, str, str, str]:
    """
    Derive the overall market verdict from the average compound score.

    Returns ``(label, svg_icon, hex_colour, bg_colour)``.
    """
    avg: float = df["Score"].mean()
    if avg >= POSITIVE_THRESHOLD:
        return "BULLISH", SVG_BULLISH, "#059669", "#F0FDF4"
    if avg <= NEGATIVE_THRESHOLD:
        return "BEARISH", SVG_BEARISH, "#DC2626", "#FEF2F2"
    return "NEUTRAL", SVG_NEUTRAL_TREND, "#D97706", "#FFFBEB"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4 · UI Rendering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def inject_custom_css() -> None:
    """Inject the light fintech stylesheet with premium typography."""
    st.markdown(
        """
        <style>
        /* ── Typography ─────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

        *, .stApp, .stMarkdown, .stText, p, span, div, h1, h2, h3, h4, h5, h6, li, td, th, label, input, button {
            font-family: 'DM Sans', system-ui, -apple-system, sans-serif !important;
        }

        .stApp {
            background-color: #F8FAFC;
        }

        /* ── Header ─────────────────────────── */
        .app-header {
            text-align: center;
            padding: 2.5rem 0 1rem;
        }
        .app-header .logo-row {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            margin-bottom: 8px;
        }
        .app-header h1 {
            font-size: 1.85rem;
            font-weight: 700;
            color: #0F172A;
            margin: 0;
            letter-spacing: -0.5px;
        }
        .app-header p {
            color: #94A3B8;
            font-size: 0.95rem;
            font-weight: 400;
            margin-top: 4px;
            letter-spacing: 0.1px;
        }

        /* ── Divider ────────────────────────── */
        .divider {
            height: 1px;
            background: #E2E8F0;
            margin: 1.25rem 0;
        }

        /* ── Verdict Card ───────────────────── */
        .verdict-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 2rem 1.5rem;
            text-align: center;
            margin: 0.75rem 0 1.25rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        }
        .verdict-card .icon-row {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            margin-bottom: 4px;
        }
        .verdict-card .label {
            font-size: 0.72rem;
            font-weight: 600;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 12px;
        }
        .verdict-card .value {
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: 1.5px;
        }
        .verdict-card .sub {
            font-size: 0.85rem;
            color: #94A3B8;
            margin-top: 8px;
            font-weight: 400;
        }
        .verdict-badge {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            margin-top: 8px;
        }

        /* ── Metric Cards ───────────────────── */
        .metric-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.25rem 1rem;
            text-align: center;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
            transition: box-shadow 0.2s ease;
        }
        .metric-card:hover {
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
        }
        .metric-card .label {
            font-size: 0.68rem;
            font-weight: 600;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 6px;
        }
        .metric-card .value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #0F172A;
        }

        /* ── Section Headers ────────────────── */
        .section-title {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.95rem;
            font-weight: 600;
            color: #0F172A;
            margin: 1.5rem 0 0.8rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #E2E8F0;
            letter-spacing: 0.2px;
        }

        /* ── Footer ─────────────────────────── */
        .app-footer {
            text-align: center;
            padding: 2.5rem 0 1rem;
            color: #CBD5E1;
            font-size: 0.78rem;
            letter-spacing: 0.3px;
        }
        .app-footer a {
            color: #94A3B8;
            text-decoration: none;
        }

        /* ── CTA placeholder text ───────────  */
        .cta-text {
            text-align: center;
            color: #94A3B8;
            margin-top: 3rem;
            font-size: 0.95rem;
        }
        .cta-text strong {
            color: #64748B;
        }

        /* ── Hide Streamlit chrome ───────────  */
        #MainMenu, footer, header { visibility: hidden; }

        /* ── Primary button override ─────────  */
        .stButton > button[kind="primary"],
        .stButton > button {
            background-color: #0F172A !important;
            color: #FFFFFF !important;
            border: none !important;
            font-weight: 600 !important;
            letter-spacing: 0.3px !important;
            padding: 0.6rem 1.5rem !important;
            border-radius: 10px !important;
            transition: background-color 0.2s ease !important;
            font-family: 'DM Sans', system-ui, sans-serif !important;
        }
        .stButton > button[kind="primary"]:hover,
        .stButton > button:hover {
            background-color: #1E293B !important;
            color: #FFFFFF !important;
        }

        /* ── Input field ─────────────────────  */
        .stTextInput > div > div > input {
            border-radius: 10px !important;
            border: 1.5px solid #E2E8F0 !important;
            background: #FFFFFF !important;
            font-size: 1rem !important;
            padding: 0.7rem 1rem !important;
            text-transform: uppercase;
            color: #0F172A !important;
            font-family: 'DM Sans', system-ui, sans-serif !important;
        }
        .stTextInput > div > div > input:focus {
            border-color: #0F172A !important;
            box-shadow: 0 0 0 2px rgba(15, 23, 42, 0.08) !important;
        }
        .stTextInput > div > div > input::placeholder {
            color: #CBD5E1 !important;
            text-transform: none;
        }

        /* ── Dataframe ───────────────────────  */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
        }

        /* ── Plotly chart background ─────────  */
        .stPlotlyChart {
            background: transparent;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── 4b.  Component renderers ────────────────
def render_header() -> None:
    """Top-of-page branding block."""
    st.markdown(
        f"""
        <div class="app-header">
            <div class="logo-row">
                {SVG_LOGO}
                <h1>Stock Sentiment Analyzer</h1>
            </div>
            <p>Real-time NLP-powered market sentiment from the latest news headlines</p>
        </div>
        <div class="divider"></div>
        """,
        unsafe_allow_html=True,
    )


def render_verdict(
    verdict: str, svg_icon: str, colour: str, bg_colour: str, avg_score: float, ticker: str
) -> None:
    """Full-width verdict banner."""
    st.markdown(
        f"""
        <div class="verdict-card">
            <div class="label">Market Sentiment Verdict · {ticker}</div>
            <div class="icon-row">
                {svg_icon}
                <span class="value" style="color:{colour};">{verdict}</span>
            </div>
            <div class="verdict-badge" style="background:{bg_colour};color:{colour};">
                Avg. Compound Score: {avg_score:+.4f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(
    total: int, avg_score: float, pos_pct: float, neg_pct: float
) -> None:
    """Row of four KPI cards."""
    definitions = [
        ("Headlines Analyzed", str(total), "#0F172A"),
        ("Avg. Score",         f"{avg_score:+.4f}", "#3B82F6"),
        ("Positive",           f"{pos_pct:.0f}%",   "#059669"),
        ("Negative",           f"{neg_pct:.0f}%",   "#DC2626"),
    ]
    cols = st.columns(4, gap="medium")
    for col, (label, value, val_colour) in zip(cols, definitions):
        col.markdown(
            f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value" style="color:{val_colour};">{value}</div>
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
        hole=0.6,
        color="Sentiment",
        color_discrete_map=SENTIMENT_COLORS,
    )
    fig.update_traces(
        textposition="outside",
        textinfo="label+percent",
        textfont_size=13,
        textfont_color="#64748B",
        marker=dict(line=dict(color="#F8FAFC", width=3)),
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Count: %{value}<br>"
            "Share: %{percent}<extra></extra>"
        ),
        pull=[0.02] * len(counts),
    )
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, system-ui, sans-serif", color="#0F172A"),
        margin=dict(t=30, b=30, l=10, r=10),
        height=370,
        annotations=[
            dict(
                text=f"<b style='font-size:22px;color:#0F172A'>{len(df)}</b>"
                     f"<br><span style='font-size:12px;color:#94A3B8'>Headlines</span>",
                x=0.5,
                y=0.5,
                font_size=16,
                showarrow=False,
            )
        ],
    )
    st.plotly_chart(fig, width="stretch")


def render_data_table(df: pd.DataFrame) -> None:
    """Styled data-table with sentiment labels."""
    # Replace SVG with text labels + colored dots for the dataframe
    label_map = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}
    display = df[["Headline", "Publisher", "Score", "Sentiment"]].copy()
    display.insert(0, " ", display["Sentiment"].map(label_map))

    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        height=400,
        column_config={
            " ":          st.column_config.TextColumn("",          width="small"),
            "Headline":   st.column_config.TextColumn("Headline",  width="large"),
            "Publisher":   st.column_config.TextColumn("Source",    width="medium"),
            "Score":       st.column_config.NumberColumn("Score",   format="%.4f", width="small"),
            "Sentiment":   st.column_config.TextColumn("Sentiment", width="small"),
        },
    )


def render_footer() -> None:
    """Minimal footer."""
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
            placeholder="e.g.  AAPL,  TSLA,  NVDA,  MSFT",
            label_visibility="collapsed",
        )
        analyse_clicked: bool = st.button(
            "Analyze Sentiment",
            width="stretch",
            type="primary",
        )

    # ── Guard clauses ──
    if analyse_clicked and not ticker_input.strip():
        st.warning("Please enter a valid stock ticker symbol.")
        return

    if not analyse_clicked:
        st.markdown(
            "<p class='cta-text'>"
            "Enter a ticker above and press <strong>Analyze Sentiment</strong> to get started."
            "</p>",
            unsafe_allow_html=True,
        )
        render_footer()
        return

    ticker: str = ticker_input.strip().upper()

    # ── Fetch ──
    with st.spinner(f"Fetching the latest headlines for {ticker}…"):
        headlines = fetch_news(ticker)

    if headlines is None:
        st.error(
            f"Could not retrieve news for **{ticker}**. "
            "Double-check the symbol and ensure you have an internet connection."
        )
        return

    # ── Analyse ──
    df = analyze_sentiment(headlines, analyzer)
    if df.empty:
        st.warning("No headlines available to analyse for this ticker.")
        return

    # ── Derived metrics ──
    verdict_label, verdict_svg, verdict_colour, verdict_bg = compute_verdict(df)
    avg_score: float = df["Score"].mean()
    total: int = len(df)
    counts = df["Sentiment"].value_counts()
    pos_pct: float = counts.get("Positive", 0) / total * 100
    neg_pct: float = counts.get("Negative", 0) / total * 100

    # ── Render dashboard ──
    render_verdict(verdict_label, verdict_svg, verdict_colour, verdict_bg, avg_score, ticker)
    st.markdown("")
    render_metrics(total, avg_score, pos_pct, neg_pct)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    chart_col, table_col = st.columns([2, 3], gap="large")
    with chart_col:
        st.markdown(
            f'<div class="section-title">{SVG_CHART} Sentiment Distribution</div>',
            unsafe_allow_html=True,
        )
        render_donut_chart(df)
    with table_col:
        st.markdown(
            f'<div class="section-title">{SVG_TABLE} Headline Breakdown</div>',
            unsafe_allow_html=True,
        )
        render_data_table(df)

    render_footer()


if __name__ == "__main__":
    main()
