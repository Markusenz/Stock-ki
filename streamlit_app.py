# app.py
# -----------------------------------------
# 📈 Börsenkurse + 🧠 Sentimentanalyse (Cloud-freundlich)
# - Multi-Source-News: yfinance.news, Yahoo RSS, Google News RSS
# - VADER als schneller Default (keine großen Downloads)
# - Optional FinBERT via USE_FINBERT=1 (kann in Streamlit Cloud langsam sein)
# -----------------------------------------

import os
import re
import datetime as dt
from typing import List, Dict, Optional

import streamlit as st
import pandas as pd
import yfinance as yf
import feedparser
from bs4 import BeautifulSoup

# ---------------------------
#          SETUP
# ---------------------------
st.set_page_config(page_title="Stocks + Sentiment", layout="wide")
st.title("📈 Börsenkurse + 🧠 Sentimentanalyse")

# ---------------------------
#       UTILITIES
# ---------------------------
def _clean_html_text(s: str) -> str:
    return " ".join(BeautifulSoup((s or ""), "html.parser").get_text().split())

@st.cache_data(ttl=600, show_spinner=False)
def get_prices(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    return yf.download(ticker, start=start, end=end)

@st.cache_data(ttl=600, show_spinner=False)
def get_company_name(ticker: str) -> Optional[str]:
    """Bestenfalls LongName/ShortName – je nach yfinance-Version."""
    try:
        info = yf.Ticker(ticker).info or {}
        return info.get("longName") or info.get("shortName")
    except Exception:
        return None

# ---------------------------
#     NEWS QUELLEN
# ---------------------------
@st.cache_data(ttl=600, show_spinner=False)
def news_from_yfinance(ticker: str) -> List[Dict]:
    try:
        t = yf.Ticker(ticker)
        items = t.news or []
        out = []
        for n in items:
            title = n.get("title") or ""
            link = n.get("link") or n.get("url") or ""
            ts = n.get("providerPublishTime")
            pub = dt.datetime.utcfromtimestamp(ts) if isinstance(ts, (int, float)) else None
            out.append({"title": title, "link": link, "published": pub, "source": "yfinance"})
        return out
    except Exception:
        return []

@st.cache_data(ttl=600, show_spinner=False)
def news_from_yahoo_rss(ticker: str) -> List[Dict]:
    # Beispiel: https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    feed = feedparser.parse(url)
    out = []
    for e in feed.entries:
        try:
            pub = dt.datetime(*e.published_parsed[:6])
        except Exception:
            pub = None
        out.append({
            "title": _clean_html_text(getattr(e, "title", "")),
            "link": getattr(e, "link", ""),
            "published": pub,
            "source": "yahoo_rss"
        })
    return out

@st.cache_data(ttl=600, show_spinner=False)
def news_from_google(query: str, days: int = 14, lang: str = "de", region: str = "DE") -> List[Dict]:
    """
    Google News RSS – robust, steuerbar über Sprache/Region und Zeitfenster.
    Beispiele:
    https://news.google.com/rss/search?q=AAPL%20when:14d&hl=de&gl=DE&ceid=DE:de
    """
    q = re.sub(r"\s+", "+", query.strip())
    url = f"https://news.google.com/rss/search?q={q}+when:{days}d&hl={lang}&gl={region}&ceid={region}:{lang}"
    feed = feedparser.parse(url)
    out = []
    for e in feed.entries:
        try:
            pub = dt.datetime(*e.published_parsed[:6])
        except Exception:
            pub = None
        out.append({
            "title": _clean_html_text(getattr(e, "title", "")),
            "link": getattr(e, "link", ""),
            "published": pub,
            "source": "google_news"
        })
    return out

@st.cache_data(ttl=600, show_spinner=False)
def get_headlines(ticker: str, max_items: int = 30, window_days: int = 14,
                  lang: str = "de", region: str = "DE") -> pd.DataFrame:
    """
    Aggregiert mehrere Quellen, dedupliziert und sortiert.
    Reihenfolge: yfinance → Yahoo RSS → Google News (Ticker) → Google News (Firmenname)
    """
    collected: List[Dict] = []
    name = get_company_name(ticker)

    collected += news_from_yfinance(ticker)
    collected += news_from_yahoo_rss(ticker)
    collected += news_from_google(ticker, days=window_days, lang=lang, region=region)
    if name:
        collected += news_from_google(name, days=window_days, lang=lang, region=region)

    if not collected:
        return pd.DataFrame(columns=["published", "title", "link", "source"])

    df = pd.DataFrame(collected)
    df["title"] = df["title"].astype(str).map(_clean_html_text)
    df = df[df["title"].str.len() > 0]

    # Sortierung nach Datum (neu → alt), dann Duplikate raus
    df = df.sort_values(by=["published"], ascending=False, na_position="last")
    df = df.drop_duplicates(subset=["title"])
    df = df.drop_duplicates(subset=["link"])

    # Zeitfenster anwenden (wo Datum vorhanden)
    now = dt.datetime.utcnow()
    cutoff = now - dt.timedelta(days=window_days)
    if "published" in df and df["published"].notna().any():
        df = df[(df["published"].isna()) | (df["published"] >= cutoff)]

    return df.head(max_items)

# ---------------------------
#   SENTIMENT ENGINES
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()

@st.cache_resource(show_spinner=True)
def load_finbert():
    # Schwere Imports NUR hier, damit die App sofort rendert
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    name = "ProsusAI/finbert"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-classification", model=mdl, tokenizer=tok, return_all_scores=True, device=device)

def vader_score(analyzer, text: str):
    vs = analyzer.polarity_scores(text)
    score = float(vs["pos"] - vs["neg"])  # [-1..1] ungefähr
    lab = "Positive" if vs["compound"] >= 0.05 else "Negative" if vs["compound"] <= -0.05 else "Neutral"
    return score, lab

def finbert_aggregate(scores: List[Dict[str, float]]):
    by = {d["label"].lower(): d["score"] for d in scores}
    score = by.get("positive", 0.0) - by.get("negative", 0.0)
    label = max(scores, key=lambda x: x["score"])["label"].capitalize()
    return score, label

# ---------------------------
#          SIDEBAR
# ---------------------------
with st.sidebar:
    st.header("Einstellungen")
    ticker = st.text_input("Ticker (z. B. AAPL, MSFT, TSLA, SAP, ^GSPC)", "AAPL").upper().strip()

    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start", value=dt.date.today() - dt.timedelta(days=365))
    with c2:
        end_date = st.date_input("Ende", value=dt.date.today())

    max_headlines = st.slider("Headlines", 5, 50, 20, step=5)
    window_days = st.slider("Zeitfenster (Tage)", 3, 60, 14, step=1)

    # Sprache/Region für Google News
    lang = st.selectbox("Sprache (Google News)", ["de", "en"], index=0)
    region = st.selectbox("Region (Google News)", ["DE", "US", "GB"], index=0)

    use_finbert_toggle = st.toggle("FinBERT verwenden (langsam in Cloud!)", value=False)
    # Nur erlauben, wenn Umgebungsvariable gesetzt → verhindert versehentliche lange Startzeiten
    use_finbert = use_finbert_toggle and (os.getenv("USE_FINBERT", "0") == "1")

    run = st.button("Analysieren")

# ---------------------------
#        KURSDATEN
# ---------------------------
if ticker:
    with st.spinner("Lade Kursdaten …"):
        prices = get_prices(ticker, start_date, end_date)

    if prices.empty:
        st.warning("Keine Kursdaten gefunden. Prüfe Ticker/Zeitraum.")
    else:
        st.subheader(f"Kursverlauf: {ticker}")
        st.line_chart(prices["Close"])

        last_row = prices.tail(1)
        if not last_row.empty:
            last_close = float(last_row["Close"].iloc[0])
            st.metric(label=f"Letzter Schlusskurs ({last_row.index[-1].date()})",
                      value=f"{last_close:,.2f}")

st.markdown("---")
st.subheader("📰 News & Sentiment")

# ---------------------------
#   NEWS + SENTIMENT LOGIK
# ---------------------------
if run and ticker:
    with st.spinner("Lade Headlines …"):
        df_news = get_headlines(
            ticker=ticker,
            max_items=max_headlines,
            window_days=window_days,
            lang=lang,
            region=region
        )

    if df_news.empty:
        st.info("Keine verwertbaren Headlines (prüfe Ticker/Zeitraum oder ändere Sprache/Region).")
    else:
        texts = df_news["title"].tolist()

        # Engine wählen (FinBERT nur wenn explizit erlaubt)
        if use_finbert:
            nlp = load_finbert()
            raw = nlp(texts, truncation=True, max_length=128, batch_size=16)
            scores, labels = zip(*(finbert_aggregate(r) for r in raw))
            engine = "FinBERT"
        else:
            vad = load_vader()
            scores, labels = zip(*(vader_score(vad, t) for t in texts))
            engine = "VADER"

        df_news["sentiment_score"] = scores
        df_news["sentiment_label"] = labels
        overall = float(pd.Series(scores).mean()) if len(df_news) else 0.0

        cA, cB = st.columns(2)
        with cA:
            st.metric("Aggregierter Sentiment-Index", f"{overall:+.3f}",
                      help=f"Engine: {engine}")
        with cB:
            st.write("Verteilung")
            st.bar_chart(pd.Series(labels).value_counts())

        st.dataframe(
            df_news[["published", "title", "sentiment_label", "sentiment_score", "link", "source"]]
                .rename(columns={
                    "published": "Datum",
                    "title": "Headline",
                    "sentiment_label": "Label",
                    "sentiment_score": "Score",
                    "link": "Quelle",
                    "source": "Kanal"
                }),
            use_container_width=True
        )

        st.caption("Hinweis: VADER ist leichtgewichtig und schnell. FinBERT nur aktiv, wenn Schalter **und** `USE_FINBERT=1` gesetzt ist.")