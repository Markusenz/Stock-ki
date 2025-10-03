import os
import time
import datetime as dt
from typing import List, Dict

import streamlit as st
import pandas as pd
import yfinance as yf
import feedparser
from bs4 import BeautifulSoup

# --- KI / NLP ---
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# ---------------------------
#          SETUP
# ---------------------------
st.set_page_config(page_title="Stocks + Sentiment (FinBERT)", layout="wide")

@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    """
    FinBERT (ProsusAI/finbert) für Finanz-Sentiment.
    Wird einmalig geladen und dann gecached.
    """
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True, device=device)

@st.cache_data(show_spinner=False, ttl=60*10)
def get_prices(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end)
    return data

@st.cache_data(show_spinner=False, ttl=60*10)
def get_yf_news(ticker: str) -> List[Dict]:
    """
    Versucht, News aus yfinance zu ziehen (Yahoo Finance).
    Falls nicht verfügbar, wird eine leere Liste zurückgegeben.
    """
    try:
        t = yf.Ticker(ticker)
        news = t.news or []
        # normalize
        out = []
        for n in news:
            title = n.get("title") or n.get("title", "")
            link = n.get("link") or n.get("url") or ""
            pub = n.get("providerPublishTime") or n.get("published_at") or None
            if isinstance(pub, (int, float)):  # epoch seconds
                pub_dt = dt.datetime.utcfromtimestamp(pub)
            else:
                pub_dt = None
            out.append({"title": title, "link": link, "published": pub_dt})
        return out
    except Exception:
        return []

@st.cache_data(show_spinner=False, ttl=60*10)
def get_rss_news(ticker: str, max_items: int = 20) -> List[Dict]:
    """
    Fallback: Yahoo Finance RSS für das Tickersymbol (nicht offiziell garantiert).
    """
    # Beispiel: https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    feed = feedparser.parse(url)
    out = []
    for entry in feed.entries[:max_items]:
        title = BeautifulSoup(entry.title, "html.parser").get_text()
        link = entry.link
        pub = None
        try:
            pub = dt.datetime(*entry.published_parsed[:6])
        except Exception:
            pass
        out.append({"title": title, "link": link, "published": pub})
    return out

def clean_text(s: str) -> str:
    s = BeautifulSoup(s or "", "html.parser").get_text().strip()
    return " ".join(s.split())

def aggregate_finbert(scores: List[Dict[str, float]]) -> float:
    """
    Erwartet FinBERT-Klassen: 'positive', 'negative', 'neutral'
    Gibt einen Score in [-1, 1] zurück (positiv minus negativ).
    """
    by_label = {d["label"].lower(): d["score"] for d in scores}
    pos = by_label.get("positive", 0.0)
    neg = by_label.get("negative", 0.0)
    # neutral ist informativ, geht aber nicht direkt in den Score
    return pos - neg

# ---------------------------
#          UI
# ---------------------------
st.title("📈 Börsenkurse + 🧠 Sentimentanalyse (FinBERT)")

with st.sidebar:
    st.header("Einstellungen")
    default_symbol = "AAPL"
    ticker = st.text_input("Ticker (z. B. AAPL, MSFT, TSLA, ^GSPC)", value=default_symbol).upper().strip()
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=dt.date.today() - dt.timedelta(days=365))
    with col2:
        end_date = st.date_input("Ende", value=dt.date.today() + dt.timedelta(days=1))
    max_headlines = st.slider("Anzahl Headlines für Sentiment", 5, 50, 20, step=5)
    do_rss_fallback = st.checkbox("RSS-Fallback nutzen, wenn keine YF-News", value=True)
    run_inference = st.button("Analysieren")

# ---------------------------
#     Preise laden/anzeigen
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
            st.metric(label=f"Letzter Schlusskurs ({last_row.index[-1].date()})", value=f"{last_close:,.2f}")

# ---------------------------
#  News + Sentiment Analyse
# ---------------------------
st.markdown("---")
st.subheader("📰 News & 🧠 Sentiment")

if run_inference and ticker:
    with st.spinner("Lade News …"):
        news = get_yf_news(ticker)
        if (not news) and do_rss_fallback:
            news = get_rss_news(ticker)

    if not news:
        st.info("Keine News gefunden.")
    else:
        # Headlines vorbereiten
        df_news = pd.DataFrame(news).dropna(subset=["title"])
        df_news["title"] = df_news["title"].apply(clean_text)
        df_news = df_news[df_news["title"].str.len() > 0].head(max_headlines)

        if df_news.empty:
            st.info("Keine verwertbaren Headlines.")
        else:
            nlp = load_sentiment_pipeline()
            texts = df_news["title"].tolist()

            # Batch-Inferenz (schneller & stabiler)
            all_scores = nlp(texts, truncation=True, max_length=128, batch_size=16)

            # Ergebnisse anhängen
            agg_scores = []
            labels = []
            for scores in all_scores:
                # scores = [{'label': 'positive', 'score': ...}, ...]
                agg = aggregate_finbert(scores)
                agg_scores.append(agg)
                # stärkste Klasse
                best = max(scores, key=lambda x: x["score"])
                labels.append(best["label"].capitalize())

            df_news["sentiment_score"] = agg_scores
            df_news["sentiment_label"] = labels

            # Gesamtindex [-1..1]
            overall = float(pd.Series(agg_scores).mean()) if len(agg_scores) else 0.0

            colA, colB = st.columns([1, 1])
            with colA:
                st.metric("Aggregierter Sentiment-Index", f"{overall:+.3f}",
                          help="Mittelwert (positiv − negativ) über alle Headlines; Bereich ≈ [-1, 1]")
            with colB:
                dist = pd.Series(labels).value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0).astype(int)
                st.write("Verteilung Labels")
                st.bar_chart(dist)

            # Tabelle anzeigen
            show_cols = ["published", "title", "sentiment_label", "sentiment_score", "link"]
            st.dataframe(df_news[show_cols].rename(columns={
                "published": "Datum",
                "title": "Headline",
                "sentiment_label": "Label",
                "sentiment_score": "Score",
                "link": "Quelle"
            }), use_container_width=True)

            st.caption("Hinweis: FinBERT ist für Finanztexte trainiert; Headlines sind kurz und können mehrdeutig sein. Kontext prüfen!")

# ---------------------------
#           Tipps
# ---------------------------
with st.expander("⚙️ Tipps für Produktion / Qualität"):
    st.markdown("""
- **Caching** ist aktiv (`@st.cache_resource` für Modell, `@st.cache_data` für Daten).  
- **Batch-Größe & max_length** bei Inferenz angepasst – Headlines sind kurz, `max_length=128` reicht.
- **Fallback** auf RSS, wenn `yfinance.Ticker.news` nichts liefert (nicht garantiert).
- **Rate Limits**: Viele Ticker/Zeiträume langsam abfragen; ggf. News-API mit Key (z. B. NewsAPI) integrieren.
- **Mehrsprachigkeit**: FinBERT ist englisch; für DE-News vorher übersetzen (z. B. `argos-translate` oder Cloud-Übersetzer).
- **GPU**: Wenn verfügbar, wird automatisch genutzt.
""")
