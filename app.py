"""
================================================================
APLIKACJA: Analizator Sygnałów Giełdowych - Watchlist
================================================================
Wpisujesz listę tickerów, aplikacja sama się odświeża w ustalonym
interwale i pokazuje tabelę sygnałów dla wszystkich naraz.

Uruchomienie:
    pip3 install streamlit yfinance pandas numpy plotly streamlit-autorefresh
    streamlit run app.py
================================================================
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

SIGNAL_THRESHOLD = 3

# ============================================================
# WSKAŹNIKI
# ============================================================

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def bollinger_bands(series: pd.Series, period=20, num_std=2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return sma + num_std * std, sma, sma - num_std * std


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = rsi(df["Close"])
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(df["Close"])
    df["BB_upper"], df["BB_mid"], df["BB_lower"] = bollinger_bands(df["Close"])
    return df


def detect_candlestick_patterns(df: pd.DataFrame) -> list:
    patterns = []
    if len(df) < 2:
        return patterns
    last, prev = df.iloc[-1], df.iloc[-2]
    o, h, l, c = last["Open"], last["High"], last["Low"], last["Close"]
    po, pc = prev["Open"], prev["Close"]
    body = abs(c - o)
    candle_range = h - l if h != l else 1e-9
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    if pc < po and c > o and c > po and o < pc:
        patterns.append(("bullish_engulfing", "bull"))
    if pc > po and c < o and c < po and o > pc:
        patterns.append(("bearish_engulfing", "bear"))
    if lower_wick > body * 2 and upper_wick < body * 0.5 and c > o:
        patterns.append(("hammer", "bull"))
    if upper_wick > body * 2 and lower_wick < body * 0.5 and c < o:
        patterns.append(("shooting_star", "bear"))
    if body < candle_range * 0.1:
        patterns.append(("doji", "neutral"))
    return patterns


def generate_signal(df: pd.DataFrame) -> dict:
    df = add_indicators(df)
    last, prev = df.iloc[-1], df.iloc[-2]
    score = 0
    reasons = []

    if pd.notna(last["RSI"]):
        if last["RSI"] < 30:
            score += 2
            reasons.append(("RSI", f"RSI={last['RSI']:.1f} — wyprzedanie (rynek może się odbić w górę)", "bull"))
        elif last["RSI"] > 70:
            score -= 2
            reasons.append(("RSI", f"RSI={last['RSI']:.1f} — wykupienie (rynek może skorygować w dół)", "bear"))
        else:
            reasons.append(("RSI", f"RSI={last['RSI']:.1f} — strefa neutralna", "neutral"))

    if pd.notna(last["MACD"]) and pd.notna(prev["MACD"]):
        if prev["MACD"] < prev["MACD_signal"] and last["MACD"] > last["MACD_signal"]:
            score += 2
            reasons.append(("MACD", "Przecięcie linii sygnału w górę — momentum wzrostowe", "bull"))
        elif prev["MACD"] > prev["MACD_signal"] and last["MACD"] < last["MACD_signal"]:
            score -= 2
            reasons.append(("MACD", "Przecięcie linii sygnału w dół — momentum spadkowe", "bear"))
        else:
            reasons.append(("MACD", "Brak świeżego przecięcia", "neutral"))

    if pd.notna(last["SMA20"]) and pd.notna(last["SMA50"]):
        if last["SMA20"] > last["SMA50"]:
            score += 1
            reasons.append(("Trend", "SMA20 > SMA50 — trend wzrostowy", "bull"))
        else:
            score -= 1
            reasons.append(("Trend", "SMA20 < SMA50 — trend spadkowy", "bear"))

    if pd.notna(last["BB_lower"]) and last["Close"] < last["BB_lower"]:
        score += 1
        reasons.append(("Bollinger", "Cena poniżej dolnej bandy — potencjalne odbicie", "bull"))
    elif pd.notna(last["BB_upper"]) and last["Close"] > last["BB_upper"]:
        score -= 1
        reasons.append(("Bollinger", "Cena powyżej górnej bandy — potencjalna korekta", "bear"))

    for name, direction in detect_candlestick_patterns(df):
        if direction == "bull":
            score += 1
            reasons.append(("Formacja", f"{name} — sygnał bullish", "bull"))
        elif direction == "bear":
            score -= 1
            reasons.append(("Formacja", f"{name} — sygnał bearish", "bear"))
        else:
            reasons.append(("Formacja", f"{name} — niezdecydowanie", "neutral"))

    if score >= SIGNAL_THRESHOLD:
        signal = "KUP"
    elif score <= -SIGNAL_THRESHOLD:
        signal = "SPRZEDAJ"
    else:
        signal = "CZEKAJ"

    return {"signal": signal, "score": score, "price": last["Close"], "reasons": reasons, "df": df}


@st.cache_data(ttl=30, show_spinner=False)
def fetch_and_analyze(ticker: str, timeframe: str, period: str):
    df = yf.download(ticker, period=period, interval=timeframe, progress=False, auto_adjust=True)
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    if df.empty or len(df) < 55:
        return None
    return generate_signal(df)


# ============================================================
# UI
# ============================================================

st.set_page_config(page_title="Analizator Sygnałów", page_icon="📈", layout="wide")

st.title("📈 Analizator Sygnałów Giełdowych — Watchlist")
st.caption("Lista obserwowanych instrumentów, sama się odświeża. Kliknij ticker poniżej, aby zobaczyć szczegóły.")

with st.sidebar:
    st.header("Ustawienia")
    default_tickers = "AAPL\nMSFT\nNVDA\nBTC-USD\nETH-USD\nXRP-USD\nLBW.WA"
    tickers_input = st.text_area(
        "Lista tickerów (jeden na linię)",
        value=default_tickers,
        height=180,
    )
    timeframe = st.selectbox("Interwał świec", ["15m", "1h", "1d"], index=1)
    refresh_seconds = st.number_input(
        "Auto-odśwież co (sekund)", min_value=15, max_value=3600, value=60, step=15
    )
    st.caption("To narzędzie analityczne, nie porada inwestycyjna.")

period_map = {"15m": "60d", "1h": "180d", "1d": "2y"}
tickers = [t.strip().upper() for t in tickers_input.splitlines() if t.strip()]

st_autorefresh(interval=refresh_seconds * 1000, key="refresh_watchlist")

st.caption(f"Ostatnie odświeżenie danych: {pd.Timestamp.now().strftime('%H:%M:%S')} · odświeża się co {refresh_seconds}s")

rows = []
results_by_ticker = {}

with st.spinner("Analizuję listę instrumentów..."):
    for ticker in tickers:
        try:
            result = fetch_and_analyze(ticker, timeframe, period_map[timeframe])
        except Exception as e:
            result = None
        if result is None:
            rows.append({"Ticker": ticker, "Cena": None, "Sygnał": "BŁĄD", "Punkty": None})
            continue
        results_by_ticker[ticker] = result
        rows.append({
            "Ticker": ticker,
            "Cena": round(result["price"], 4),
            "Sygnał": result["signal"],
            "Punkty": result["score"],
        })

if not rows:
    st.info("Dodaj przynajmniej jeden ticker w panelu po lewej.")
else:
    df_table = pd.DataFrame(rows)

    def color_signal(val):
        if val == "KUP":
            return "background-color: #d4f8d4"
        elif val == "SPRZEDAJ":
            return "background-color: #f8d4d4"
        elif val == "BŁĄD":
            return "background-color: #eeeeee; color: #999999"
        return "background-color: #fff8d4"

    try:
        styled = df_table.style.map(color_signal, subset=["Sygnał"])
    except AttributeError:
        styled = df_table.style.applymap(color_signal, subset=["Sygnał"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.subheader("Szczegóły instrumentu")
    valid_tickers = [t for t in tickers if t in results_by_ticker]
    if valid_tickers:
        selected = st.selectbox("Wybierz ticker do analizy szczegółowej", valid_tickers)
        result = results_by_ticker[selected]

        signal_color = {"KUP": "🟢", "SPRZEDAJ": "🔴", "CZEKAJ": "🟡"}[result["signal"]]
        c1, c2, c3 = st.columns(3)
        c1.metric("Cena", f"{result['price']:.4f}")
        c2.metric("Sygnał", f"{signal_color} {result['signal']}")
        c3.metric("Punkty", f"{result['score']:+d}")

        plot_df = result["df"].tail(80)
        fig = go.Figure(data=[go.Candlestick(
            x=plot_df.index, open=plot_df["Open"], high=plot_df["High"],
            low=plot_df["Low"], close=plot_df["Close"], name=selected,
        )])
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA20"], name="SMA20", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA50"], name="SMA50", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_upper"], name="BB górna", line=dict(width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_lower"], name="BB dolna", line=dict(width=1, dash="dot")))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Uzasadnienie sygnału**")
        icon = {"bull": "🟢", "bear": "🔴", "neutral": "⚪"}
        for category, text, direction in result["reasons"]:
            st.write(f"{icon[direction]} **{category}** — {text}")
