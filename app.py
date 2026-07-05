"""
================================================================
APLIKACJA: Analizator Sygnałów Giełdowych
================================================================
Prosta aplikacja webowa (Streamlit) - wpisujesz ticker, klikasz
"Analizuj", dostajesz sygnał BUY/SELL/HOLD + wykres + uzasadnienie.

Uruchomienie:
    pip3 install streamlit yfinance pandas numpy plotly
    streamlit run app.py

Otworzy się w przeglądarce na http://localhost:8501
================================================================
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

SIGNAL_THRESHOLD = 3

# ============================================================
# WSKAŹNIKI (identyczna logika jak w signal_analyzer.py)
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
        signal = "BUY"
    elif score <= -SIGNAL_THRESHOLD:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {"signal": signal, "score": score, "price": last["Close"], "reasons": reasons, "df": df}


# ============================================================
# UI
# ============================================================

st.set_page_config(page_title="Analizator Sygnałów", page_icon="📈", layout="centered")

st.title("📈 Analizator Sygnałów Giełdowych")
st.caption("Wpisz ticker, wybierz interwał i sprawdź sygnał techniczny na żywo.")

col1, col2 = st.columns([2, 1])
with col1:
    ticker = st.text_input(
        "Instrument (ticker)",
        value="AAPL",
        placeholder="np. AAPL, BTC-USD, XRP-USD, TSLA, EURUSD=X",
    ).strip().upper()
with col2:
    timeframe = st.selectbox("Interwał świec", ["15m", "1h", "1d"], index=1)

period_map = {"15m": "60d", "1h": "180d", "1d": "2y"}

analyze = st.button("🔍 Analizuj", type="primary", use_container_width=True)

if analyze and ticker:
    with st.spinner(f"Pobieram dane dla {ticker}..."):
        try:
            df = yf.download(
                ticker, period=period_map[timeframe], interval=timeframe,
                progress=False, auto_adjust=True,
            )
            if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
                df.columns = df.columns.get_level_values(0)
        except Exception as e:
            st.error(f"Błąd pobierania danych: {e}")
            df = None

    if df is None or df.empty:
        st.error(f"Nie znaleziono danych dla '{ticker}'. Sprawdź czy ticker jest poprawny (format Yahoo Finance).")
    elif len(df) < 55:
        st.warning(f"Za mało danych historycznych ({len(df)} świec) do policzenia wszystkich wskaźników. Spróbuj dłuższy interwał (np. 1d).")
    else:
        result = generate_signal(df)

        # --- Karta z sygnałem ---
        signal_color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}[result["signal"]]
        c1, c2, c3 = st.columns(3)
        c1.metric("Cena", f"{result['price']:.4f}")
        c2.metric("Sygnał", f"{signal_color} {result['signal']}")
        c3.metric("Punkty", f"{result['score']:+d}")

        st.progress(min(max((result["score"] + 5) / 10, 0), 1))

        # --- Wykres świecowy ---
        plot_df = result["df"].tail(80)
        fig = go.Figure(data=[go.Candlestick(
            x=plot_df.index, open=plot_df["Open"], high=plot_df["High"],
            low=plot_df["Low"], close=plot_df["Close"], name=ticker,
        )])
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA20"], name="SMA20", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA50"], name="SMA50", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_upper"], name="BB górna", line=dict(width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_lower"], name="BB dolna", line=dict(width=1, dash="dot")))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- Uzasadnienie ---
        st.subheader("Uzasadnienie sygnału")
        icon = {"bull": "🟢", "bear": "🔴", "neutral": "⚪"}
        for category, text, direction in result["reasons"]:
            st.write(f"{icon[direction]} **{category}** — {text}")

        st.caption(
            f"Próg sygnału: ±{SIGNAL_THRESHOLD} punktów. "
            "To narzędzie analityczne, nie porada inwestycyjna."
        )
elif analyze:
    st.warning("Wpisz ticker instrumentu.")
