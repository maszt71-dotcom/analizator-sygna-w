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


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["RSI"] = rsi(df["Close"])
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(df["Close"])
    df["BB_upper"], df["BB_mid"], df["BB_lower"] = bollinger_bands(df["Close"])
    df["ATR"] = atr(df)
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
    else:
        reasons.append(("Bollinger", "Cena w normalnym zakresie (między bandami)", "neutral"))

    patterns = detect_candlestick_patterns(df)
    if patterns:
        for name, direction in patterns:
            if direction == "bull":
                score += 1
                reasons.append(("Formacja", f"{name} — sygnał bullish", "bull"))
            elif direction == "bear":
                score -= 1
                reasons.append(("Formacja", f"{name} — sygnał bearish", "bear"))
            else:
                reasons.append(("Formacja", f"{name} — niezdecydowanie", "neutral"))
    else:
        reasons.append(("Formacja", "Brak rozpoznanej formacji świecowej", "neutral"))

    if score >= SIGNAL_THRESHOLD:
        signal = "KUP"
    elif score <= -SIGNAL_THRESHOLD:
        signal = "SPRZEDAJ"
    else:
        signal = "CZEKAJ"

    return {"signal": signal, "score": score, "price": last["Close"], "reasons": reasons, "df": df}


def run_backtest(
    df: pd.DataFrame,
    use_trend_filter: bool = True,
    stop_loss_pct: float = 8.0,
    take_profit_pct: float = 15.0,
):
    """
    Symuluje realne transakcje wg dopracowanej strategii:
    - Wejście na sygnale KUP, ale TYLKO gdy cena jest nad SMA200 (filtr głównego trendu),
      żeby nie kupować wbrew długoterminowemu kierunkowi rynku.
    - Wyjście: automatyczny Stop Loss / Take Profit (żeby nie czekać bezczynnie na
      spóźniony sygnał SPRZEDAJ), albo sam sygnał SPRZEDAJ - co nastąpi pierwsze.
    Zwraca (tabela_transakcji, czy_pozycja_wciąż_otwarta, data_wejścia_otwartej, cena_wejścia_otwartej).
    """
    df_ind = add_indicators(df)
    trades = []
    in_position = False
    entry_price = None
    entry_date = None

    min_bars = 200 if use_trend_filter else 55

    for i in range(min_bars, len(df_ind)):
        last = df_ind.iloc[i]
        prev = df_ind.iloc[i - 1]

        # --- Sprawdzenie wyjścia z pozycji (Stop Loss / Take Profit) na bieżącej świecy ---
        if in_position:
            stop_price = entry_price * (1 - stop_loss_pct / 100)
            target_price = entry_price * (1 + take_profit_pct / 100)
            hit_stop = last["Low"] <= stop_price
            hit_target = last["High"] >= target_price

            exit_price = None
            exit_reason = None
            if hit_stop:
                exit_price, exit_reason = stop_price, "Stop Loss"
            elif hit_target:
                exit_price, exit_reason = target_price, "Take Profit"

            if exit_price is not None:
                exit_date = df_ind.index[i]
                ret_pct = (exit_price - entry_price) / entry_price * 100
                trades.append({
                    "Data kupna": entry_date,
                    "Cena kupna": round(entry_price, 4),
                    "Data sprzedaży": exit_date,
                    "Cena sprzedaży": round(exit_price, 4),
                    "Dni w pozycji": (exit_date - entry_date).days,
                    "Zwrot %": round(ret_pct, 2),
                    "Trafiony": ret_pct > 0,
                    "Wyjście": exit_reason,
                })
                in_position = False

        # --- Liczenie sygnału (do wejścia i do sygnałowego wyjścia) ---
        score = 0

        if pd.notna(last["RSI"]):
            if last["RSI"] < 30:
                score += 2
            elif last["RSI"] > 70:
                score -= 2

        if pd.notna(last["MACD"]) and pd.notna(prev["MACD"]):
            if prev["MACD"] < prev["MACD_signal"] and last["MACD"] > last["MACD_signal"]:
                score += 2
            elif prev["MACD"] > prev["MACD_signal"] and last["MACD"] < last["MACD_signal"]:
                score -= 2

        if pd.notna(last["SMA20"]) and pd.notna(last["SMA50"]):
            score += 1 if last["SMA20"] > last["SMA50"] else -1

        if pd.notna(last["BB_lower"]) and last["Close"] < last["BB_lower"]:
            score += 1
        elif pd.notna(last["BB_upper"]) and last["Close"] > last["BB_upper"]:
            score -= 1

        for name, direction in detect_candlestick_patterns(df_ind.iloc[i - 1:i + 1]):
            if direction == "bull":
                score += 1
            elif direction == "bear":
                score -= 1

        if score >= SIGNAL_THRESHOLD:
            signal = "KUP"
        elif score <= -SIGNAL_THRESHOLD:
            signal = "SPRZEDAJ"
        else:
            signal = "CZEKAJ"

        # --- Sygnałowe wyjście (jeśli SL/TP nie zadziałał wcześniej na tej świecy) ---
        if in_position and signal == "SPRZEDAJ":
            exit_price = last["Close"]
            exit_date = df_ind.index[i]
            ret_pct = (exit_price - entry_price) / entry_price * 100
            trades.append({
                "Data kupna": entry_date,
                "Cena kupna": round(entry_price, 4),
                "Data sprzedaży": exit_date,
                "Cena sprzedaży": round(exit_price, 4),
                "Dni w pozycji": (exit_date - entry_date).days,
                "Zwrot %": round(ret_pct, 2),
                "Trafiony": ret_pct > 0,
                "Wyjście": "Sygnał SPRZEDAJ",
            })
            in_position = False

        # --- Wejście w pozycję ---
        if not in_position and signal == "KUP":
            trend_ok = True
            if use_trend_filter:
                trend_ok = pd.notna(last["SMA200"]) and last["Close"] > last["SMA200"]
            if trend_ok:
                in_position = True
                entry_price = last["Close"]
                entry_date = df_ind.index[i]

    return pd.DataFrame(trades), in_position, entry_date, entry_price


@st.cache_data(ttl=3600, show_spinner=False)
def get_accuracy(ticker: str):
    """Szybki backtest w tle (transakcje KUP->SPRZEDAJ z ostatniego roku) - zwraca % trafionych transakcji."""
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 60:
            return None
        trades, _, _, _ = run_backtest(df)
        if trades.empty:
            return None
        return round(trades["Trafiony"].mean() * 100, 0)
    except Exception:
        return None


import requests

@st.cache_data(ttl=3600, show_spinner=False)
def resolve_ticker(query: str):
    """Próbuje znaleźć ticker na podstawie nazwy firmy/instrumentu przez wyszukiwarkę Yahoo Finance."""
    query = query.strip()
    if not query:
        return None
    try:
        resp = requests.get(
            "https://query1.finance.yahoo.com/v1/finance/search",
            params={"q": query, "quotesCount": 5, "newsCount": 0},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=6,
        )
        data = resp.json()
        quotes = data.get("quotes", [])
        preferred_types = ("EQUITY", "CRYPTOCURRENCY", "ETF", "INDEX", "CURRENCY")
        for q in quotes:
            if q.get("quoteType") in preferred_types and q.get("symbol"):
                return q["symbol"], q.get("shortname") or q.get("longname") or q["symbol"]
        if quotes and quotes[0].get("symbol"):
            return quotes[0]["symbol"], quotes[0].get("shortname") or quotes[0]["symbol"]
    except Exception:
        pass
    return None


@st.cache_data(ttl=30, show_spinner=False)
def fetch_and_analyze(query: str, timeframe: str, period: str):
    """Analizuje instrument. Najpierw próbuje wprost jako ticker, potem szuka po nazwie."""
    ticker_used = query.strip().upper()
    display_name = ticker_used
    df = yf.download(ticker_used, period=period, interval=timeframe, progress=False, auto_adjust=True)
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    if df.empty or len(df) < 55:
        resolved = resolve_ticker(query)
        if resolved:
            ticker_used, display_name = resolved
            df = yf.download(ticker_used, period=period, interval=timeframe, progress=False, auto_adjust=True)
            if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
                df.columns = df.columns.get_level_values(0)

    if df.empty or len(df) < 55:
        return None

    result = generate_signal(df)
    result["ticker_used"] = ticker_used
    result["display_name"] = display_name
    return result


# ============================================================
# UI
# ============================================================

st.set_page_config(page_title="Analizator Sygnałów", page_icon="📈", layout="wide")

st.title("📈 Analizator Sygnałów Giełdowych")

# Domyślna lista instrumentów do automatycznego skanowania (Top Sygnałów) - edytowalna w sidebarze
DEFAULT_UNIVERSE_TEXT = """AAPL
MSFT
GOOGL
AMZN
NVDA
META
TSLA
JPM
V
UNH
XOM
JNJ
WMT
PG
MA
HD
CVX
MRK
ABBV
KO
PEP
COST
AVGO
ADBE
CSCO
CRM
NFLX
AMD
INTC
ORCL
BTC-USD
ETH-USD
XRP-USD
SOL-USD
ADA-USD
DOGE-USD
BNB-USD
AVAX-USD
PKN.WA
PKO.WA
PZU.WA
KGH.WA
CDR.WA
DNP.WA
LPP.WA
ALE.WA
PGE.WA
SPL.WA
LBW.WA
JSW.WA"""

tab1, tab2, tab3 = st.tabs(["📋 Moja Watchlista", "🏆 Top Sygnałów", "📊 Backtest"])

with tab1:
    st.caption("Lista obserwowanych instrumentów, sama się odświeża. Kliknij instrument poniżej, aby zobaczyć szczegóły.")

    with st.sidebar:
        st.header("Ustawienia — Watchlista")
        default_tickers = "Apple\nBitcoin\nDino Polska\nSpotify\nXRP\nLubawa"
        tickers_input = st.text_area(
            "Lista instrumentów (jeden na linię) — nazwa firmy albo ticker",
            value=default_tickers,
            height=180,
        )
        st.caption("Możesz wpisać nazwę firmy (np. 'Dino') albo dokładny ticker (np. 'DNP.WA') — aplikacja sama znajdzie właściwy symbol.")
        timeframe = st.selectbox("Interwał świec", ["15m", "1h", "1d"], index=1)
        refresh_seconds = st.number_input(
            "Auto-odśwież co (sekund)", min_value=15, max_value=3600, value=60, step=15
        )
        st.caption("To narzędzie analityczne, nie porada inwestycyjna.")

    period_map = {"15m": "60d", "1h": "180d", "1d": "2y"}
    tickers = [t.strip() for t in tickers_input.splitlines() if t.strip()]

    st_autorefresh(interval=refresh_seconds * 1000, key="refresh_watchlist")

    st.caption(f"Ostatnie odświeżenie danych: {pd.Timestamp.now().strftime('%H:%M:%S')} · odświeża się co {refresh_seconds}s")

    rows = []
    results_by_ticker = {}

    with st.spinner("Analizuję listę instrumentów..."):
        for query in tickers:
            try:
                result = fetch_and_analyze(query, timeframe, period_map[timeframe])
            except Exception:
                result = None
            if result is None:
                rows.append({"Wpisano": query, "Ticker": "—", "Cena": None, "Sygnał": "BŁĄD", "Punkty": None})
                continue
            results_by_ticker[query] = result
            accuracy = get_accuracy(result["ticker_used"])
            rows.append({
                "Wpisano": query,
                "Ticker": result["ticker_used"],
                "Cena": round(result["price"], 4),
                "Sygnał": result["signal"],
                "Punkty": result["score"],
                "Skuteczność %": accuracy if accuracy is not None else "—",
            })

    if not rows:
        st.info("Dodaj przynajmniej jeden instrument w panelu po lewej.")
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
        st.caption(
            "Skuteczność % = jak duży odsetek pełnych transakcji (kupno na sygnale KUP, sprzedaż na "
            "kolejnym sygnale SPRZEDAJ) w ostatnim roku zamknął się zyskiem. Historia nie gwarantuje przyszłości."
        )

        st.subheader("Szczegóły instrumentu")
        valid_tickers = [t for t in tickers if t in results_by_ticker]
        if valid_tickers:
            selected = st.selectbox("Wybierz instrument do analizy szczegółowej", valid_tickers, key="watchlist_select")
            result = results_by_ticker[selected]

            signal_color = {"KUP": "🟢", "SPRZEDAJ": "🔴", "CZEKAJ": "🟡"}[result["signal"]]
            st.caption(f"Rozpoznano jako: **{result['display_name']}** ({result['ticker_used']})")
            c1, c2, c3 = st.columns(3)
            c1.metric("Cena", f"{result['price']:.4f}")
            c2.metric("Sygnał", f"{signal_color} {result['signal']}")
            c3.metric("Punkty", f"{result['score']:+d}")

            plot_df = result["df"].tail(80)
            fig = go.Figure(data=[go.Candlestick(
                x=plot_df.index, open=plot_df["Open"], high=plot_df["High"],
                low=plot_df["Low"], close=plot_df["Close"], name=result["ticker_used"],
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


with tab2:
    st.caption(
        "Automatyczne skanowanie szerokiej listy popularnych instrumentów "
        "(duże spółki USA, krypto, WIG20) — 10 najbardziej wzrostowych i 10 najbardziej spadkowych sygnałów."
    )

    with st.sidebar:
        st.header("Ustawienia — Top Sygnałów")
        universe_input = st.text_area(
            "Lista instrumentów do skanowania (max 50, jeden na linię)",
            value=DEFAULT_UNIVERSE_TEXT,
            height=220,
            key="universe_input",
        )
        universe_lines = [t.strip() for t in universe_input.splitlines() if t.strip()]
        if len(universe_lines) > 50:
            st.warning(f"Wpisano {len(universe_lines)} pozycji — używam pierwszych 50.")
            universe_lines = universe_lines[:50]
        top_timeframe = st.selectbox("Interwał świec (Top Sygnałów)", ["15m", "1h", "1d"], index=1, key="top_timeframe")
        top_refresh_minutes = st.number_input(
            "Auto-odśwież co (minut)", min_value=1, max_value=60, value=5, key="top_refresh"
        )
        st.caption(f"Skanowane instrumenty: {len(universe_lines)}/50. Dłuższy interwał odświeżania zapobiega blokadom Yahoo Finance.")

    st_autorefresh(interval=top_refresh_minutes * 60 * 1000, key="refresh_top_signals")

    st.caption(f"Ostatni skan: {pd.Timestamp.now().strftime('%H:%M:%S')} · odświeża się co {top_refresh_minutes} min")

    top_period_map = {"15m": "60d", "1h": "180d", "1d": "2y"}
    scan_results = []

    with st.spinner(f"Skanuję {len(universe_lines)} instrumentów..."):
        for tkr in universe_lines:
            try:
                result = fetch_and_analyze(tkr, top_timeframe, top_period_map[top_timeframe])
            except Exception:
                result = None
            if result is None:
                continue
            accuracy = get_accuracy(result["ticker_used"])
            scan_results.append({
                "Ticker": result["ticker_used"],
                "Nazwa": result["display_name"],
                "Cena": round(result["price"], 4),
                "Sygnał": result["signal"],
                "Punkty": result["score"],
                "Skuteczność %": accuracy if accuracy is not None else "—",
            })

    if not scan_results:
        st.warning("Nie udało się pobrać danych. Spróbuj ponownie za chwilę (możliwy limit zapytań Yahoo Finance).")
    else:
        scan_df = pd.DataFrame(scan_results)

        col_buy, col_sell = st.columns(2)

        with col_buy:
            st.markdown("### 🟢 Top 10 — najbardziej wzrostowe (KUP)")
            top_buy = scan_df.sort_values("Punkty", ascending=False).head(10).reset_index(drop=True)
            top_buy.index = top_buy.index + 1
            st.dataframe(top_buy, use_container_width=True)

        with col_sell:
            st.markdown("### 🔴 Top 10 — najbardziej spadkowe (SPRZEDAJ)")
            top_sell = scan_df.sort_values("Punkty", ascending=True).head(10).reset_index(drop=True)
            top_sell.index = top_sell.index + 1
            st.dataframe(top_sell, use_container_width=True)

        st.caption(
            f"Przeskanowano {len(scan_df)}/{len(universe_lines)} instrumentów. "
            "Ranking wg punktów: RSI, MACD, trend SMA, Bollinger Bands, formacje świecowe. "
            "To narzędzie analityczne, nie porada inwestycyjna."
        )


with tab3:
    st.caption(
        "Sprawdź, jak sygnały KUP/SPRZEDAJ radziły sobie historycznie na danym instrumencie. "
        "To narzędzie analityczne, nie porada inwestycyjna — wyniki z przeszłości nie gwarantują przyszłych."
    )

    with st.sidebar:
        st.header("Ustawienia — Backtest")
        bt_query = st.text_input("Instrument (nazwa firmy albo ticker)", value="AAPL", key="bt_query")
        bt_timeframe = st.selectbox("Interwał świec", ["1h", "1d"], index=1, key="bt_timeframe")
        bt_period_map = {"1h": "730d", "1d": "5y"}
        st.markdown("**Dopracowana strategia:**")
        use_trend_filter = st.checkbox(
            "Filtr trendu (kupuj tylko nad SMA200)", value=True, key="bt_trend_filter"
        )
        stop_loss_pct = st.number_input(
            "Stop Loss %", min_value=1.0, max_value=50.0, value=8.0, step=1.0, key="bt_sl"
        )
        take_profit_pct = st.number_input(
            "Take Profit %", min_value=1.0, max_value=100.0, value=15.0, step=1.0, key="bt_tp"
        )
        st.caption(
            "Wejście na sygnale KUP (nad SMA200, jeśli filtr włączony). "
            "Wyjście: Stop Loss, Take Profit albo sygnał SPRZEDAJ — co nastąpi pierwsze."
        )
        run_bt = st.button("▶️ Uruchom backtest", type="primary", use_container_width=True)

    if run_bt and bt_query:
        with st.spinner(f"Pobieram historię i liczę sygnały dla '{bt_query}'..."):
            ticker_bt = bt_query.strip().upper()
            df_bt = yf.download(
                ticker_bt, period=bt_period_map[bt_timeframe], interval=bt_timeframe,
                progress=False, auto_adjust=True,
            )
            if hasattr(df_bt.columns, "nlevels") and df_bt.columns.nlevels > 1:
                df_bt.columns = df_bt.columns.get_level_values(0)

            display_name_bt = ticker_bt
            if df_bt.empty or len(df_bt) < 60:
                resolved = resolve_ticker(bt_query)
                if resolved:
                    ticker_bt, display_name_bt = resolved
                    df_bt = yf.download(
                        ticker_bt, period=bt_period_map[bt_timeframe], interval=bt_timeframe,
                        progress=False, auto_adjust=True,
                    )
                    if hasattr(df_bt.columns, "nlevels") and df_bt.columns.nlevels > 1:
                        df_bt.columns = df_bt.columns.get_level_values(0)

        if df_bt.empty or len(df_bt) < 60:
            st.error(f"Nie znaleziono wystarczających danych dla '{bt_query}'.")
        else:
            trades, still_open, open_entry_date, open_entry_price = run_backtest(
                df_bt,
                use_trend_filter=use_trend_filter,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
            )

            if trades.empty:
                st.warning(
                    f"W historii '{display_name_bt}' ({ticker_bt}) nie wystąpił żaden kompletny cykl "
                    f"KUP→SPRZEDAJ (próg ±{SIGNAL_THRESHOLD} punktów) w analizowanym okresie."
                )
            else:
                st.subheader(f"Wyniki backtestu — {display_name_bt} ({ticker_bt})")
                st.caption(
                    f"Okres: {df_bt.index[0].strftime('%Y-%m-%d')} do {df_bt.index[-1].strftime('%Y-%m-%d')} · "
                    f"{len(trades)} kompletnych transakcji · świece {bt_timeframe}"
                )

                win_rate = trades["Trafiony"].mean() * 100
                avg_return = trades["Zwrot %"].mean()
                avg_days = trades["Dni w pozycji"].mean()
                total_return = ((1 + trades["Zwrot %"] / 100).prod() - 1) * 100

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Liczba transakcji", len(trades))
                c2.metric("Win rate", f"{win_rate:.0f}%")
                c3.metric("Śr. zwrot / transakcję", f"{avg_return:+.2f}%")
                c4.metric("Śr. dni w pozycji", f"{avg_days:.0f}")

                st.metric(
                    "Zwrot łączny (składany, wszystkie transakcje po sobie)",
                    f"{total_return:+.1f}%",
                )

                exit_counts = trades["Wyjście"].value_counts()
                exit_summary = " · ".join(f"{k}: {v}" for k, v in exit_counts.items())
                st.caption(f"Powody wyjścia z pozycji — {exit_summary}")

                if still_open:
                    st.info(
                        f"Otwarta pozycja bez sygnału wyjścia: KUP z dnia {open_entry_date.strftime('%Y-%m-%d')} "
                        f"po cenie {open_entry_price:.4f} — wciąż aktywna (nie liczona w statystykach powyżej)."
                    )

                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(
                    x=df_bt.index, y=df_bt["Close"], name="Cena", line=dict(color="lightgray", width=1)
                ))
                fig_bt.add_trace(go.Scatter(
                    x=trades["Data kupna"], y=trades["Cena kupna"], mode="markers",
                    name="KUP", marker=dict(color="green", size=9, symbol="triangle-up"),
                ))
                fig_bt.add_trace(go.Scatter(
                    x=trades["Data sprzedaży"], y=trades["Cena sprzedaży"], mode="markers",
                    name="SPRZEDAJ", marker=dict(color="red", size=9, symbol="triangle-down"),
                ))
                fig_bt.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_bt, use_container_width=True)

                st.markdown("**Historia transakcji**")
                st.dataframe(
                    trades.sort_values("Data kupna", ascending=False).reset_index(drop=True),
                    use_container_width=True, hide_index=True,
                )

                st.caption(
                    "Win rate = % transakcji zamknięte z zyskiem. Każda transakcja to pełny cykl: "
                    "kupno na sygnale KUP, sprzedaż na najbliższym kolejnym sygnale SPRZEDAJ. "
                    "Wyniki historyczne nie gwarantują przyszłych rezultatów. To narzędzie analityczne, nie porada inwestycyjna."
                )

            st.divider()
            st.markdown("### 🔧 Automatyczna optymalizacja parametrów")
            st.caption(
                "Przetestuje ~40 kombinacji Stop Loss / Take Profit / filtra trendu na historii tego instrumentu "
                "i pokaże, które dały najlepszy łączny wynik. Uwaga: to dopasowanie do przeszłości (overfitting) — "
                "najlepsza kombinacja z historii nie musi być najlepsza w przyszłości."
            )
            if st.button("Znajdź najlepsze ustawienia dla tego instrumentu", key="opt_btn"):
                with st.spinner("Testuję kombinacje..."):
                    combos = []
                    for tf_opt in [True, False]:
                        for sl_opt in [4, 6, 8, 12, 16]:
                            for tp_opt in [10, 15, 20, 30]:
                                t_opt, _, _, _ = run_backtest(
                                    df_bt, use_trend_filter=tf_opt,
                                    stop_loss_pct=sl_opt, take_profit_pct=tp_opt,
                                )
                                if len(t_opt) == 0:
                                    continue
                                total_ret_opt = ((1 + t_opt["Zwrot %"] / 100).prod() - 1) * 100
                                combos.append({
                                    "Filtr trendu": "Tak" if tf_opt else "Nie",
                                    "Stop Loss %": sl_opt,
                                    "Take Profit %": tp_opt,
                                    "Transakcje": len(t_opt),
                                    "Win rate %": round(t_opt["Trafiony"].mean() * 100, 0),
                                    "Zwrot łączny %": round(total_ret_opt, 1),
                                })

                if not combos:
                    st.warning("Żadna kombinacja nie wygenerowała transakcji w tym okresie.")
                else:
                    combos_df = pd.DataFrame(combos).sort_values("Zwrot łączny %", ascending=False)
                    st.dataframe(
                        combos_df.head(15).reset_index(drop=True),
                        use_container_width=True, hide_index=True,
                    )
                    best = combos_df.iloc[0]
                    st.success(
                        f"Najlepsza znaleziona kombinacja: Stop Loss {best['Stop Loss %']:.0f}%, "
                        f"Take Profit {best['Take Profit %']:.0f}%, filtr trendu: {best['Filtr trendu']} "
                        f"→ {best['Transakcje']:.0f} transakcji, win rate {best['Win rate %']:.0f}%, "
                        f"zwrot łączny {best['Zwrot łączny %']:+.1f}%. "
                        f"Wpisz te wartości w panelu po lewej i kliknij 'Uruchom backtest' ponownie, "
                        f"żeby zobaczyć szczegóły."
                    )
    elif not bt_query:
        st.info("Wpisz instrument w panelu po lewej i kliknij 'Uruchom backtest'.")
