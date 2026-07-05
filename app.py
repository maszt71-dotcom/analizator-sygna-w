"""
================================================================
ANALIZATOR SYGNAŁÓW GIEŁDOWYCH / KRYPTO
================================================================
Łączy wskaźniki techniczne (RSI, MACD, SMA, Bollinger Bands)
z formacjami świecowymi, aby generować sygnały BUY / SELL / HOLD.

Dane: Yahoo Finance (yfinance) - darmowe, bez limitów, obejmuje
akcje US oraz krypto (format BTC-USD, ETH-USD itd.)

UWAGA: To narzędzie ANALITYCZNE, nie łączy się z kontem XTB.
Nie składa żadnych zleceń automatycznie - tylko generuje sygnały.
Nie jest to porada inwestycyjna.
================================================================
"""

import time
import datetime
import requests
import numpy as np
import pandas as pd
import yfinance as yf

# ============================================================
# KONFIGURACJA - edytuj tutaj
# ============================================================

TICKERS = [
    "AAPL",      # Apple
    "MSFT",      # Microsoft
    "NVDA",      # Nvidia
    "BTC-USD",   # Bitcoin
    "ETH-USD",   # Ethereum
]

INTERVAL_MINUTES = 5          # co ile minut sprawdzać rynek
PERIOD = "60d"                 # ile historii ściągać (max dla świec 15m w yfinance)
TIMEFRAME = "15m"              # świece 15-minutowe (bardziej "na bieżąco")

# Telegram alerty (opcjonalne) - wpisz swoje dane, albo zostaw puste żeby wyłączyć
TELEGRAM_BOT_TOKEN = ""       # np. "123456789:AAExxxxxx"
TELEGRAM_CHAT_ID = ""         # np. "987654321"

# Próg punktowy do wygenerowania sygnału (im wyżej, tym rzadziej i "mocniej" sygnał)
SIGNAL_THRESHOLD = 3

# ============================================================
# WSKAŹNIKI TECHNICZNE
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
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(series: pd.Series, period=20, num_std=2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, sma, lower


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = rsi(df["Close"])
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(df["Close"])
    df["BB_upper"], df["BB_mid"], df["BB_lower"] = bollinger_bands(df["Close"])
    return df


# ============================================================
# FORMACJE ŚWIECOWE (proste reguły, bez zewnętrznych bibliotek TA-Lib)
# ============================================================

def detect_candlestick_patterns(df: pd.DataFrame) -> list:
    """Zwraca listę wykrytych formacji na podstawie 2 ostatnich świec."""
    patterns = []
    if len(df) < 2:
        return patterns

    last = df.iloc[-1]
    prev = df.iloc[-2]

    o, h, l, c = last["Open"], last["High"], last["Low"], last["Close"]
    po, ph, pl, pc = prev["Open"], prev["High"], prev["Low"], prev["Close"]

    body = abs(c - o)
    candle_range = h - l if h != l else 1e-9
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    # Bullish engulfing
    if pc < po and c > o and c > po and o < pc:
        patterns.append(("bullish_engulfing", "bull"))

    # Bearish engulfing
    if pc > po and c < o and c < po and o > pc:
        patterns.append(("bearish_engulfing", "bear"))

    # Hammer (mały korpus na górze, długi dolny cień, w trendzie spadkowym)
    if lower_wick > body * 2 and upper_wick < body * 0.5 and c > o:
        patterns.append(("hammer", "bull"))

    # Shooting star (mały korpus na dole, długi górny cień)
    if upper_wick > body * 2 and lower_wick < body * 0.5 and c < o:
        patterns.append(("shooting_star", "bear"))

    # Doji (bardzo mały korpus względem zakresu świecy)
    if body < candle_range * 0.1:
        patterns.append(("doji", "neutral"))

    return patterns


# ============================================================
# SILNIK SYGNAŁÓW - łączy wskaźniki + formacje w system punktowy
# ============================================================

def generate_signal(df: pd.DataFrame) -> dict:
    df = add_indicators(df)
    last = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0
    reasons = []

    # --- RSI ---
    if pd.notna(last["RSI"]):
        if last["RSI"] < 30:
            score += 2
            reasons.append(f"RSI={last['RSI']:.1f} (wyprzedanie)")
        elif last["RSI"] > 70:
            score -= 2
            reasons.append(f"RSI={last['RSI']:.1f} (wykupienie)")

    # --- MACD crossover ---
    if pd.notna(last["MACD"]) and pd.notna(prev["MACD"]):
        if prev["MACD"] < prev["MACD_signal"] and last["MACD"] > last["MACD_signal"]:
            score += 2
            reasons.append("MACD: przecięcie w górę (bull)")
        elif prev["MACD"] > prev["MACD_signal"] and last["MACD"] < last["MACD_signal"]:
            score -= 2
            reasons.append("MACD: przecięcie w dół (bear)")

    # --- Trend SMA20/50 ---
    if pd.notna(last["SMA20"]) and pd.notna(last["SMA50"]):
        if last["SMA20"] > last["SMA50"]:
            score += 1
            reasons.append("Trend: SMA20 > SMA50 (wzrostowy)")
        else:
            score -= 1
            reasons.append("Trend: SMA20 < SMA50 (spadkowy)")

    # --- Bollinger Bands ---
    if pd.notna(last["BB_lower"]) and last["Close"] < last["BB_lower"]:
        score += 1
        reasons.append("Cena poniżej dolnej Bollingera (potencjalne odbicie)")
    elif pd.notna(last["BB_upper"]) and last["Close"] > last["BB_upper"]:
        score -= 1
        reasons.append("Cena powyżej górnej Bollingera (potencjalna korekta)")

    # --- Formacje świecowe ---
    patterns = detect_candlestick_patterns(df)
    for name, direction in patterns:
        if direction == "bull":
            score += 1
            reasons.append(f"Formacja: {name} (bull)")
        elif direction == "bear":
            score -= 1
            reasons.append(f"Formacja: {name} (bear)")

    # --- Klasyfikacja sygnału ---
    if score >= SIGNAL_THRESHOLD:
        signal = "KUP"
    elif score <= -SIGNAL_THRESHOLD:
        signal = "SPRZEDAJ"
    else:
        signal = "CZEKAJ"

    return {
        "signal": signal,
        "score": score,
        "price": round(last["Close"], 2),
        "reasons": reasons,
        "time": df.index[-1],
    }


# ============================================================
# TELEGRAM ALERT
# ============================================================

def send_telegram_alert(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
    except Exception as e:
        print(f"[Telegram] Błąd wysyłki: {e}")


# ============================================================
# GŁÓWNA PĘTLA
# ============================================================

def analyze_ticker(ticker: str) -> dict | None:
    try:
        df = yf.download(ticker, period=PERIOD, interval=TIMEFRAME, progress=False, auto_adjust=True)
        if df.empty or len(df) < 55:
            print(f"[{ticker}] Za mało danych do analizy.")
            return None
        result = generate_signal(df)
        result["ticker"] = ticker
        return result
    except Exception as e:
        print(f"[{ticker}] Błąd pobierania/analizy: {e}")
        return None


def run_once():
    print(f"\n=== Analiza: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    for ticker in TICKERS:
        result = analyze_ticker(ticker)
        if result is None:
            continue

        line = f"{ticker:10s} | Cena: {result['price']:>10} | Sygnał: {result['signal']:5s} | Punkty: {result['score']:+d}"
        print(line)
        for r in result["reasons"]:
            print(f"    - {r}")

        if result["signal"] in ("KUP", "SPRZEDAJ"):
            msg = (
                f"🔔 SYGNAŁ: {result['signal']} - {ticker}\n"
                f"Cena: {result['price']}\n"
                f"Punkty: {result['score']:+d}\n"
                f"Powody:\n" + "\n".join(f"- {r}" for r in result["reasons"])
            )
            send_telegram_alert(msg)


def run_loop():
    print("Start monitoringu. Ctrl+C aby zatrzymać.")
    while True:
        try:
            run_once()
        except Exception as e:
            print(f"Błąd w pętli głównej: {e}")
        time.sleep(INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    run_loop()
