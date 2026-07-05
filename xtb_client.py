"""
================================================================
KLIENT xStation5 API (XTB) - dla aplikacji webowej Streamlit
================================================================
Wersja uproszczona pod użycie wewnątrz app.py. Dane logowania
przekazywane bezpośrednio (nie ze zmiennych środowiskowych),
bo aplikacja jest publiczna - każdy użytkownik wpisuje własne.

NIC z danych logowania nie jest tu zapisywane na dysk ani
wysyłane gdziekolwiek poza sam XTB.
================================================================
"""

import json
import time
import threading
import websocket


class XTBClient:
    PERIOD_MAP = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "4h": 240, "1d": 1440, "1w": 10080, "1M": 43200,
    }

    def __init__(self, user_id, password, account_type="demo"):
        if not user_id or not password:
            raise ValueError("Brak numeru konta lub hasła XTB.")
        self.user_id = str(user_id)
        self.password = password
        self.account_type = account_type
        host = "wss://ws.xtb.com/demo" if account_type == "demo" else "wss://ws.xtb.com/real"
        self.ws_url = host
        self.ws = None
        self._lock = threading.Lock()
        self._last_command_time = 0

    def connect(self):
        self.ws = websocket.create_connection(self.ws_url, timeout=15)
        self._login()

    def _login(self):
        response = self._send_command("login", {
            "userId": self.user_id,
            "password": self.password,
            "appId": "signal_analyzer_web",
            "appName": "SignalAnalyzerWeb",
        })
        if not response.get("status"):
            error_code = response.get("errorCode", "unknown")
            error_desc = response.get("errorDescr", "")
            raise ConnectionError(f"Logowanie do XTB nie powiodło się [{error_code}]: {error_desc}")

    def _send_command(self, command, arguments=None, expect_response=True):
        with self._lock:
            payload = {"command": command}
            if arguments:
                payload["arguments"] = arguments
            self.ws.send(json.dumps(payload))
            self._last_command_time = time.time()
            if not expect_response:
                return {}
            raw = self.ws.recv()
            return json.loads(raw)

    def get_candles(self, symbol, timeframe, lookback_minutes):
        period = self.PERIOD_MAP.get(timeframe)
        if period is None:
            raise ValueError(f"Nieznany timeframe: {timeframe}")
        start_ms = int((time.time() - lookback_minutes * 60) * 1000)
        response = self._send_command("getChartLastRequest", {
            "info": {"period": period, "start": start_ms, "symbol": symbol}
        })
        if not response.get("status"):
            raise RuntimeError(f"Błąd pobierania świec dla {symbol}: {response.get('errorDescr', 'nieznany błąd')}")
        data = response["returnData"]
        digits = data["digits"]
        scale = 10 ** digits
        candles = []
        for rate in data["rateInfos"]:
            open_price = rate["open"] / scale
            candles.append({
                "time": rate["ctm"] / 1000,
                "open": open_price,
                "close": open_price + rate["close"] / scale,
                "high": open_price + rate["high"] / scale,
                "low": open_price + rate["low"] / scale,
                "volume": rate.get("vol", 0),
            })
        return candles

    def disconnect(self):
        try:
            self._send_command("logout", expect_response=False)
        except Exception:
            pass
        if self.ws:
            self.ws.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()‹
