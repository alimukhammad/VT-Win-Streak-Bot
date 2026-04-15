#!/usr/bin/env python3
from __future__ import annotations

import random
import threading
import time
import logging
from datetime import datetime, timezone
from typing import Any

from flask import Flask, jsonify, render_template

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional runtime dependency fallback
    yf = None

logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("curl_cffi").setLevel(logging.CRITICAL)


app = Flask(__name__)

# =========================
# Strategy + Runtime Config
# =========================
TICKER = "VT"
PAPER_TRADING = True
LOOP_INTERVAL_SECONDS = 10

INITIAL_BALANCE = 10_000.0
BASE_SHARE_SIZE = 1.0
WIN_STREAK_MULTIPLIER = 1.5


# =========================
# Shared state
# =========================
state_lock = threading.Lock()
stop_event = threading.Event()
trading_thread: threading.Thread | None = None

state: dict[str, Any] = {
    "running": False,
    "paper_trading": PAPER_TRADING,
    "ticker": TICKER,
    "cash_balance": INITIAL_BALANCE,
    "account_balance": INITIAL_BALANCE,
    "position_shares": 0.0,
    "entry_price": None,
    "current_share_size": BASE_SHARE_SIZE,
    "base_share_size": BASE_SHARE_SIZE,
    "win_streak": 0,
    "wins": 0,
    "losses": 0,
    "last_price": None,
    "ma20": None,
    "last_signal": "WAITING",
    "last_trade_pnl": 0.0,
    "last_error": "",
    "data_source": "uninitialized",
    "loop_count": 0,
    "updated_at": "",
}

live_fetch_cooldown_until = 0.0


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fetch_market_data() -> tuple[float, float, str]:
    if yf is None:
        raise RuntimeError("yfinance is not installed")

    history = yf.Ticker(TICKER).history(period="3mo", interval="1d", auto_adjust=True)
    closes = history.get("Close")
    if closes is None or closes.dropna().shape[0] < 20:
        raise RuntimeError("not enough market data for MA20")

    closes = closes.dropna()
    last_price = float(closes.iloc[-1])
    ma20 = float(closes.tail(20).mean())
    return last_price, ma20, "yfinance"


def simulate_market_data() -> tuple[float, float, str]:
    with state_lock:
        previous_price = float(state["last_price"] or 100.0)
        previous_ma = float(state["ma20"] or previous_price)

    drift = random.uniform(-0.8, 0.8)
    next_price = max(1.0, previous_price + drift)
    next_ma20 = (previous_ma * 0.9) + (next_price * 0.1)
    return next_price, next_ma20, "simulated"


def get_market_snapshot() -> tuple[float, float, str, str]:
    global live_fetch_cooldown_until

    now = time.time()
    if now < live_fetch_cooldown_until:
        price, ma20, source = simulate_market_data()
        return price, ma20, source, "Live feed cooldown active; using simulated feed."

    try:
        price, ma20, source = fetch_market_data()
        return price, ma20, source, ""
    except Exception as exc:
        live_fetch_cooldown_until = time.time() + 120
        price, ma20, source = simulate_market_data()
        error = f"Live market data unavailable ({exc}); using simulated feed."
        return price, ma20, source, error


def trading_step() -> None:
    price, ma20, source, warning = get_market_snapshot()

    with state_lock:
        state["loop_count"] += 1
        state["last_price"] = price
        state["ma20"] = ma20
        state["data_source"] = source
        state["updated_at"] = iso_now()
        state["last_error"] = warning

        has_position = state["position_shares"] > 0.0

        if not has_position and price > ma20:
            shares = float(state["current_share_size"])
            cost = shares * price
            if cost <= float(state["cash_balance"]):
                state["cash_balance"] = float(state["cash_balance"]) - cost
                state["position_shares"] = shares
                state["entry_price"] = price
                state["last_signal"] = "BUY"
            else:
                state["last_signal"] = "NO_CASH"

        elif has_position and price < ma20:
            shares = float(state["position_shares"])
            entry_price = float(state["entry_price"])
            proceeds = shares * price
            pnl = (price - entry_price) * shares

            state["cash_balance"] = float(state["cash_balance"]) + proceeds
            state["position_shares"] = 0.0
            state["entry_price"] = None
            state["last_trade_pnl"] = pnl
            state["last_signal"] = "SELL"

            if pnl > 0:
                state["wins"] = int(state["wins"]) + 1
                state["win_streak"] = int(state["win_streak"]) + 1
                state["current_share_size"] = float(state["current_share_size"]) * WIN_STREAK_MULTIPLIER
            else:
                state["losses"] = int(state["losses"]) + 1
                state["win_streak"] = 0
                state["current_share_size"] = BASE_SHARE_SIZE
        else:
            state["last_signal"] = "HOLD"

        mark_to_market = float(state["cash_balance"]) + (float(state["position_shares"]) * price)
        state["account_balance"] = mark_to_market


def trading_loop() -> None:
    while not stop_event.is_set():
        try:
            trading_step()
        except Exception as exc:  # pragma: no cover - runtime safety guard
            with state_lock:
                state["last_error"] = f"Trading step error: {exc}"
                state["updated_at"] = iso_now()
        stop_event.wait(LOOP_INTERVAL_SECONDS)

    with state_lock:
        state["running"] = False


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/api/status")
def api_status():
    with state_lock:
        return jsonify(dict(state))


@app.post("/api/start")
def api_start():
    global trading_thread

    with state_lock:
        if state["running"]:
            return jsonify({"ok": True, "message": "already running", "state": dict(state)})
        state["running"] = True
        state["last_error"] = ""
        state["updated_at"] = iso_now()

    stop_event.clear()
    trading_thread = threading.Thread(target=trading_loop, daemon=True, name="trading-loop")
    trading_thread.start()

    with state_lock:
        return jsonify({"ok": True, "message": "started", "state": dict(state)})


@app.post("/api/stop")
def api_stop():
    stop_event.set()

    global trading_thread
    if trading_thread and trading_thread.is_alive():
        trading_thread.join(timeout=1.0)

    with state_lock:
        state["running"] = False
        state["updated_at"] = iso_now()
        return jsonify({"ok": True, "message": "stopped", "state": dict(state)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
