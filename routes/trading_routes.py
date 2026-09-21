import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Blueprint, jsonify, request, render_template

import engines.strategy_engine as strategy_engine

trading_bp = Blueprint("trading", __name__)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

PERIOD_MAP = {
    "1W":  ("7d",  "1d"),
    "1M":  ("1mo", "1d"),
    "3M":  ("3mo", "1d"),
    "6M":  ("6mo", "1d"),
    "1Y":  ("1y",  "1d"),
    "2Y":  ("2y",  "1wk"),
    "5Y":  ("5y",  "1wk"),
    "All": ("max", "1mo"),
}

QUOTE_KEYS = [
    "longName", "shortName", "sector", "industry",
    "currentPrice", "previousClose", "open", "dayLow", "dayHigh",
    "fiftyTwoWeekLow", "fiftyTwoWeekHigh", "marketCap",
]


def clean(val):
    """Makes numpy/NaN values safe for JSON."""
    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        v = float(val)
        return None if (math.isnan(v) or math.isinf(v)) else v
    return val


def series_to_json(s: pd.Series):
    """Turns a pandas Series into {labels, values} for Chart.js."""
    labels, values = [], []
    for ts, v in s.items():
        v = clean(v)
        if v is None:
            continue
        try:
            labels.append(ts.strftime("%Y-%m-%d"))
            values.append(v)
        except Exception:
            pass
    return {"labels": labels, "values": values}



@trading_bp.route("/trading")
def trading_page():
    return render_template("trading.html")

@trading_bp.route("/api/trading/quote/<symbol>")
def get_quote(symbol):
    try:
        info = yf.Ticker(symbol.upper()).info or {}
        if not info.get("currentPrice") and not info.get("regularMarketPrice"):
            return jsonify({"error": f"No data for '{symbol}'"}), 404

        out = {k: clean(info.get(k)) for k in QUOTE_KEYS}
        if out.get("currentPrice") is None:
            out["currentPrice"] = clean(info.get("regularMarketPrice"))
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@trading_bp.route("/api/trading/history/<symbol>")
def get_history(symbol):
    period = request.args.get("period", "1Y")
    p, i = PERIOD_MAP.get(period, ("1y", "1d"))
    try:
        df = yf.Ticker(symbol.upper()).history(period=p, interval=i)
        if df is None or df.empty:
            return jsonify({"error": "No history data"}), 404
        return jsonify({"close": series_to_json(df["Close"])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _calculate_performance_metrics(cap_series, bah_series, trades, capital):
    run_max = cap_series.cummax()
    final_algo = cap_series.iloc[-1]
    final_bah = bah_series.iloc[-1]

    algo_ret = (final_algo - capital) / capital * 100
    bah_ret = (final_bah - capital) / capital * 100
    max_dd = float(((cap_series - run_max) / run_max).min() * 100)

    wins = [t for t in trades if t[2] > 0]
    losses = [t for t in trades if t[2] <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0

    return run_max, {
        "total_trades": len(trades),
        "win_rate": round(win_rate, 1),
        "max_drawdown": round(max_dd, 1),
        "algo_final": round(final_algo, 2),
        "algo_return": round(algo_ret, 1),
        "bah_final": round(final_bah, 2),
        "bah_return": round(bah_ret, 1),
        "vs_bah": round(algo_ret - bah_ret, 1),
    }


def _build_chart_markers(df, trades):
    labels_all = [ts.strftime("%Y-%m-%d") for ts in df.index]
    label_index = {l: i for i, l in enumerate(labels_all)}
    buy_markers, sell_markers = [], []

    for open_trade, sell_trade, pnl in trades:
        buy_date = open_trade[1].strftime("%Y-%m-%d")
        sell_date = sell_trade[1].strftime("%Y-%m-%d")
        bi = label_index.get(buy_date)
        si = label_index.get(sell_date)
        if bi is not None:
            buy_markers.append({"index": bi, "price": open_trade[2]})
        if si is not None:
            sell_markers.append({"index": si, "price": sell_trade[2], "pnl": pnl})

    return buy_markers, sell_markers


@trading_bp.route("/api/trading/backtest", methods=["POST"])
def run_backtest():
    try:
        body = request.get_json()
        symbol = body["symbol"].upper()
        period = body.get("period", "2Y")
        capital = float(body.get("capital", 10000))
        sma_fast = int(body.get("sma_fast", 20))
        sma_slow = int(body.get("sma_slow", 50))
        stop_ma = int(body.get("stop_ma", 150))

        p, i = PERIOD_MAP.get(period, ("2y", "1d"))
        df = yf.Ticker(symbol).history(period=p, interval=i)
        if df is None or df.empty:
            return jsonify({"error": "No history data"}), 404

        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        df = df[["Close"]].dropna()

        # Run the actual strategy
        df = strategy_engine.compute_signals(df, sma_fast, sma_slow, stop_ma)
        trades, final_profit, cap_series = strategy_engine.trade_loop(df, capital)

        start_price = df["Close"].iloc[0]
        bah_series = df["Close"] * (capital / start_price)

        run_max, stats = _calculate_performance_metrics(cap_series, bah_series, trades, capital)
        buy_markers, sell_markers = _build_chart_markers(df, trades)

        def ms(col):
            return series_to_json(df[col].dropna()) if col in df.columns else {"labels": [], "values": []}

        return jsonify({
            "equity": series_to_json(cap_series),
            "bah": series_to_json(bah_series),
            "sma_fast": ms("sma_fast"),
            "sma_slow": ms("sma_slow"),
            "close": ms("Close"),
            "buy_markers": buy_markers,
            "sell_markers": sell_markers,
            "stats": stats,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500