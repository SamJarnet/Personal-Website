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
        transaction_fee = float(body.get("transaction_fee", 0.1)) / 100
        avg_div_yield = float(body.get("dividend_yield", 4.5)) / 100

        p, i = PERIOD_MAP.get(period, ("2y", "1d"))
        df = yf.Ticker(symbol).history(period=p, interval=i)
        if df is None or df.empty:
            return jsonify({"error": "No history data"}), 404

        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        df = df[["Close"]].dropna()

        years_held = (df.index[-1] - df.index[0]).days / 365.25
        if years_held <= 0: years_held = 1.0

        # Run strategy engine purely abstracted[cite: 1]
        df = strategy_engine.compute_signals(
            df, name=symbol, 
            sma_fast_len=sma_fast, sma_slow_len=sma_slow, stop_ma_len=stop_ma
        )
        trades, final_profit, cap_series = strategy_engine.trade_loop(
            df, starting_capital=capital, position_size=1.0, transaction_fee=transaction_fee
        )
        
        gain, divs, bah_series = strategy_engine.calculate_noalgorithm_trade(
            symbol, df, capital, average_dividend_yield=avg_div_yield, years_held=years_held
        )

        algo_volatility, algo_drawdown, algo_sharpe = strategy_engine.calculate_risk_metrics(cap_series)
        bah_volatility, bah_drawdown, bah_sharpe = strategy_engine.calculate_risk_metrics(bah_series)

        algo_return = (strategy_engine.calculate_yearly_return(cap_series.iloc[-1], capital, years_held) - 1) * 100
        bah_return = (strategy_engine.calculate_yearly_return(bah_series.iloc[-1] + divs, capital, years_held) - 1) * 100

        wins = [t for t in trades if t[2] > 0]
        win_rate = len(wins) / len(trades) * 100 if trades else 0

        stats = {
            "total_trades": len(trades),
            "win_rate": round(win_rate, 1),
            "algo_return": round(algo_return, 2),
            "algo_sharpe": round(algo_sharpe, 2),
            "algo_volatility": round(algo_volatility * 100, 2),
            "algo_drawdown": round(algo_drawdown * 100, 2),
            "bah_return": round(bah_return, 2),
            "bah_sharpe": round(bah_sharpe, 2),
            "bah_volatility": round(bah_volatility * 100, 2),
            "bah_drawdown": round(bah_drawdown * 100, 2),
            "vs_bah": round(algo_return - bah_return, 1),
        }

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