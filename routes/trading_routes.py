import math
import os
import json
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Blueprint, jsonify, request, render_template

import engines.strategy_engine as strategy_engine
import engines.admin_engine as admin_engine

trading_bp = Blueprint("trading", __name__)

ADMIN_SECRET_TOKEN = os.environ.get("ADMIN_SECRET_TOKEN")
if not ADMIN_SECRET_TOKEN:
    raise RuntimeError("CRITICAL ERROR: 'ADMIN_SECRET_TOKEN' environment variable is missing!")

DATA_DIR = admin_engine.DATA_DIR
TICKERS_FILE = admin_engine.TICKERS_FILE

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
    "longName", "shortName", "sector", "industry", "longBusinessSummary",
    "currentPrice", "previousClose", "open", "dayLow", "dayHigh",
    "fiftyTwoWeekLow", "fiftyTwoWeekHigh", "marketCap",
]

def clean(val):
    if val is None: return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)): return None
    if isinstance(val, np.integer): return int(val)
    if isinstance(val, np.floating):
        v = float(val)
        return None if (math.isnan(v) or math.isinf(v)) else v
    return val

def series_to_json(s: pd.Series):
    labels, values = [], []
    for ts, v in s.items():
        v = clean(v)
        if v is None: continue
        try:
            labels.append(ts.strftime("%Y-%m-%d"))
        except Exception: pass
        values.append(v)
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
            
        mos = admin_engine.calculate_margin_of_safety(info)
        out["intrinsicValue"] = mos["intrinsic_value"]
        out["marginOfSafety"] = mos["margin_of_safety"]
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
        return jsonify({"close": series_to_json(df["Close"]), "volume": series_to_json(df["Volume"])})
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

        # Run native strategy engine[cite: 6, 7]
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
            "trades": trades,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@trading_bp.route("/api/trading/verify_admin", methods=["POST"])
def verify_admin():
    data = request.get_json() or {}
    if data.get("password") == ADMIN_SECRET_TOKEN:
        return jsonify({"valid": True})
    return jsonify({"valid": False}), 401

@trading_bp.route("/api/trading/insider/<symbol>")
def get_insider(symbol):
    try:
        t = yf.Ticker(symbol.upper())
        transactions = admin_engine.df_to_records(t.get_insider_transactions())
        roster = admin_engine.df_to_records(t.get_insider_roster_holders())
        purchases = admin_engine.df_to_records(t.get_insider_purchases())
        return jsonify({
            "symbol": symbol.upper(),
            "transactions": transactions,
            "roster": roster,
            "purchases": purchases,
            "signal": admin_engine.derive_insider_signal(purchases),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@trading_bp.route("/api/trading/tickers", methods=["GET", "POST"])
def manage_tickers():
    if request.method == "GET":
        content = ""
        if os.path.exists(TICKERS_FILE):
            with open(TICKERS_FILE, "r") as f: content = f.read()
        return jsonify({"content": content, "tickers": admin_engine.get_ticker_list()})

    token = request.headers.get("X-Admin-Token")
    if token != ADMIN_SECRET_TOKEN:
        return jsonify({"error": "Unauthorized admin action"}), 401

    data = request.get_json() or {}
    content = data.get("content", "")
    try:
        with open(TICKERS_FILE, "w") as f: f.write(content)
        updated = admin_engine.get_ticker_list()
        return jsonify({"message": f"Updated {TICKERS_FILE}", "count": len(updated), "tickers": updated})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@trading_bp.route("/api/trading/update_screener_data", methods=["POST"])
def update_screener_data():
    token = request.headers.get("X-Admin-Token")
    if token != ADMIN_SECRET_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401

    if admin_engine.sync_status["running"]:
        return jsonify({"status": "busy", "message": "Sync currently running."}), 400

    thread = threading.Thread(target=admin_engine.run_sync_in_background)
    thread.daemon = True
    thread.start()
    return jsonify({"status": "started", "message": "Background sync started successfully on the Pi."})

@trading_bp.route("/api/trading/screen", methods=["POST"])
def screen_tickers():
    try:
        body = request.get_json() or {}
        min_gm = float(body.get("min_gm", 30))
        min_roce = float(body.get("min_roce", 15))
        min_fcf = float(body.get("min_fcf", 1.0))
        max_debt_equity = float(body.get("max_debt_equity", 0.5))
        max_pegy = float(body.get("max_pegy", 1.0))
        max_pe = float(body.get("max_pe", 20))
        min_div_yield = float(body.get("min_div_yield", 0.0))
        min_rating = int(body.get("min_rating", 4))
        wiggle_room = float(body.get("wiggle_room", 0))

        w_lower = 1.0 - (wiggle_room / 100.0)
        w_upper = 1.0 + (wiggle_room / 100.0)

        tickers = admin_engine.get_ticker_list()
        results = []

        for symbol in tickers:
            try:
                info_file = os.path.join(DATA_DIR, f"{symbol}_info.json")
                if not os.path.exists(info_file): continue
                with open(info_file, "r") as f: info = json.load(f)

                fin_file = os.path.join(DATA_DIR, f"{symbol}_financials.csv")
                bs_file = os.path.join(DATA_DIR, f"{symbol}_balance_sheet.csv")
                fin_df = pd.read_csv(fin_file, index_col=0) if os.path.exists(fin_file) else None
                bs_df = pd.read_csv(bs_file, index_col=0) if os.path.exists(bs_file) else None

                m = admin_engine.compute_metrics(info, fin_df, bs_df)
                m = {k: clean(v) for k, v in m.items()}

                raw_dy = m.get("dividend_yield")
                dy_pct = (raw_dy * 100.0) if (raw_dy is not None and raw_dy < 1.0) else raw_dy

                score = 0
                if m["gross_margin"] is not None and m["gross_margin"] >= (min_gm * w_lower): score += 1
                if m["roce"] is not None and m["roce"] >= (min_roce * w_lower): score += 1
                if m["fcf"] is not None and m["fcf"] >= (min_fcf * w_lower): score += 1
                if m["debt_equity"] is not None and m["debt_equity"] <= (max_debt_equity * w_upper): score += 1
                if m["pegy"] is not None and m["pegy"] <= (max_pegy * w_upper): score += 1
                if m["pe"] is not None and m["pe"] <= (max_pe * w_upper): score += 1
                if dy_pct is not None and dy_pct >= (min_div_yield * w_lower): score += 1

                if score < min_rating: continue

                results.append({
                    "symbol": symbol,
                    "name": info.get("longName", symbol),
                    "rating": f"{score}/7",
                    "gross_margin": round(m["gross_margin"], 1) if m["gross_margin"] is not None else "—",
                    "roce": round(m["roce"], 1) if m["roce"] is not None else "—",
                    "fcf": round(m["fcf"], 2) if m["fcf"] is not None else "—",
                    "debt_equity": round(m["debt_equity"], 2) if m["debt_equity"] is not None else "—",
                    "pegy": round(m["pegy"], 2) if m["pegy"] is not None else "—",
                    "pe": round(m["pe"], 1) if m["pe"] is not None else "—",
                    "dividend_yield": round(dy_pct, 2) if dy_pct is not None else "—",
                    "margin_of_safety": round(m["margin_of_safety"], 1) if m["margin_of_safety"] is not None else "—",
                    "insider": admin_engine.load_insider_summary(symbol),
                })
            except Exception: continue

        results.sort(key=lambda x: int(x["rating"].split('/')[0]), reverse=True)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@trading_bp.route("/api/trading/interesting_activity", methods=["GET"])
def get_interesting_activity():
    if os.path.exists(admin_engine.INTERESTING_ACTIVITY_FILE):
        try:
            with open(admin_engine.INTERESTING_ACTIVITY_FILE, "r") as f: data = json.load(f)
            filtered = [item for item in data if not admin_engine.is_investment_trust(item.get("name"))]
            return jsonify({"activity": filtered})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"activity": []})