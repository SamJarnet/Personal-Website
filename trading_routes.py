"""
trading_routes.py
Flask blueprints handling API endpoints and web layout rendering.
Integrates the user-defined core algorithmic engine.
"""

import math
from flask import Blueprint, jsonify, request, render_template
import yfinance as yf
import pandas as pd
import numpy as np

# Import your native strategy definitions
import strategy_engine 

trading_bp = Blueprint("trading", __name__)

STAT_KEYS = [
    "currentPrice", "previousClose", "open", "dayLow", "dayHigh",
    "fiftyTwoWeekLow", "fiftyTwoWeekHigh", "trailingPE", "forwardPE", 
    "priceToBook", "priceToSalesTrailing12Months", "enterpriseToEbitda", 
    "marketCap", "enterpriseValue", "freeCashflow", "operatingCashflow", 
    "totalRevenue", "grossProfits", "ebitda", "netIncomeToCommon", 
    "trailingEps", "forwardEps", "revenueGrowth", "earningsGrowth", 
    "grossMargins", "operatingMargins", "profitMargins", "returnOnEquity", 
    "returnOnAssets", "totalCash", "totalDebt", "debtToEquity", 
    "currentRatio", "quickRatio", "bookValue", "dividendYield", 
    "dividendRate", "payoutRatio", "sharesOutstanding", "floatShares", 
    "heldPercentInsiders", "heldPercentInstitutions", "shortRatio",
    "targetMeanPrice", "targetHighPrice", "targetLowPrice",
    "recommendationMean", "numberOfAnalystOpinions", "symbol", 
    "longName", "shortName", "sector", "industry",
]

PERIOD_MAP = {
    "1W":  ("7d",   "1d"),
    "1M":  ("1mo",  "1d"),
    "3M":  ("3mo",  "1d"),
    "6M":  ("6mo",  "1d"),
    "1Y":  ("1y",   "1d"),
    "2Y":  ("2y",   "1wk"),
    "5Y":  ("5y",   "1wk"),
    "All": ("max",  "1mo"),
}

# ── data sanitization helpers ─────────────────────────────────────────────────

def clean(val):
    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        v = float(val)
        return None if (math.isnan(v) or math.isinf(v)) else v
    return val


def series_to_json(s: pd.Series):
    """Maps Pandas series data directly into Chart.js scannable configurations."""
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


# ── page routes ───────────────────────────────────────────────────────────────

@trading_bp.route("/trading")
def trading_page():
    return render_template("trading.html")


# ── stock data endpoints ──────────────────────────────────────────────────────

@trading_bp.route("/api/trading/quote/<symbol>")
def get_quote(symbol):
    try:
        t = yf.Ticker(symbol.upper())
        info = t.info or {}
        if not info.get("currentPrice") and not info.get("regularMarketPrice"):
            return jsonify({"error": f"No data for '{symbol}'"}), 404
        out = {k: clean(info.get(k)) for k in STAT_KEYS}
        if out.get("currentPrice") is None:
            out["currentPrice"] = clean(info.get("regularMarketPrice"))
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@trading_bp.route("/api/trading/history/<symbol>")
def get_history(symbol):
    period = request.args.get("period", "1Y")
    p, i   = PERIOD_MAP.get(period, ("1y", "1d"))
    try:
        t  = yf.Ticker(symbol.upper())
        df = t.history(period=p, interval=i)
        if df is None or df.empty:
            return jsonify({"error": "No history data"}), 404
        return jsonify({
            "close":  series_to_json(df["Close"]),
            "volume": series_to_json(df["Volume"]),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── backtest engine integration ───────────────────────────────────────────────

def _calculate_performance_metrics(cap_series, bah_series, trades, capital):
    """Modular helper isolated to parse quantitative portfolio stats."""
    run_max     = cap_series.cummax()
    final_algo = cap_series.iloc[-1]
    final_bah  = bah_series.iloc[-1]
    n_days     = (cap_series.index[-1] - cap_series.index[0]).days or 1
    years      = n_days / 365.25

    algo_ret  = (final_algo - capital) / capital * 100
    bah_ret   = (final_bah  - capital) / capital * 100
    algo_cagr = ((final_algo / capital) ** (1 / years) - 1) * 100 if years > 0 else 0
    bah_cagr  = ((final_bah  / capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    dr     = cap_series.pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * (252 ** 0.5)) if dr.std() > 0 else 0
    max_dd = float(((cap_series - run_max) / run_max).min() * 100)

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win  = sum(t["pnl"] for t in wins)   / len(wins)   if wins   else 0
    avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
    pf_denom = abs(sum(t["pnl"] for t in losses))
    pf       = abs(sum(t["pnl"] for t in wins)) / pf_denom if pf_denom else None

    return run_max, {
        "total_trades":  len(trades),
        "win_rate":      round(win_rate, 1),
        "avg_win":       round(avg_win, 2),
        "avg_loss":      round(avg_loss, 2),
        "profit_factor": round(pf, 2) if pf else None,
        "sharpe":        round(sharpe, 2),
        "max_drawdown":  round(max_dd, 1),
        "algo_final":    round(final_algo, 2),
        "algo_return":   round(algo_ret, 1),
        "algo_cagr":     round(algo_cagr, 1),
        "bah_final":     round(final_bah, 2),
        "bah_return":    round(bah_ret, 1),
        "bah_cagr":      round(bah_cagr, 1),
        "vs_bah":        round(algo_ret - bah_ret, 1),
    }


def _build_chart_markers(df, trades):
    """Modular helper to parse scatter-plot transaction point keys for UI."""
    buy_indices  = []
    sell_indices = []
    labels_all   = [ts.strftime("%Y-%m-%d") for ts in df.index]
    label_index  = {l: i for i, l in enumerate(labels_all)}
    
    for tr in trades:
        bi = label_index.get(tr["buy_date"])
        si = label_index.get(tr["sell_date"])
        if bi is not None:
            buy_indices.append({"index": bi, "price": tr["buy_price"]})
        if si is not None:
            sell_indices.append({"index": si, "price": tr["sell_price"], "pnl": tr["pnl"]})
            
    return buy_indices, sell_indices


@trading_bp.route("/api/trading/backtest", methods=["POST"])
def run_backtest():
    try:
        body     = request.get_json()
        symbol   = body["symbol"].upper()
        period   = body.get("period", "2Y")
        capital  = float(body.get("capital", 10000))
        sma_fast = int(body.get("sma_fast", 20))
        sma_slow = int(body.get("sma_slow", 50))
        stop_ma  = int(body.get("stop_ma",  150))

        # 1. Fetch market payload via context constraints
        p, i = PERIOD_MAP.get(period, ("2y", "1d"))
        t    = yf.Ticker(symbol)
        df   = t.history(period=p, interval=i)
        if df is None or df.empty:
            return jsonify({"error": "No history data"}), 404

        df = df[["Close"]].dropna()
        
        # 2. RUN USER'S ORIGINAL ALGORITHM 
        df = strategy_engine.compute_signals(df, sma_fast, sma_slow, stop_ma)
        trades, final_profit, cap_series = strategy_engine.trade_loop(df, capital)

        # 3. Process Buy-and-Hold Baseline Analytics
        start_price = df["Close"].iloc[0]
        bah_series  = df["Close"] * (capital / start_price)
        
        # 4. Generate modular statistics and layout markers
        run_max, stats = _calculate_performance_metrics(cap_series, bah_series, trades, capital)
        buy_markers, sell_markers = _build_chart_markers(df, trades)

        def ms(col):
            return series_to_json(df[col].dropna()) if col in df.columns else {"labels": [], "values": []}

        # 5. Package output for delivery to frontend
        return jsonify({
            "trades":       trades,
            "equity":       series_to_json(cap_series),
            "bah":          series_to_json(bah_series),
            "drawdown":     series_to_json((cap_series - run_max) / run_max * 100),
            "sma_fast":     ms("sma_fast"),
            "sma_slow":     ms("sma_slow"),
            "stop_ma":      ms("stop_ma"),
            "close":        ms("Close"),
            "buy_markers":  buy_markers,
            "sell_markers": sell_markers,
            "stats":        stats,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500