"""
trading_routes.py
Flask blueprints handling API endpoints and web layout rendering.
Integrates the user-defined core algorithmic engine alongside a persistent local screener cache.
"""

import math
import os
import json
from datetime import datetime
from flask import Blueprint, jsonify, request, render_template
import yfinance as yf
import pandas as pd
import numpy as np
from dotenv import load_dotenv
# Import your native strategy definitions
import engines.strategy_engine as strategy_engine 

load_dotenv()

trading_bp = Blueprint("trading", __name__)

ADMIN_SECRET_TOKEN = os.environ.get("ADMIN_SECRET_TOKEN")
if not ADMIN_SECRET_TOKEN:
    raise RuntimeError("CRITICAL ERROR: ADMIN_SECRET_TOKEN environment variable is not set!")

DATA_DIR = "data"
TICKERS_FILE = "tickers.txt"
os.makedirs(DATA_DIR, exist_ok=True)

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
            labels.append(ts.strftime("%Y-%m-%d")) # type: ignore
            values.append(v)
        except Exception:
            pass
    return {"labels": labels, "values": values}


# ── screener caching and math helpers ─────────────────────────────────────────

def get_ticker_list():
    """Reads tickers from tickers.txt or populates a default list."""
    if os.path.exists(TICKERS_FILE):
        with open(TICKERS_FILE, "r") as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
            if tickers:
                return tickers
    default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX', 'AMD', 'INTC']
    with open(TICKERS_FILE, "w") as f:
        for t in default_tickers:
            f.write(f"{t}\n")
    return default_tickers


def update_ticker_data(symbol):
    """Updates stock data folder cache incrementally, adding only missing days."""
    symbol = symbol.upper()
    history_file = os.path.join(DATA_DIR, f"{symbol}_history.csv")
    info_file = os.path.join(DATA_DIR, f"{symbol}_info.json")
    financials_file = os.path.join(DATA_DIR, f"{symbol}_financials.csv")
    balance_file = os.path.join(DATA_DIR, f"{symbol}_balance_sheet.csv")
    
    ticker = yf.Ticker(symbol)
    today_str = datetime.today().strftime("%Y-%m-%d")
    
    # Avoid scraping fundamentals multiple times in a single day
    fundamentals_updated_today = False
    if os.path.exists(info_file):
        mtime = os.path.getmtime(info_file)
        if datetime.fromtimestamp(mtime).strftime("%Y-%m-%d") == today_str:
            fundamentals_updated_today = True

    if not fundamentals_updated_today:
        try:
            info = ticker.info
            if info:
                with open(info_file, 'w') as f:
                    json.dump(info, f)
            
            fin = ticker.financials
            if fin is not None and not fin.empty:
                fin.to_csv(financials_file)
                
            bs = ticker.balance_sheet
            if bs is not None and not bs.empty:
                bs.to_csv(balance_file)
        except Exception as e:
            print(f"Error updating fundamentals for {symbol}: {e}")

    # Incremental Price Data Update (Fetch only missing days)
    try:
        if os.path.exists(history_file):
            existing_df = pd.read_csv(history_file, index_col=0)
            if not existing_df.empty:
                # FIX: Force uniform timezone-naive mapping on cached data
                existing_df.index = pd.to_datetime(existing_df.index, utc=True).tz_localize(None)

                last_date = existing_df.index.max()
                new_df = ticker.history(start=last_date.strftime("%Y-%m-%d"))
                
                if not new_df.empty:
                    # FIX: Force uniform timezone-naive mapping on incoming data
                    new_df.index = pd.to_datetime(new_df.index, utc=True).tz_localize(None)

                    combined_df = pd.concat([existing_df, new_df])
                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                    combined_df.to_csv(history_file)
            else:
                df_fresh = ticker.history(period="5y")
                df_fresh.index = pd.to_datetime(df_fresh.index, utc=True).tz_localize(None)
                df_fresh.to_csv(history_file)
        else:
            df_fresh = ticker.history(period="5y")
            df_fresh.index = pd.to_datetime(df_fresh.index, utc=True).tz_localize(None)
            df_fresh.to_csv(history_file)
    except Exception as e:
        print(f"Error updating history for {symbol}: {e}")


def compute_metrics(info, financials=None, balance_sheet=None):
    """Parses custom analytical parameters from the cached file datasets."""
    metrics = {}
    
    # 1. Gross Margin
    gm = info.get("grossMargins")
    metrics["gross_margin"] = gm * 100 if gm is not None else None
    
    # 2. P/E Ratio
    metrics["pe"] = info.get("trailingPE") or info.get("forwardPE")
    
    # 3. Debt to Equity
    de = info.get("debtToEquity")
    metrics["debt_equity"] = de / 100.0 if de and de > 5 else de
    
    # 4. Free Cash Flow (In Billions)
    fcf = info.get("freeCashflow")
    metrics["fcf"] = fcf / 1e9 if fcf is not None else None
    
    # 5. PEGY Ratio
    pe = metrics["pe"]
    eg = info.get("earningsGrowth") 
    dy = info.get("dividendYield")   
    if pe and eg:
        growth_total = (eg * 100) + ((dy * 100) if dy else 0)
        metrics["pegy"] = pe / growth_total if growth_total > 0 else None
    else:
        metrics["pegy"] = None

    # 6. ROCE Calculation with fallback to ROE
    roce = None
    if financials is not None and balance_sheet is not None:
        try:
            ebit = None
            for k in ['Operating Income', 'EBIT']:
                if k in financials.index:
                    ebit = financials.loc[k].iloc[0]
                    break
            total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else None
            current_liab = 0
            for k in ['Current Liabilities', 'Total Current Liabilities']:
                if k in balance_sheet.index:
                    current_liab = balance_sheet.loc[k].iloc[0]
                    break
            if ebit is not None and total_assets is not None:
                cap_employed = total_assets - current_liab
                if cap_employed > 0:
                    roce = (ebit / cap_employed) * 100
        except Exception:
            pass
    if roce is None:
        roe = info.get("returnOnEquity")
        roce = roe * 100 if roe is not None else None
        
    metrics["roce"] = roce

    # 7. Dividend Yield
    metrics["dividend_yield"] = dy * 100 if dy is not None else None

    return metrics


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


# ── screener endpoints ────────────────────────────────────────────────────────

@trading_bp.route("/api/trading/update_screener_data", methods=["POST"])
def update_screener_data():
    user_token = request.headers.get("X-Admin-Token")
    if not user_token or user_token != ADMIN_SECRET_TOKEN:
        return jsonify({"error": "Unauthorized: Admin access required"}), 401
        
    try:
        tickers = get_ticker_list()
        updated = 0
        for symbol in tickers:
            try:
                update_ticker_data(symbol)
                updated += 1
            except Exception:
                continue
        return jsonify({"status": "success", "message": "Screener data synchronized."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
        
        # New Scoring & Rating inputs
        min_rating = int(body.get("min_rating", 4))
        wiggle_room = float(body.get("wiggle_room", 0))  # Provided as percentage (e.g., 5 means 5%)
        
        # Compute modifiers based on wiggle room percentage
        w_lower = 1.0 - (wiggle_room / 100.0)
        w_upper = 1.0 + (wiggle_room / 100.0)

        tickers = get_ticker_list()
        results = []
        
        for symbol in tickers:
            info_file = os.path.join(DATA_DIR, f"{symbol}_info.json")
            if not os.path.exists(info_file):
                continue
            with open(info_file, "r") as f:
                info = json.load(f)
                
            financials_file = os.path.join(DATA_DIR, f"{symbol}_financials.csv")
            balance_file = os.path.join(DATA_DIR, f"{symbol}_balance_sheet.csv")
            fin_df = pd.read_csv(financials_file, index_col=0) if os.path.exists(financials_file) else None
            bs_df = pd.read_csv(balance_file, index_col=0) if os.path.exists(balance_file) else None
            
            m = compute_metrics(info, fin_df, bs_df)
            
            # Evaluate the 6 distinct analytical criteria using wiggle room factors
            score = 0
            if m["gross_margin"] is not None and m["gross_margin"] >= (min_gm * w_lower): score += 1
            if m["roce"] is not None and m["roce"] >= (min_roce * w_lower): score += 1
            if m["fcf"] is not None and m["fcf"] >= (min_fcf * w_lower): score += 1
            if m["debt_equity"] is not None and m["debt_equity"] <= (max_debt_equity * w_upper): score += 1
            if m["pegy"] is not None and m["pegy"] <= (max_pegy * w_upper): score += 1
            if m["pe"] is not None and m["pe"] <= (max_pe * w_upper): score += 1
            
            # Filter matches based on user's minimum rating selection
            if score < min_rating: 
                continue
            
            results.append({
                "symbol": symbol,
                "name": info.get("longName", symbol),
                "rating": f"{score}/6",
                "gross_margin": round(m["gross_margin"], 1) if m["gross_margin"] is not None else "—",
                "roce": round(m["roce"], 1) if m["roce"] is not None else "—",
                "fcf": round(m["fcf"], 2) if m["fcf"] is not None else "—",
                "debt_equity": round(m["debt_equity"], 2) if m["debt_equity"] is not None else "—",
                "pegy": round(m["pegy"], 2) if m["pegy"] is not None else "—",
                "pe": round(m["pe"], 1) if m["pe"] is not None else "—",
                "dividend_yield": round(m["dividend_yield"], 2) if m["dividend_yield"] is not None else "—"
            })
            
        # Sort results by higher scoring setups first
        results.sort(key=lambda x: int(x["rating"].split('/')[0]), reverse=True)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@trading_bp.route("/api/trading/verify_admin", methods=["POST"])
def verify_admin():
    data = request.get_json() or {}
    user_password = data.get("password")
    
    if user_password and user_password == ADMIN_SECRET_TOKEN:
        return jsonify({"valid": True})
        
    return jsonify({"valid": False}), 401

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

    # FIX: trades is a tuple structure where PnL is at index 2
    wins   = [t for t in trades if t[2] > 0]
    losses = [t for t in trades if t[2] <= 0]
    
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win  = sum(t[2] for t in wins)   / len(wins)   if wins   else 0
    avg_loss = sum(t[2] for t in losses) / len(losses) if losses else 0
    pf_denom = abs(sum(t[2] for t in losses))
    pf       = abs(sum(t[2] for t in wins)) / pf_denom if pf_denom else None

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
        # FIX: Extract dates and data using tuple integers
        # tr[0] is open_trade, tr[1] is sell_trade, tr[2] is pnl
        buy_date_str  = tr[0][1].strftime("%Y-%m-%d") if hasattr(tr[0][1], "strftime") else str(tr[0][1])
        sell_date_str = tr[1][1].strftime("%Y-%m-%d") if hasattr(tr[1][1], "strftime") else str(tr[1][1])
        
        bi = label_index.get(buy_date_str)
        si = label_index.get(sell_date_str)
        
        if bi is not None:
            buy_indices.append({"index": bi, "price": tr[0][2]}) # tr[0][2] is buy price
        if si is not None:
            sell_indices.append({"index": si, "price": tr[1][2], "pnl": tr[2]}) # tr[1][2] is sell price, tr[2] is pnl
            
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

        # Check local data folder for historical prices to drastically speed up execution
        history_file = os.path.join(DATA_DIR, f"{symbol}_history.csv")
        if os.path.exists(history_file):
            df = pd.read_csv(history_file, index_col=0)
            if not df.empty:
                # FIX: Force strict UTC timeline, then drop it to make it tz-naive
                df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
                
                today = datetime.now()
                # Local slicing to match standard timeframe behavior
                days_map = {"1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365, "2Y": 730, "5Y": 1825}
                if period in days_map:
                    start_date = today - pd.Timedelta(days=days_map[period])
                    df = df[df.index >= pd.to_datetime(start_date.date())]
        else:
            p, i = PERIOD_MAP.get(period, ("2y", "1d"))
            df = yf.Ticker(symbol).history(period=p, interval=i)
            # FIX: Sanitize live yfinance data immediately
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
            
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