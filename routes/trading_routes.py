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
import time
import threading
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
    "longBusinessSummary", "currentPrice", "previousClose", "open", "dayLow", "dayHigh",
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


def df_to_records(df):
    """Converts a yfinance DataFrame (insider data etc.) into clean JSON records."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    out = []
    for _, row in df.reset_index().iterrows():
        rec = {}
        for k, v in row.items():
            if hasattr(v, "strftime"):
                try:
                    v = v.strftime("%Y-%m-%d")
                except Exception:
                    v = str(v)
            elif isinstance(v, (np.integer, np.floating)):
                v = clean(v)
            else:
                try:
                    if pd.isna(v):
                        v = None
                except (TypeError, ValueError):
                    pass
            rec[str(k)] = v
        out.append(rec)
    return out


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


# ── margin of safety (Graham revised formula) ─────────────────────────────────

_bond_yield_cache = {"date": None, "value": 4.5}

def get_current_bond_yield():
    """Fetches the 10Y US Treasury yield (^TNX) as a risk-free rate proxy for the
    Graham formula, cached once per day. Falls back to a conservative 4.5% if the
    lookup fails (e.g. no network, or the ticker's home market has no equivalent)."""
    today_str = datetime.today().strftime("%Y-%m-%d")
    if _bond_yield_cache["date"] == today_str:
        return _bond_yield_cache["value"]
    try:
        tnx = yf.Ticker("^TNX").history(period="5d")
        if tnx is not None and not tnx.empty:
            y = float(tnx["Close"].dropna().iloc[-1])
            if y > 0:
                _bond_yield_cache["value"] = y
    except Exception as e:
        print(f"Error fetching bond yield, using fallback: {e}")
    _bond_yield_cache["date"] = today_str
    return _bond_yield_cache["value"]


def calculate_margin_of_safety(info, current_yield=None):
    """
    Calculates Intrinsic Value using a 2-Stage Free Cash Flow (FCF) DCF model.
    This provides a much more accurate valuation than the Graham EPS formula by 
    accounting for debt, cash, and strict growth caps.
    """
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    fcf = info.get("freeCashflow")
    shares = info.get("sharesOutstanding")
    total_debt = info.get("totalDebt") or 0
    total_cash = info.get("totalCash") or 0

    # Normalize Pence (GBp/GBX) to Pounds (GBP) for UK market tickers
    currency = info.get("currency", "")
    symbol = (info.get("symbol") or "").upper()
    if currency in ["GBp", "GBX"] or symbol.endswith(".L"):
        if current_price:
            current_price = current_price / 100.0

    # If we are missing core data or FCF is negative, we cannot confidently value it
    if not current_price or not fcf or fcf <= 0 or not shares:
        return {"intrinsic_value": None, "margin_of_safety": None}

    # ── DCF Assumptions ──
    discount_rate = 0.09      # 9% required rate of return (hurdle rate)
    terminal_growth = 0.02    # 2% long-term terminal growth rate (inflation avg)
    
    # Get expected growth, default to a conservative 5% if missing
    growth_rate = info.get("earningsGrowth")
    if growth_rate is None:
        growth_rate = 0.05
        
    # Strictly cap growth between 0% and 15% to prevent cyclical spikes from breaking the model
    growth_rate = max(0.0, min(growth_rate, 0.15))

    # ── Stage 1: Project FCF for 5 Years ──
    projected_fcf_pv = 0
    current_fcf = fcf
    for year in range(1, 6):
        current_fcf *= (1 + growth_rate)
        projected_fcf_pv += current_fcf / ((1 + discount_rate) ** year)

    # ── Stage 2: Terminal Value ──
    # Value of the company from year 6 to infinity
    terminal_value = (current_fcf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
    terminal_value_pv = terminal_value / ((1 + discount_rate) ** 5)

    # ── Enterprise Value to Equity Value ──
    enterprise_value = projected_fcf_pv + terminal_value_pv
    equity_value = enterprise_value + total_cash - total_debt

    intrinsic_value = equity_value / shares

    # If debt entirely wipes out the enterprise value
    if intrinsic_value <= 0:
        return {"intrinsic_value": None, "margin_of_safety": None}

    margin_of_safety = (intrinsic_value - current_price) / intrinsic_value
    
    return {
        "intrinsic_value": round(intrinsic_value, 2),
        "margin_of_safety": round(margin_of_safety * 100, 2),
    }


# ── screener caching and math helpers ─────────────────────────────────────────

def get_ticker_list():
    """Reads tickers from tickers.txt or populates a default list.

    Lines starting with '#' are section-header comments (used to bundle
    tickers by parent index, e.g. FTSE 100 / DAX 40 / CAC 40) and are
    skipped, as are blank lines.
    """
    if os.path.exists(TICKERS_FILE):
        with open(TICKERS_FILE, "r") as f:
            tickers = [
                line.strip().upper() for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
            if tickers:
                return tickers
    # UK & EU large-cap default (LSE / Xetra / Euronext Paris), ETFs excluded
    default_tickers = ['AZN.L', 'SHEL.L', 'HSBA.L', 'ULVR.L', 'BP.L',
                        'SAP.DE', 'SIE.DE', 'ALV.DE',
                        'MC.PA', 'OR.PA', 'TTE.PA']
    with open(TICKERS_FILE, "w") as f:
        for t in default_tickers:
            f.write(f"{t}\n")
    return default_tickers

# ── ticker file management endpoint ───────────────────────────────────────────

@trading_bp.route("/api/trading/tickers", methods=["GET", "POST"])
def manage_tickers():
    """Endpoint to view or update the tickers.txt file used by the screener and sync engine."""
    if request.method == "GET":
        content = ""
        if os.path.exists(TICKERS_FILE):
            with open(TICKERS_FILE, "r") as f:
                content = f.read()
        return jsonify({"content": content, "tickers": get_ticker_list()})

    # Admin verification for write operations
    token = request.headers.get("X-Admin-Token")
    if token != ADMIN_SECRET_TOKEN:
        return jsonify({"error": "Unauthorized admin action"}), 401

    data = request.get_json() or {}
    content = data.get("content", "")

    try:
        with open(TICKERS_FILE, "w") as f:
            f.write(content)
        updated_tickers = get_ticker_list()
        return jsonify({
            "message": f"Successfully updated {TICKERS_FILE}",
            "count": len(updated_tickers),
            "tickers": updated_tickers
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def update_ticker_data(symbol):
    """Updates stock data folder cache incrementally, adding only missing days."""
    symbol = symbol.upper()
    history_file = os.path.join(DATA_DIR, f"{symbol}_history.csv")
    info_file = os.path.join(DATA_DIR, f"{symbol}_info.json")
    financials_file = os.path.join(DATA_DIR, f"{symbol}_financials.csv")
    balance_file = os.path.join(DATA_DIR, f"{symbol}_balance_sheet.csv")
    insider_file = os.path.join(DATA_DIR, f"{symbol}_insider.json")
    
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

        # Insider activity: Form 4-style data. Note this is a US SEC filing
        # concept, so LSE/Xetra/Euronext names will typically come back empty
        # via yfinance — that's expected, not a bug, and is handled downstream.
        try:
            insider_data = {
                "transactions": df_to_records(ticker.get_insider_transactions()),
                "roster":       df_to_records(ticker.get_insider_roster_holders()),
                "purchases":    df_to_records(ticker.get_insider_purchases()),
            }
            with open(insider_file, "w") as f:
                json.dump(insider_data, f)
        except Exception as e:
            print(f"Error updating insider data for {symbol}: {e}")

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
    metrics["dividend_yield"] = dy if dy is not None else None

    # 8. Margin of Safety (revised Graham formula)
    mos = calculate_margin_of_safety(info)
    metrics["intrinsic_value"] = mos["intrinsic_value"]
    metrics["margin_of_safety"] = mos["margin_of_safety"]

    return metrics


def derive_insider_signal(purchases_records):
    """Best-effort read of yfinance's 'insider purchases last 6m' summary table
    into a simple Buying / Selling / Flat / No Data signal for the screener.
    yfinance's exact column names can vary by ticker/market, so this scans for
    a 'net' row and the first numeric value on it rather than a hard-coded key."""
    if not purchases_records:
        return "—"
    for rec in purchases_records:
        label_val = next(iter(rec.values()), "")
        if isinstance(label_val, str) and "net" in label_val.lower():
            for k, v in rec.items():
                if isinstance(v, (int, float)):
                    if v > 0:
                        return "Buying"
                    if v < 0:
                        return "Selling"
                    return "Flat"
    return "—"


def load_insider_summary(symbol):
    """Reads the cached insider file (written by update_ticker_data) for the screener."""
    insider_file = os.path.join(DATA_DIR, f"{symbol}_insider.json")
    if not os.path.exists(insider_file):
        return "—"
    try:
        with open(insider_file, "r") as f:
            data = json.load(f)
        return derive_insider_signal(data.get("purchases", []))
    except Exception:
        return "—"


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

        mos = calculate_margin_of_safety(info)
        out["intrinsicValue"] = mos["intrinsic_value"]
        out["marginOfSafety"] = mos["margin_of_safety"]

        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@trading_bp.route("/api/trading/insider/<symbol>")
def get_insider(symbol):
    """Live insider-activity lookup for the Market Data tab. Note: this is
    based on US SEC Form 4 filings, so results are usually empty for LSE /
    Xetra / Euronext-listed names via yfinance."""
    try:
        t = yf.Ticker(symbol.upper())
        transactions = df_to_records(t.get_insider_transactions())
        roster = df_to_records(t.get_insider_roster_holders())
        purchases = df_to_records(t.get_insider_purchases())
        return jsonify({
            "symbol": symbol.upper(),
            "transactions": transactions,
            "roster": roster,
            "purchases": purchases,
            "signal": derive_insider_signal(purchases),
        })
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
sync_status = {
    "running": False,
    "progress": 0,
    "total": 0,
    "current_ticker": "",
    "message": "Idle",
    "last_run": None,
    "errors": []
}

def run_sync_in_background():
    """Background worker function executing ticker data downloads, orphan cleanup, and insider activity detection."""
    global sync_status
    sync_status["running"] = True
    sync_status["errors"] = []
    sync_status["message"] = "Initializing dataset sync..."
    
    try:
        tickers = get_ticker_list()
        sync_status["total"] = len(tickers)
        sync_status["progress"] = 0
        
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # ── 1. Cleanup Orphaned Data Files ──
        sync_status["message"] = "Cleaning up old data..."
        active_symbols = set(tickers)
        for filename in os.listdir(DATA_DIR):
            # Explicitly protect system log and configuration files from deletion
            if filename in ("interesting_activity.json", "tickers.txt"):
                continue
                
            if filename.endswith(('.json', '.csv')):
                file_symbol = filename.split('_')[0]
                if file_symbol not in active_symbols:
                    try:
                        os.remove(os.path.join(DATA_DIR, filename))
                    except Exception as e:
                        print(f"Could not delete orphan file {filename}: {e}")

        # ── 2. Download and Cache Ticker Data ──
        for index, symbol in enumerate(tickers, 1):
            sync_status["current_ticker"] = symbol
            sync_status["progress"] = index
            sync_status["message"] = f"Syncing {symbol} ({index}/{len(tickers)})"
            
            try:
                t = yf.Ticker(symbol)
                
                # Fetch & cache Info metadata
                info = t.info or {}
                with open(os.path.join(DATA_DIR, f"{symbol}_info.json"), "w") as f:
                    json.dump(info, f, indent=2)
                    
                # Fetch & cache Financials CSV
                fin_df = t.financials
                if fin_df is not None and not fin_df.empty:
                    fin_df.to_csv(os.path.join(DATA_DIR, f"{symbol}_financials.csv"))
                    
                # Fetch & cache Balance Sheet CSV
                bs_df = t.balance_sheet
                if bs_df is not None and not bs_df.empty:
                    bs_df.to_csv(os.path.join(DATA_DIR, f"{symbol}_balance_sheet.csv"))
                    
                # Fetch & cache Insider Transactions JSON
                insider_data = {
                    "transactions": df_to_records(t.get_insider_transactions()),
                    "roster":       df_to_records(t.get_insider_roster_holders()),
                    "purchases":    df_to_records(t.get_insider_purchases()),
                }
                with open(os.path.join(DATA_DIR, f"{symbol}_insider.json"), "w") as f:
                    json.dump(insider_data, f, indent=2)

                # Fetch & cache Historical Price Data (Incremental Update)
                history_file = os.path.join(DATA_DIR, f"{symbol}_history.csv")
                if os.path.exists(history_file):
                    try:
                        existing_df = pd.read_csv(history_file, index_col=0)
                        if not existing_df.empty:
                            existing_df.index = pd.to_datetime(existing_df.index, utc=True).tz_localize(None)
                            last_date = existing_df.index.max()
                            new_df = t.history(start=last_date.strftime("%Y-%m-%d"))
                            if not new_df.empty:
                                new_df.index = pd.to_datetime(new_df.index, utc=True).tz_localize(None)
                                combined_df = pd.concat([existing_df, new_df])
                                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                                combined_df.to_csv(history_file)
                        else:
                            df_fresh = t.history(period="5y")
                            df_fresh.index = pd.to_datetime(df_fresh.index, utc=True).tz_localize(None)
                            df_fresh.to_csv(history_file)
                    except Exception:
                        df_fresh = t.history(period="5y")
                        df_fresh.index = pd.to_datetime(df_fresh.index, utc=True).tz_localize(None)
                        df_fresh.to_csv(history_file)
                else:
                    df_fresh = t.history(period="5y")
                    df_fresh.index = pd.to_datetime(df_fresh.index, utc=True).tz_localize(None)
                    df_fresh.to_csv(history_file)

                # ── 3. Scan for Significant Insider Activity ──
                company_name = info.get("longName") or info.get("shortName") or symbol
                process_interesting_activity(
                    symbol=symbol,
                    company_name=company_name,
                    insider_records=insider_data.get("transactions", []),
                    min_shares_threshold=5000  # Minimum share threshold trigger
                )

            except Exception as e:
                err_msg = f"Failed {symbol}: {str(e)}"
                print(err_msg)
                sync_status["errors"].append(err_msg)
            
            # Rate-limiting delay: Prevents CPU overheating & API rate-limiting
            time.sleep(1.0)
            
        sync_status["message"] = f"Completed sync of {len(tickers)} tickers!"
        sync_status["last_run"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
    except Exception as e:
        sync_status["message"] = f"Fatal sync error: {str(e)}"
        print(f"Critical sync thread exception: {e}")
        
    finally:
        sync_status["running"] = False
        sync_status["current_ticker"] = ""


@trading_bp.route("/api/trading/update_screener_data", methods=["POST"])
def update_screener_data():
    """Trigger background dataset sync."""
    if sync_status["running"]:
        return jsonify({
            "status": "busy",
            "message": f"Sync currently running ({sync_status['progress']}/{sync_status['total']} tickers processed)."
        }), 400

    # Launch worker in daemon thread
    thread = threading.Thread(target=run_sync_in_background)
    thread.daemon = True
    thread.start()

    return jsonify({
        "status": "started",
        "message": "Background sync started successfully on the Pi."
    })


@trading_bp.route("/api/trading/sync_status", methods=["GET"])
def get_sync_status():
    """Poll progress status from frontend UI."""
    return jsonify(sync_status)

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
        min_div_yield = float(body.get("min_div_yield", 0.0))  # Replaced min_mos with Dividend Yield (%)
        
        # Scoring & Rating inputs
        min_rating = int(body.get("min_rating", 4))
        wiggle_room = float(body.get("wiggle_room", 0))  # Provided as percentage
        
        # Compute modifiers based on wiggle room percentage
        w_lower = 1.0 - (wiggle_room / 100.0)
        w_upper = 1.0 + (wiggle_room / 100.0)

        tickers = get_ticker_list()
        results = []
        
        for symbol in tickers:
            try:
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
                m = {k: clean(v) for k, v in m.items()} 

                # Convert dividend yield to percentage format if returned as a decimal
                raw_dy = m.get("dividend_yield")
                dy_pct = (raw_dy * 100.0) if (raw_dy is not None and raw_dy < 1.0) else raw_dy

                # Evaluate the 7 distinct analytical criteria using wiggle room factors
                score = 0
                if m["gross_margin"] is not None and m["gross_margin"] >= (min_gm * w_lower): score += 1
                if m["roce"] is not None and m["roce"] >= (min_roce * w_lower): score += 1
                if m["fcf"] is not None and m["fcf"] >= (min_fcf * w_lower): score += 1
                if m["debt_equity"] is not None and m["debt_equity"] <= (max_debt_equity * w_upper): score += 1
                if m["pegy"] is not None and m["pegy"] <= (max_pegy * w_upper): score += 1
                if m["pe"] is not None and m["pe"] <= (max_pe * w_upper): score += 1
                if dy_pct is not None and dy_pct >= (min_div_yield * w_lower): score += 1
                
                # Filter matches based on user's minimum rating selection
                if score < min_rating: 
                    continue
                
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
                    "insider": load_insider_summary(symbol),
                })
            except Exception as e:
                print(f"Skipping {symbol}: {e}")
                continue

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

INTERESTING_ACTIVITY_FILE = os.path.join(DATA_DIR, "interesting_activity.json")

BLACKLISTED_TICKERS = {"AAF.L"}  # Add any other spammy tickers here if needed

def is_investment_trust(company_name):
    if not company_name:
        return False
    name = company_name.lower()
    spam_keywords = ["trust", "fund", "ord ", "blackrock", "fidelity", "jpmorgan", "aberdeen", "schiehallion"]
    return any(keyword in name for keyword in spam_keywords)


def process_interesting_activity(symbol, company_name, insider_records, min_shares_threshold=10000):
    """Scans insider records, blocks noisy tickers, and prevents duplicate monthly spam."""
    if not insider_records:
        return
    
    # 1. Explicitly drop blacklisted tickers (like Airtel Africa's daily scrip feed)
    if symbol.upper() in BLACKLISTED_TICKERS or is_investment_trust(company_name):
        return

    activity_list = []
    if os.path.exists(INTERESTING_ACTIVITY_FILE):
        try:
            with open(INTERESTING_ACTIVITY_FILE, "r") as f:
                activity_list = json.load(f)
        except Exception:
            activity_list = []

    # Map existing entries by symbol_date
    activity_map = {f"{item['symbol']}_{item['date']}": item for item in activity_list}

    for tx in insider_records:
        shares = abs(float(tx.get("shares") or tx.get("Shares") or tx.get("amount") or 0))
        tx_text = str(tx.get("text") or tx.get("transactionText") or "").lower()
        acq_disp = str(tx.get("acquisitionOrDisposition") or "").upper()
        
        is_buy = "buy" in tx_text or "purchase" in tx_text or acq_disp == "A" or shares > 0
        
        # Increased threshold to 10,000 shares to ensure we only catch massive block buys
        if is_buy and shares >= min_shares_threshold:
            date = str(tx.get("Start Date") or tx.get("startDate") or tx.get("date") or "").split("T")[0]
            if not date:
                continue
                
            composite_key = f"{symbol}_{date}"
            
            # Rate limit check: Ensure this symbol hasn't already logged an event in the same month
            month_prefix = date[:7] # YYYY-MM
            already_logged_this_month = any(
                item["symbol"] == symbol and item["date"].startswith(month_prefix) 
                for item in activity_map.values()
            )
            if already_logged_this_month:
                continue

            raw_insider = tx.get("filerName") or tx.get("insider") or tx.get("name") or "Director"
            insider = "Board Member / Director" if "executive" in str(raw_insider).lower() else raw_insider
            relation = tx.get("position") or tx.get("title") or tx.get("relation") or "Management"

            if composite_key in activity_map:
                activity_map[composite_key]["shares"] += shares
            else:
                activity_map[composite_key] = {
                    "symbol": symbol,
                    "name": company_name,
                    "insider": insider,
                    "relation": relation,
                    "shares": shares,
                    "date": date,
                    "detected_at": datetime.now().strftime("%Y-%m-%d %H:%M")
                }

    final_list = sorted(list(activity_map.values()), key=lambda x: x["date"], reverse=True)[:250]
    
    with open(INTERESTING_ACTIVITY_FILE, "w") as f:
        json.dump(final_list, f, indent=2)

@trading_bp.route("/api/trading/interesting_activity", methods=["GET"])
def get_interesting_activity():
    """Returns the flagged insider buy activity log."""
    if os.path.exists(INTERESTING_ACTIVITY_FILE):
        try:
            with open(INTERESTING_ACTIVITY_FILE, "r") as f:
                data = json.load(f)
            # Filter out Trust items on read just in case
            filtered = [
                item for item in data 
                if not is_investment_trust(item.get("name"))
            ]
            return jsonify({"activity": filtered})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"activity": []})