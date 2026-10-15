"""
admin_engine.py
Engine for Admin functions: Screener computations, ticker cache updates,
and insider buy activity tracking.
"""

import os
import math
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TICKERS_FILE = os.path.join(BASE_DIR, "tickers.txt")
INTERESTING_ACTIVITY_FILE = os.path.join(DATA_DIR, "interesting_activity.json")
BLACKLISTED_TICKERS = {"AAF.L"}

os.makedirs(DATA_DIR, exist_ok=True)

sync_status = {
    "running": False,
    "progress": 0,
    "total": 0,
    "current_ticker": "",
    "message": "Idle",
    "last_run": None,
    "errors": []
}

def df_to_records(df):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    out = []
    for _, row in df.reset_index().iterrows():
        rec = {}
        for k, v in row.items():
            if hasattr(v, "strftime"):
                try: v = v.strftime("%Y-%m-%d")
                except Exception: v = str(v)
            elif isinstance(v, (np.integer, np.floating)):
                v = float(v) if isinstance(v, np.floating) else int(v)
            else:
                try:
                    if pd.isna(v): v = None
                except (TypeError, ValueError): pass
            rec[str(k)] = v
        out.append(rec)
    return out

def get_ticker_list():
    if os.path.exists(TICKERS_FILE):
        with open(TICKERS_FILE, "r") as f:
            tickers = [
                line.strip().upper() for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
            if tickers: return tickers
    default_tickers = ['AZN.L', 'SHEL.L', 'HSBA.L', 'ULVR.L', 'BP.L', 'SAP.DE', 'SIE.DE', 'ALV.DE', 'MC.PA', 'OR.PA', 'TTE.PA']
    with open(TICKERS_FILE, "w") as f:
        for t in default_tickers: f.write(f"{t}\n")
    return default_tickers

def calculate_margin_of_safety(info):
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    fcf = info.get("freeCashflow")
    shares = info.get("sharesOutstanding")
    total_debt = info.get("totalDebt") or 0
    total_cash = info.get("totalCash") or 0

    currency = info.get("currency", "")
    symbol = (info.get("symbol") or "").upper()
    if currency in ["GBp", "GBX"] or symbol.endswith(".L"):
        if current_price: current_price /= 100.0

    if not current_price or not fcf or fcf <= 0 or not shares:
        return {"intrinsic_value": None, "margin_of_safety": None}

    discount_rate = 0.09
    terminal_growth = 0.02
    growth_rate = info.get("earningsGrowth") or 0.05
    growth_rate = max(0.0, min(growth_rate, 0.15))

    projected_fcf_pv = 0
    current_fcf = fcf
    for year in range(1, 6):
        current_fcf *= (1 + growth_rate)
        projected_fcf_pv += current_fcf / ((1 + discount_rate) ** year)

    terminal_value = (current_fcf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
    terminal_value_pv = terminal_value / ((1 + discount_rate) ** 5)

    enterprise_value = projected_fcf_pv + terminal_value_pv
    equity_value = enterprise_value + total_cash - total_debt
    intrinsic_value = equity_value / shares

    if intrinsic_value <= 0:
        return {"intrinsic_value": None, "margin_of_safety": None}

    margin_of_safety = (intrinsic_value - current_price) / intrinsic_value
    return {
        "intrinsic_value": round(intrinsic_value, 2),
        "margin_of_safety": round(margin_of_safety * 100, 2),
    }

def compute_metrics(info, financials=None, balance_sheet=None):
    metrics = {}
    gm = info.get("grossMargins")
    metrics["gross_margin"] = gm * 100 if gm is not None else None
    metrics["pe"] = info.get("trailingPE") or info.get("forwardPE")
    de = info.get("debtToEquity")
    metrics["debt_equity"] = de / 100.0 if de and de > 5 else de
    fcf = info.get("freeCashflow")
    metrics["fcf"] = fcf / 1e9 if fcf is not None else None
    
    pe = metrics["pe"]
    eg = info.get("earningsGrowth") 
    dy = info.get("dividendYield")   
    if pe and eg:
        growth_total = (eg * 100) + ((dy * 100) if dy else 0)
        metrics["pegy"] = pe / growth_total if growth_total > 0 else None
    else: metrics["pegy"] = None

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
                if cap_employed > 0: roce = (ebit / cap_employed) * 100
        except Exception: pass
    if roce is None:
        roe = info.get("returnOnEquity")
        roce = roe * 100 if roe is not None else None
        
    metrics["roce"] = roce
    metrics["dividend_yield"] = dy if dy is not None else None
    mos = calculate_margin_of_safety(info)
    metrics["intrinsic_value"] = mos["intrinsic_value"]
    metrics["margin_of_safety"] = mos["margin_of_safety"]
    return metrics

def derive_insider_signal(purchases_records):
    if not purchases_records: return "—"
    for rec in purchases_records:
        label_val = next(iter(rec.values()), "")
        if isinstance(label_val, str) and "net" in label_val.lower():
            for k, v in rec.items():
                if isinstance(v, (int, float)):
                    if v > 0: return "Buying"
                    if v < 0: return "Selling"
                    return "Flat"
    return "—"

def load_insider_summary(symbol):
    insider_file = os.path.join(DATA_DIR, f"{symbol}_insider.json")
    if not os.path.exists(insider_file): return "—"
    try:
        with open(insider_file, "r") as f: data = json.load(f)
        return derive_insider_signal(data.get("purchases", []))
    except Exception: return "—"

def is_investment_trust(company_name):
    if not company_name: return False
    name = company_name.lower()
    spam_keywords = ["trust", "fund", "ord ", "blackrock", "fidelity", "jpmorgan", "aberdeen", "schiehallion"]
    return any(keyword in name for keyword in spam_keywords)

def process_interesting_activity(symbol, company_name, insider_records, min_shares_threshold=10000):
    if not insider_records or symbol.upper() in BLACKLISTED_TICKERS or is_investment_trust(company_name):
        return
    activity_list = []
    if os.path.exists(INTERESTING_ACTIVITY_FILE):
        try:
            with open(INTERESTING_ACTIVITY_FILE, "r") as f: activity_list = json.load(f)
        except Exception: activity_list = []

    activity_map = {f"{item['symbol']}_{item['date']}": item for item in activity_list}

    for tx in insider_records:
        shares = abs(float(tx.get("shares") or tx.get("Shares") or tx.get("amount") or 0))
        tx_text = str(tx.get("text") or tx.get("transactionText") or "").lower()
        acq_disp = str(tx.get("acquisitionOrDisposition") or "").upper()
        is_buy = "buy" in tx_text or "purchase" in tx_text or acq_disp == "A" or shares > 0
        
        if is_buy and shares >= min_shares_threshold:
            date = str(tx.get("Start Date") or tx.get("startDate") or tx.get("date") or "").split("T")[0]
            if not date: continue
            composite_key = f"{symbol}_{date}"
            month_prefix = date[:7]
            if any(item["symbol"] == symbol and item["date"].startswith(month_prefix) for item in activity_map.values()):
                continue

            raw_insider = tx.get("filerName") or tx.get("insider") or tx.get("name") or "Director"
            insider = "Board Member / Director" if "executive" in str(raw_insider).lower() else raw_insider
            relation = tx.get("position") or tx.get("title") or tx.get("relation") or "Management"

            if composite_key in activity_map:
                activity_map[composite_key]["shares"] += shares
            else:
                activity_map[composite_key] = {
                    "symbol": symbol, "name": company_name, "insider": insider,
                    "relation": relation, "shares": shares, "date": date,
                    "detected_at": datetime.now().strftime("%Y-%m-%d %H:%M")
                }

    final_list = sorted(list(activity_map.values()), key=lambda x: x["date"], reverse=True)[:250]
    with open(INTERESTING_ACTIVITY_FILE, "w") as f:
        json.dump(final_list, f, indent=2)

def run_sync_in_background():
    global sync_status
    sync_status["running"] = True
    sync_status["errors"] = []
    sync_status["message"] = "Initializing dataset sync..."
    try:
        tickers = get_ticker_list()
        sync_status["total"] = len(tickers)
        sync_status["progress"] = 0
        
        active_symbols = set(tickers)
        for filename in os.listdir(DATA_DIR):
            if filename in ("interesting_activity.json", "tickers.txt"): continue
            if filename.endswith(('.json', '.csv')):
                file_symbol = filename.split('_')[0]
                if file_symbol not in active_symbols:
                    try: os.remove(os.path.join(DATA_DIR, filename))
                    except Exception: pass

        for index, symbol in enumerate(tickers, 1):
            sync_status["current_ticker"] = symbol
            sync_status["progress"] = index
            sync_status["message"] = f"Syncing {symbol} ({index}/{len(tickers)})"
            try:
                t = yf.Ticker(symbol)
                info = t.info or {}
                with open(os.path.join(DATA_DIR, f"{symbol}_info.json"), "w") as f:
                    json.dump(info, f, indent=2)
                fin_df = t.financials
                if fin_df is not None and not fin_df.empty:
                    fin_df.to_csv(os.path.join(DATA_DIR, f"{symbol}_financials.csv"))
                bs_df = t.balance_sheet
                if bs_df is not None and not bs_df.empty:
                    bs_df.to_csv(os.path.join(DATA_DIR, f"{symbol}_balance_sheet.csv"))
                
                # Fetch and save historical daily stock prices incrementally
                history_file = os.path.join(DATA_DIR, f"{symbol}_history.csv")
                df_fresh = t.history(period="5y")
                if df_fresh is not None and not df_fresh.empty:
                    df_fresh.index = pd.to_datetime(df_fresh.index, utc=True).tz_localize(None)
                    df_fresh.to_csv(history_file)

                insider_data = {
                    "transactions": df_to_records(t.get_insider_transactions()),
                    "roster": df_to_records(t.get_insider_roster_holders()),
                    "purchases": df_to_records(t.get_insider_purchases()),
                }
                with open(os.path.join(DATA_DIR, f"{symbol}_insider.json"), "w") as f:
                    json.dump(insider_data, f, indent=2)

                process_interesting_activity(
                    symbol=symbol, company_name=info.get("longName") or symbol,
                    insider_records=insider_data.get("transactions", []), min_shares_threshold=5000
                )
            except Exception as e:
                sync_status["errors"].append(f"Failed {symbol}: {str(e)}")
            time.sleep(1.0)
        sync_status["message"] = f"Completed sync of {len(tickers)} tickers!"
        sync_status["last_run"] = time.strftime("%Y-%m-%d %H:%M:%S")
    finally:
        sync_status["running"] = False
        sync_status["current_ticker"] = ""