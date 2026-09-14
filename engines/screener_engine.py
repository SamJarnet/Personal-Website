"""
screener_engine.py
Core quantitative screening and valuation engine.
Encapsulates DCF valuation models, metric derivation, and stock filtering logic.
"""

import os
import json
import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

_bond_yield_cache = {"date": None, "value": 4.5}


def clean_val(val):
    """Sanitizes numerical outputs for clean JSON serialization."""
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


def get_current_bond_yield():
    """Fetches the 10Y US Treasury yield (^TNX) with a single-day cache fallback."""
    today_str = datetime.today().strftime("%Y-%m-%d")
    if _bond_yield_cache["date"] == today_str:
        return _bond_yield_cache["value"]
    try:
        tnx = yf.Ticker("^TNX").history(period="5d")
        if tnx is not None and not tnx.empty:
            y = float(tnx["Close"].dropna().iloc[-1])
            if y > 0:
                _bond_yield_cache["value"] = y
    except Exception:
        pass
    _bond_yield_cache["date"] = today_str
    return _bond_yield_cache["value"]


def calculate_margin_of_safety(info):
    """Calculates Intrinsic Value using a 2-Stage Free Cash Flow (FCF) DCF model."""
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    fcf = info.get("freeCashflow")
    shares = info.get("sharesOutstanding")
    total_debt = info.get("totalDebt") or 0
    total_cash = info.get("totalCash") or 0

    currency = info.get("currency", "")
    symbol = (info.get("symbol") or "").upper()
    if currency in ["GBp", "GBX"] or symbol.endswith(".L"):
        if current_price:
            current_price /= 100.0

    if not current_price or not fcf or fcf <= 0 or not shares:
        return {"intrinsic_value": None, "margin_of_safety": None}

    discount_rate = 0.09
    terminal_growth = 0.02
    growth_rate = info.get("earningsGrowth") or 0.05
    growth_rate = max(0.0, min(growth_rate, 0.15))

    # Stage 1: 5-Year Projection
    projected_fcf_pv = 0
    current_fcf = fcf
    for year in range(1, 6):
        current_fcf *= (1 + growth_rate)
        projected_fcf_pv += current_fcf / ((1 + discount_rate) ** year)

    # Stage 2: Terminal Value
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
    """Parses core financial metrics from raw balance sheet and income statements."""
    metrics = {}
    gm = info.get("grossMargins")
    metrics["gross_margin"] = gm * 100 if gm is not None else None
    metrics["pe"] = info.get("trailingPE") or info.get("forwardPE")
    
    de = info.get("debtToEquity")
    metrics["debt_equity"] = de / 100.0 if de and de > 5 else de
    
    fcf = info.get("freeCashflow")
    metrics["fcf"] = fcf / 1e9 if fcf is not None else None

    pe, eg, dy = metrics["pe"], info.get("earningsGrowth"), info.get("dividendYield")
    if pe and eg:
        growth_total = (eg * 100) + ((dy * 100) if dy else 0)
        metrics["pegy"] = pe / growth_total if growth_total > 0 else None
    else:
        metrics["pegy"] = None

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
    metrics["dividend_yield"] = dy
    
    mos = calculate_margin_of_safety(info)
    metrics["intrinsic_value"] = mos["intrinsic_value"]
    metrics["margin_of_safety"] = mos["margin_of_safety"]
    return metrics


def derive_insider_signal(purchases_records):
    """Evaluates insider purchase logs into Buying/Selling/Flat signals."""
    if not purchases_records:
        return "—"
    for rec in purchases_records:
        label_val = next(iter(rec.values()), "")
        if isinstance(label_val, str) and "net" in label_val.lower():
            for k, v in rec.items():
                if isinstance(v, (int, float)):
                    return "Buying" if v > 0 else ("Selling" if v < 0 else "Flat")
    return "—"


def screen_stocks(tickers, data_dir, filters):
    """
    Main quantitative screening algorithm.
    Filters local dataset cache against 7 criteria targets with tolerance bounds.
    """
    min_gm = float(filters.get("min_gm", 30))
    min_roce = float(filters.get("min_roce", 15))
    min_fcf = float(filters.get("min_fcf", 1.0))
    max_de = float(filters.get("max_debt_equity", 0.5))
    max_pegy = float(filters.get("max_pegy", 1.0))
    max_pe = float(filters.get("max_pe", 20))
    min_div = float(filters.get("min_div_yield", 0.0))
    min_rating = int(filters.get("min_rating", 4))
    wiggle = float(filters.get("wiggle_room", 0)) / 100.0

    w_lower, w_upper = 1.0 - wiggle, 1.0 + wiggle
    results = []

    for symbol in tickers:
        try:
            info_file = os.path.join(data_dir, f"{symbol}_info.json")
            if not os.path.exists(info_file):
                continue

            with open(info_file, "r") as f:
                info = json.load(f)

            fin_file = os.path.join(data_dir, f"{symbol}_financials.csv")
            bs_file = os.path.join(data_dir, f"{symbol}_balance_sheet.csv")
            fin_df = pd.read_csv(fin_file, index_col=0) if os.path.exists(fin_file) else None
            bs_df = pd.read_csv(bs_file, index_col=0) if os.path.exists(bs_file) else None

            m = compute_metrics(info, fin_df, bs_df)
            m = {k: clean_val(v) for k, v in m.items()}

            raw_dy = m.get("dividend_yield")
            dy_pct = (raw_dy * 100.0) if (raw_dy is not None and raw_dy < 1.0) else raw_dy

            score = 0
            if m["gross_margin"] is not None and m["gross_margin"] >= (min_gm * w_lower): score += 1
            if m["roce"] is not None and m["roce"] >= (min_roce * w_lower): score += 1
            if m["fcf"] is not None and m["fcf"] >= (min_fcf * w_lower): score += 1
            if m["debt_equity"] is not None and m["debt_equity"] <= (max_de * w_upper): score += 1
            if m["pegy"] is not None and m["pegy"] <= (max_pegy * w_upper): score += 1
            if m["pe"] is not None and m["pe"] <= (max_pe * w_upper): score += 1
            if dy_pct is not None and dy_pct >= (min_div * w_lower): score += 1

            if score < min_rating:
                continue

            # Insider Summary Signal
            insider_file = os.path.join(data_dir, f"{symbol}_insider.json")
            insider_sig = "—"
            if os.path.exists(insider_file):
                try:
                    with open(insider_file, "r") as f:
                        insider_sig = derive_insider_signal(json.load(f).get("purchases", []))
                except Exception:
                    pass

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
                "insider": insider_sig,
            })
        except Exception as e:
            continue

    results.sort(key=lambda x: int(x["rating"].split('/')[0]), reverse=True)
    return results