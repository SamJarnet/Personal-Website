"""
strategy_engine.py
Core backtesting engine ported directly from main.py.
This encapsulates the pure mathematical and algorithmic trading logic.
"""

import math
import pandas as pd
import numpy as np


def compute_signals(df: pd.DataFrame, fast_len: int = 20, slow_len: int = 50, stop_len: int = 150):
    df = df.copy()
    
    df["sma_fast"] = df["Close"].rolling(fast_len).mean()
    df["sma_slow"] = df["Close"].rolling(slow_len).mean()
    
    # df["RSI"] = compute_rsi(df, rsi_len)
    
    df["signal"] = 0 
    
    #df.loc[(df["sma_fast"] > df["sma_slow"]) & (df["RSI"] < rsi_buy), "signal"] = 1
    #df.loc[(df["sma_fast"] < df["sma_slow"]) | (df["RSI"] > rsi_sell), "signal"] = -1
    
    df.loc[(df["sma_fast"] > df["sma_slow"]), "signal"] = 1
    df.loc[(df["sma_fast"] < df["sma_slow"]), "signal"] = 0
   
    df["signal_change"] = df["signal"].diff()
    
    df["stop_ma"] = df["Close"].rolling(stop_len).mean()
    
    return df


def trade_loop(df, starting_capital=10000.0, position_size=1.0):
    cash = starting_capital
    shares_held = 0
    trades = []
    capital_tracker = []  
    open_trade = None
        

    for row in df.itertuples():
        signal = row.signal_change
        price = row.Close
        date = row.Index

        if signal == 1 and open_trade is None:
            investment_amount = cash * position_size
            shares_held = investment_amount // price 
            if shares_held > 0:
                cost = shares_held * price
                cash -= cost
                open_trade = ("buy", date, price, shares_held)

        elif open_trade is not None:
            

            stop_price = row.stop_ma
            if signal == -1 or price < stop_price:
                sell_price = price
                revenue = shares_held * sell_price
                cash += revenue
                
                pnl = (sell_price - open_trade[2]) * shares_held
                sell_trade = ("sell", date, sell_price)
                trades.append((open_trade, sell_trade, pnl))
                
                open_trade = None
                shares_held = 0

        current_valuation = cash + (shares_held * price)
        capital_tracker.append(current_valuation)


    padding_needed = len(df) - len(capital_tracker)
    if padding_needed > 0:
        capital_tracker = [starting_capital] * padding_needed + capital_tracker

    final_profit = capital_tracker[-1] - starting_capital
    return trades, final_profit, pd.Series(capital_tracker, index=df.index)