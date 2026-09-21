import math
import pandas as pd

def compute_signals(df, name="ticker", sma_fast_len=20, sma_slow_len=50, rsi_len=14, rsi_buy=20, rsi_sell=80, stop_ma_len=150):
    df = df.copy()
    
    df["sma_fast"] = df["Close"].rolling(sma_fast_len).mean()
    df["sma_slow"] = df["Close"].rolling(sma_slow_len).mean()
    
    df["signal"] = 0 
    
    df.loc[(df["sma_fast"] > df["sma_slow"]), "signal"] = 1
    df.loc[(df["sma_fast"] < df["sma_slow"]), "signal"] = 0
   
    df["signal_change"] = df["signal"].diff().shift(1)
    
    df["stop_ma"] = df["Close"].rolling(stop_ma_len).mean()
    return df

def trade_loop(df, starting_capital=10000.0, position_size=1.0, transaction_fee=0.001):
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
                transaction_cost = shares_held * price * transaction_fee
                cost = (shares_held * price) + transaction_cost
                cash -= cost
                open_trade = ("buy", date, price, shares_held, transaction_cost)

        elif open_trade is not None:
            stop_price = row.stop_ma
            if signal == -1 or price < stop_price:
                sell_price = price
                transaction_cost = shares_held * price * transaction_fee
                revenue = (shares_held * sell_price) - transaction_cost
                cash += revenue

                pnl = revenue - (open_trade[2] * shares_held + open_trade[4])
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

def calculate_risk_metrics(capital_series):
    capital = list(capital_series)
    returns = []

    for i in range(1, len(capital)):
        daily_return = (capital[i] / capital[i-1])-1
        returns.append(daily_return)

    if not returns:
        return 0, 0, 0

    mean_return = sum(returns)/len(returns)
    count = 0
    for r in returns:
        count += (r-mean_return) ** 2
        
    variance = count / len(returns)
    daily_volatility = math.sqrt(variance)

    annual_volatility = daily_volatility * math.sqrt(252)

    maximum_drawdown = 0
    peak = capital[0]

    for value in capital:
        if value > peak:
            peak = value
        drawdown = (value - peak) / peak
        if drawdown < maximum_drawdown:
            maximum_drawdown = drawdown

    if daily_volatility != 0:
        annual_sharpe = (mean_return/daily_volatility) * math.sqrt(252)
    else:
        annual_sharpe = 0

    return (annual_volatility, maximum_drawdown, annual_sharpe)

def calculate_noalgorithm_trade(ticker, df_ticker, investment_per_ticker, average_dividend_yield=0.045, years_held=1.0):
    start_price = df_ticker.iloc[0]["Close"]
    end_price = df_ticker.iloc[-1]["Close"]
    shares_held = investment_per_ticker // start_price
    total_gain = shares_held * (end_price - start_price)
    
    invested_amount = shares_held * start_price
    dividend_gain = invested_amount * average_dividend_yield * years_held
    
    remaining_money = investment_per_ticker - invested_amount
    buy_hold = (df_ticker["Close"] * shares_held) + remaining_money
    
    return total_gain, dividend_gain, buy_hold

def calculate_yearly_return(end_price, start_price, length):
    if length <= 0 or start_price <= 0: return 1.0
    return (end_price/start_price)**(1/length)