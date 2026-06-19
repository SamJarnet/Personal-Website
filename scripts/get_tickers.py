from pytickersymbols import PyTickerSymbols


stock_data = PyTickerSymbols()
yahoo_tickers = set()

all_indices = stock_data.get_all_indices()

for index_name in all_indices:
    stocks = stock_data.get_stocks_by_index(index_name)
    
    for stock in stocks:
        symbols_list = stock.get('symbols', [])
        for sym_entry in symbols_list:
            yahoo_ticker = sym_entry.get('yahoo')
            if yahoo_ticker:
                yahoo_tickers.add(yahoo_ticker.strip().upper())

sorted_tickers = sorted(list(yahoo_tickers))

output_filename = "y_tickers.txt"
with open(output_filename, "w") as f:
    for ticker in sorted_tickers:
        f.write(f"{ticker}\n")
