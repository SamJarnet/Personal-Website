import io
import pandas as pd
import requests


def get_yfinance_index_tickers():
    index_configs = [
        {
            "name": "FTSE 100",
            "url": "https://en.wikipedia.org/wiki/FTSE_100_Index",
            "suffix": ".L",
        },
        {
            "name": "FTSE 250",
            "url": "https://en.wikipedia.org/wiki/FTSE_250_Index",
            "suffix": ".L",
        },
        {
            "name": "CAC 40",
            "url": "https://en.wikipedia.org/wiki/CAC_40",
            "suffix": ".PA",
        },
        {
            "name": "DAX 40",
            "url": "https://en.wikipedia.org/wiki/DAX",
            "suffix": ".DE",
        },
    ]

    # Custom User-Agent to bypass HTTP 403 Forbidden
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    formatted_tickers = []

    for config in index_configs:
        response = requests.get(config["url"], headers=headers)
        # Parse text content explicitly
        tables = pd.read_html(io.StringIO(response.text))

        for df in tables:
            ticker_col = next(
                (
                    col
                    for col in df.columns
                    if any(
                        k in str(col).lower()
                        for k in ["ticker", "epic", "symbol"]
                    )
                ),
                None,
            )

            if ticker_col is not None:
                raw_tickers = df[ticker_col].dropna().astype(str).tolist()

                for ticker in raw_tickers:
                    clean_ticker = ticker.strip().replace(".", "-").split()[0]
                    formatted_tickers.append(f"{clean_ticker}{config['suffix']}\n")
                break

    return sorted(list(set(formatted_tickers)))


tickers = get_yfinance_index_tickers()
with open('tickers.txt', 'w') as file:
    file.writelines(tickers)