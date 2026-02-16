import os
import yaml
import yfinance as yf
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()
FRED_KEY= os.getenv("FRED_API_KEY")

if not FRED_KEY:
    raise ValueError('FRED_API_KEY not found in environment variables. Please set it in your .env file.')

fred=Fred(api_key=FRED_KEY)

def load_config(config_path="config/config.yaml"):
    """Load configuration variables."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def fetch_yfinance_data(tickers, start_date, end_date):
    """Fetch various financial data from Yahoo Finance."""
    data_dict = {}
    for name, ticker in tickers.items():
        print(f"Fetching {name} ({ticker})...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval='1mo', progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Using 'Adj Close' or 'Close'
            price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            data_dict[name] = df[price_col]
        except Exception as e:
            print(f"Error fetching {name} ({ticker}): {e}")
    
    return pd.DataFrame(data_dict)

def fetch_fred_data(indicators, start_date, end_date):
    """Fetch macroeconomic indicators from FRED."""
    macro_data = []
    for series_id, name in indicators.items():
        print(f"Fetching {name} from FRED ({series_id})...")
        try:
            series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            # FRED data might be daily/monthly, resample will handle it later
            series = series.resample('MS').mean()  # Ensure monthly start
            series.name = name
            macro_data.append(series)
        except Exception as e:
            print(f"Error fetching {name} ({series_id}): {e}")
            
    if macro_data:
        return pd.concat(macro_data, axis=1)
    return pd.DataFrame()

def main():
    config = load_config()
    start_date = config.get('start_date', '2000-01-01')
    end_date = config.get('end_date', '2024-12-31')
    
    # 1. Yahoo Finance Data Sources
    yf_tickers = config.get('yfinance_tickers', {})

    # 2. FRED Data Sources
    fred_indicators = config.get('fred_indicators', {})

    print("--- Fetching Yahoo Finance Data ---")
    yf_df = fetch_yfinance_data(yf_tickers, start_date, end_date)
    
    print("\n--- Fetching FRED Data ---")
    fred_df = fetch_fred_data(fred_indicators, start_date, end_date)
    
    # Merge datasets
    print("\n--- Merging Data ---")
    # Resample everything to Month Start to match
    yf_df = yf_df.resample('MS').mean()
    fred_df = fred_df.resample('MS').mean()
    
    final_df = pd.merge(yf_df, fred_df, left_index=True, right_index=True, how='outer')
    
    # Filter to start date
    final_df = final_df[final_df.index >= pd.to_datetime(start_date)]
    
    # Fill logic (Forward fill to handle minor gaps, then drop remaining NaNs)
    final_df = final_df.ffill()
    
    # Save Raw Data
    output_path = 'data/raw/copper_raw.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path)
    print(f"\nSaved raw data to {output_path}")
    
    # Display check
    print("\nData Overview:")
    print(final_df.tail())
    print("\nMissing Values:")
    print(final_df.isna().sum())

if __name__ == "__main__":
    main()

