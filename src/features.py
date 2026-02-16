import pandas as pd
import numpy as np
import os

def load_raw_data(filepath):
    """Load raw data from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    # Read CSV, setting Date as index and parsing dates
    return pd.read_csv(filepath, index_col='Date', parse_dates=True)

def create_features(df):
    """Generate technical and macroeconomic features."""
    # Work on a copy to avoid SettingWithCopy warnings
    df = df.copy()
    
    # 1. Target Creation (Forward Looking)
    # We want to predict the price 3 months ahead (t+3)
    # The target for today (t) is the price at (t+3)
    df['Target_Copper_3M'] = df['Copper_Price'].shift(-3)    
    # NEW: Target expressed as Return (Stationary Target)
    # Target Return = (Price_t+3 - Price_t) / Price_t
    df['Target_Copper_Return_3M'] = df['Copper_Price'].pct_change(3).shift(-3)
    # 2. Lagged Returns (Momentum)
    # How much did price change over last 1 and 3 months?
    # This captures momentum and trend
    df['Copper_Return_1M'] = df['Copper_Price'].pct_change(1)
    df['Copper_Return_3M'] = df['Copper_Price'].pct_change(3)

    # 3. Price Ratios (Relative Value)
    # Copper vs Gold (Risk Sentiment / Industrial vs Precious)
    if 'Gold_Price' in df.columns:
        df['Copper_Gold_Ratio'] = df['Copper_Price'] / df['Gold_Price']
    
    # Copper vs Aluminum (Substitution Effect)
    if 'Aluminum_Price' in df.columns:
        df['Copper_Aluminum_Ratio'] = df['Copper_Price'] / df['Aluminum_Price']

    # Copper vs Oil (Energy Input Cost / Global Growth)
    if 'Oil_Price' in df.columns:
        df['Copper_Oil_Ratio'] = df['Copper_Price'] / df['Oil_Price']

    # 4. Macro Variance (Rate of Change)
    # Instead of raw levels, use % change for correlations where trends matter more than levels
    # DXY: Dollar strength change
    if 'DXY' in df.columns:
        df['DXY_Change'] = df['DXY'].pct_change()
        # Lagged DXY (Currency impact takes time)
        df['DXY_Change_Lag1'] = df['DXY_Change'].shift(1)
        
    # China Growth: Industrial demand change
    if 'China_Manuf_Prod' in df.columns:
        df['China_Manuf_Change'] = df['China_Manuf_Prod'].pct_change()
        # Lagged China PMI (Leading indicator)
        df['China_Manuf_Change_Lag1'] = df['China_Manuf_Change'].shift(1)
        
    # AUD/USD: Commodity currency strength change
    if 'AUD_USD' in df.columns:
        df['AUD_USD_Change'] = df['AUD_USD'].pct_change()

    # 5. Volatility (Risk)
    # Rolling standard deviation of returns (3M and 6M)
    df['Copper_Vol_3M'] = df['Copper_Return_1M'].rolling(window=3).std()
    df['Copper_Vol_6M'] = df['Copper_Return_1M'].rolling(window=6).std()

    # 6. Trend (Moving Averages)
    # Price relative to SMA (Signal: >1 is Bullish trend, <1 is Bearish)
    df['Copper_SMA_3M'] = df['Copper_Price'] / df['Copper_Price'].rolling(window=3).mean()
    df['Copper_SMA_6M'] = df['Copper_Price'] / df['Copper_Price'].rolling(window=6).mean()

    # 7. Seasonality (Cyclical Features)
    # Encode Month as Sin/Cos to preserve cyclic nature (Dec close to Jan)
    df['Month_Sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df.index.month / 12)

    # 8. Data Cleaning
    df = df.dropna(subset=['Copper_Price'])
    
    target_col = 'Target_Copper_3M'
    return_target_col = 'Target_Copper_Return_3M'
    
    # Exclude BOTH targets from ffill
    exclude_cols = [target_col, return_target_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Forward fill ONLY the feature columns
    df[feature_cols] = df[feature_cols].ffill()
    
    # Drop rows where features are still NaN (at the start of the dataset)
    df = df.dropna(subset=feature_cols)

    return df

def main():
    input_path = 'data/raw/copper_raw.csv'
    output_path = 'data/processed/copper_features.csv'
    
    print(f"Loading raw data from {input_path}...")
    try:
        raw_df = load_raw_data(input_path)
    except FileNotFoundError as e:
        print(e)
        return

    print("Engineering features...")
    processed_df = create_features(raw_df)
    
    # Ensure processed directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    processed_df.to_csv(output_path)
    print(f"Success! Processed data saved to: {output_path}")
    print(f"Final Dataset Shape: {processed_df.shape}")
    
    print("\nSample Data (Tail):")
    print(processed_df[['Copper_Price', 'Target_Copper_3M', 'Copper_Return_1M']].tail())

if __name__ == "__main__":
    main()
