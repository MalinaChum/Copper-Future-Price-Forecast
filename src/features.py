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
    # Target expressed as Return (Stationary Target)
    # Target Return = (Price_t+3 - Price_t) / Price_t
    df['Target_Copper_Return_3M'] = df['Copper_Price'].pct_change(3).shift(-3)
    # 2. Lagged Returns (Momentum) — USE CHANGES, NOT RAW PRICES
    # Raw price lags cause the "delay" problem (model just copies past price).
    # Instead, use % changes which capture DIRECTION of movement.
    df['Copper_Return_1M'] = df['Copper_Price'].pct_change(1)
    df['Copper_Return_3M'] = df['Copper_Price'].pct_change(3)
    df['Copper_Return_6M'] = df['Copper_Price'].pct_change(6)
    df['Copper_Return_12M'] = df['Copper_Price'].pct_change(12)

    # --- Momentum Acceleration (2nd Derivative) ---
    # Is the trend SPEEDING UP or SLOWING DOWN?
    # Positive = accelerating up, Negative = decelerating / reversing
    df['Copper_Accel_1M'] = df['Copper_Return_1M'] - df['Copper_Return_1M'].shift(1)
    df['Copper_Accel_3M'] = df['Copper_Return_3M'] - df['Copper_Return_3M'].shift(3)
    
    # --- Momentum Divergence (Trend Reversal Detection) ---
    # When short-term momentum diverges from long-term = potential reversal
    df['Momentum_Divergence_1v3'] = df['Copper_Return_1M'] - df['Copper_Return_3M']
    df['Momentum_Divergence_3v6'] = df['Copper_Return_3M'] - df['Copper_Return_6M']
    df['Momentum_Divergence_3v12'] = df['Copper_Return_3M'] - df['Copper_Return_12M']
    
    # --- Cross-Asset Momentum (Leading Indicators for Copper) ---
    # These assets often LEAD copper price moves
    if 'Oil_Price' in df.columns:
        df['Oil_Return_1M'] = df['Oil_Price'].pct_change(1)
        df['Oil_Return_3M'] = df['Oil_Price'].pct_change(3)
        df['Oil_Accel_1M'] = df['Oil_Return_1M'] - df['Oil_Return_1M'].shift(1)
        # Oil-Copper momentum spread (divergence = copper will catch up)
        df['Oil_Copper_Mom_Spread'] = df['Oil_Return_3M'] - df['Copper_Return_3M']
    if 'Gold_Price' in df.columns:
        df['Gold_Return_1M'] = df['Gold_Price'].pct_change(1)
        df['Gold_Return_3M'] = df['Gold_Price'].pct_change(3)
        # Gold-Copper: risk sentiment shift
        df['Gold_Copper_Mom_Spread'] = df['Gold_Return_3M'] - df['Copper_Return_3M']
    if 'SP500' in df.columns:
        df['SP500_Return_1M'] = df['SP500'].pct_change(1)
        df['SP500_Return_3M'] = df['SP500'].pct_change(3)
        df['SP500_Accel_1M'] = df['SP500_Return_1M'] - df['SP500_Return_1M'].shift(1)
        # Equity-Copper divergence (equities often lead commodities)
        df['SP500_Copper_Mom_Spread'] = df['SP500_Return_3M'] - df['Copper_Return_3M']
    if 'Aluminum_Price' in df.columns:
        df['Aluminum_Return_1M'] = df['Aluminum_Price'].pct_change(1)
        df['Aluminum_Return_3M'] = df['Aluminum_Price'].pct_change(3)
        # Aluminum leads copper (same industrial demand)
        df['Aluminum_Copper_Mom_Spread'] = df['Aluminum_Return_3M'] - df['Copper_Return_3M']

    # --- Yield Curve / Macro Leading Indicators ---
    if 'US_10yr_Yield' in df.columns:
        df['Yield_Change_1M'] = df['US_10yr_Yield'].diff(1)
        df['Yield_Change_3M'] = df['US_10yr_Yield'].diff(3)
        # Rising yields = growth expectations = copper bullish
        df['Yield_Accel'] = df['Yield_Change_1M'] - df['Yield_Change_1M'].shift(1)
    
    if 'DXY' in df.columns:
        df['DXY_Return_1M'] = df['DXY'].pct_change(1)
        df['DXY_Return_3M'] = df['DXY'].pct_change(3)
        # Dollar weakening = copper strengthening (inverse relationship)
        df['DXY_Copper_Divergence'] = df['DXY_Return_3M'] + df['Copper_Return_3M']
    
    if 'China_Manuf_Prod' in df.columns:
        df['China_Manuf_Change_1M'] = df['China_Manuf_Prod'].pct_change(1)
        df['China_Manuf_Change_3M'] = df['China_Manuf_Prod'].pct_change(3)
        df['China_Manuf_Accel'] = df['China_Manuf_Change_1M'] - df['China_Manuf_Change_1M'].shift(1)

    if 'AUD_USD' in df.columns:
        df['AUD_USD_Return_1M'] = df['AUD_USD'].pct_change(1)
        df['AUD_USD_Return_3M'] = df['AUD_USD'].pct_change(3)
        # AUD is a commodity currency — its momentum often leads copper
        df['AUD_Copper_Mom_Spread'] = df['AUD_USD_Return_3M'] - df['Copper_Return_3M']

    # --- NEW: Technical Indicators ---
    # RSI-like Momentum (Ratio of Up vs Down moves over 6 months)
    delta = df['Copper_Price'].diff()
    gain = delta.clip(lower=0).rolling(window=6).mean()
    loss = (-delta.clip(upper=0)).rolling(window=6).mean()
    df['Copper_RSI_6M'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # MACD-like Signal (Short MA - Long MA, normalized)
    sma3 = df['Copper_Price'].rolling(window=3).mean()
    sma12 = df['Copper_Price'].rolling(window=12).mean()
    df['Copper_MACD'] = (sma3 - sma12) / sma12

    # Bollinger Band Position (Where is price relative to its band?)
    sma6 = df['Copper_Price'].rolling(window=6).mean()
    std6 = df['Copper_Price'].rolling(window=6).std()
    df['Copper_BB_Position'] = (df['Copper_Price'] - sma6) / std6.replace(0, np.nan)

    # Rate of Change Acceleration (Is momentum speeding up or slowing down?)
    df['Copper_Momentum_Accel'] = df['Copper_Return_1M'] - df['Copper_Return_1M'].shift(1)

    # --- Interaction Features ---
    # Combine key drivers to capture joint effects
    if 'AUD_USD' in df.columns and 'DXY' in df.columns:
        df['AUD_DXY_Interaction'] = df['AUD_USD'].pct_change() * df['DXY'].pct_change()
    
    if 'Oil_Price' in df.columns and 'China_Manuf_Prod' in df.columns:
        df['Oil_China_Interaction'] = df['Oil_Price'].pct_change() * df['China_Manuf_Prod'].pct_change()

    if 'SP500' in df.columns and 'US_10yr_Yield' in df.columns:
        df['SP500_Yield_Interaction'] = df['SP500'].pct_change() * df['US_10yr_Yield'].pct_change()

    # --- Rolling Statistics ---
    # Rolling mean of returns (trend strength)
    df['Copper_Return_Mean_6M'] = df['Copper_Return_1M'].rolling(window=6).mean()
    df['Copper_Return_Mean_12M'] = df['Copper_Return_1M'].rolling(window=12).mean()
    
    # Rolling skewness (asymmetry of returns - crash risk indicator)
    df['Copper_Skew_6M'] = df['Copper_Return_1M'].rolling(window=6).skew()

    # Year-over-Year change of key macros
    if 'US_Auto_Sales' in df.columns:
        df['US_Auto_Sales_YoY'] = df['US_Auto_Sales'].pct_change(12)
    if 'PPI_All_Commodities' in df.columns:
        df['PPI_YoY'] = df['PPI_All_Commodities'].pct_change(12)

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

    # 5. Volatility (Risk)
    # Rolling standard deviation of returns (3M and 6M)
    df['Copper_Vol_3M'] = df['Copper_Return_1M'].rolling(window=3).std()
    df['Copper_Vol_6M'] = df['Copper_Return_1M'].rolling(window=6).std()

    # 6. Trend (Moving Averages)
    # Price relative to SMA (Signal: >1 is Bullish trend, <1 is Bearish)
    df['Copper_SMA_3M'] = df['Copper_Price'] / df['Copper_Price'].rolling(window=3).mean()
    df['Copper_SMA_6M'] = df['Copper_Price'] / df['Copper_Price'].rolling(window=6).mean()

    # --- NORMALIZED LEVEL FEATURES (no-delay level awareness) ---
    # These give the model a sense of "where are we?" without raw price anchoring.
    
    # Percentile Rank: Where is today's price relative to its own 12-month history?
    # 1.0 = at 12-month high, 0.0 = at 12-month low
    df['Copper_Pctile_12M'] = df['Copper_Price'].rolling(window=12).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Z-Score: How many standard deviations away from 12-month mean?
    # +2 = very overbought, -2 = very oversold
    roll_mean_12 = df['Copper_Price'].rolling(window=12).mean()
    roll_std_12 = df['Copper_Price'].rolling(window=12).std()
    df['Copper_ZScore_12M'] = (df['Copper_Price'] - roll_mean_12) / roll_std_12.replace(0, np.nan)
    
    # Distance from 12-month high and low (normalized)
    roll_max_12 = df['Copper_Price'].rolling(window=12).max()
    roll_min_12 = df['Copper_Price'].rolling(window=12).min()
    df['Copper_Dist_From_High'] = (df['Copper_Price'] - roll_max_12) / roll_max_12
    df['Copper_Dist_From_Low'] = (df['Copper_Price'] - roll_min_12) / roll_min_12.replace(0, np.nan)
    
    # Same for key cross-assets (normalized, not raw)
    if 'Oil_Price' in df.columns:
        oil_mean12 = df['Oil_Price'].rolling(window=12).mean()
        oil_std12 = df['Oil_Price'].rolling(window=12).std()
        df['Oil_ZScore_12M'] = (df['Oil_Price'] - oil_mean12) / oil_std12.replace(0, np.nan)
    
    if 'DXY' in df.columns:
        dxy_mean12 = df['DXY'].rolling(window=12).mean()
        dxy_std12 = df['DXY'].rolling(window=12).std()
        df['DXY_ZScore_12M'] = (df['DXY'] - dxy_mean12) / dxy_std12.replace(0, np.nan)

    # Ratio changes (rate of change of ratios — captures shifting relationships)
    if 'Gold_Price' in df.columns:
        df['Copper_Gold_Ratio_Change'] = (df['Copper_Price'] / df['Gold_Price']).pct_change(3)
    if 'Aluminum_Price' in df.columns:
        df['Copper_Aluminum_Ratio_Change'] = (df['Copper_Price'] / df['Aluminum_Price']).pct_change(3)
    if 'Oil_Price' in df.columns:
        df['Copper_Oil_Ratio_Change'] = (df['Copper_Price'] / df['Oil_Price']).pct_change(3)

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
