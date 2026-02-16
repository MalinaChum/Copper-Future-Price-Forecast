import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

def load_processed_data(filepath):
    """Load processed feature data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath, index_col='Date', parse_dates=True)

def prepare_data(df, target_col='Target_Copper_Return_3M'):
    """Prepare data, using RETURN as target."""
    print(f"Data Shape: {df.shape}")
    
    # We need to filter out rows where target is NaN (for training)
    # But keep the ones where target is NaN (for inference)
    data_known = df.dropna(subset=[target_col])
    data_future = df[df[target_col].isna()]
    
    # Also check if features are clean (no NaNs allowed in models)
    # The models usually can't handle NaNs.
    # Note: features.py already dropped feature NaNs.
    
    print(f"Train/Test Data (Known Target): {data_known.shape}")
    print(f"Inference Data (Unknown Target): {data_future.shape}")
    
    return data_known, data_future

def train_rf_model(X_train, y_train):
    """Train Random Forest."""
    model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_and_plot(model, X_train, y_train, X_test, y_test, prices_test, current_prices_test, model_name):
    """
    Evaluate model on RETURNS, but also plot PRICES.
    prices_test: Actual future prices (Target_Copper_3M)
    current_prices_test: Prices at time t (Copper_Price)
    """
    # 1. Predict Returns
    # pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    # 2. Metrics (on Returns)
    rmse_return = np.sqrt(mean_squared_error(y_test, pred_test))
    r2_return = r2_score(y_test, pred_test)
    
    # 3. Convert to Price: P_t+3 = P_t * (1 + R_pred)
    # Note: If target was 3M return, then P_t+3 = P_t * (1 + R)
    pred_prices = current_prices_test * (1 + pred_test)
    
    # Metrics (on Prices)
    rmse_price = np.sqrt(mean_squared_error(prices_test, pred_prices))
    mae_price = mean_absolute_error(prices_test, pred_prices)
    
    print(f"--- {model_name} ---")
    print(f"Return RMSE: {rmse_return:.4f} (Target std: {y_test.std():.4f})")
    print(f"Return R2:   {r2_return:.4f}")
    print(f"Price RMSE:  ${rmse_price:.2f}")
    print(f"Price MAE:   ${mae_price:.2f}")
    
    return pred_prices

def plot_feature_importance(model, feature_names):
    """Plot feature importance for Random Forest."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importance (Random Forest)")
    plt.bar(range(len(feature_names)), importances[indices], align="center")
    plt.xticks(range(len(feature_names)), feature_names[indices], rotation=90)
    plt.xlim([-1, len(feature_names)])
    plt.tight_layout()
    
    output_path = 'outputs/figures/feature_importance.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Feature importance plot saved to {output_path}")
    
    # Print top 10 features
    print("\nTop 10 Important Features:")
    for f in range(min(10, len(feature_names))):
        print(f"{f+1}. {feature_names[indices[f]]} ({importances[indices[f]]:.4f})")

def main():
    input_path = 'data/processed/copper_features.csv'
    target_col = 'Target_Copper_Return_3M'
    price_target_col = 'Target_Copper_3M'
    
    print("Loading Data...")
    df = load_processed_data(input_path)
    
    # Split data
    data_known, data_future = prepare_data(df, target_col)
    drop_cols = [target_col, price_target_col]
  
    X = data_known.drop(columns=drop_cols)
    y = data_known[target_col]
    
    test_size = 0.15
    split_idx = int(len(X) * (1 - test_size)) # Time series split
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Keep prices for reconstruction
    prices_test = data_known[price_target_col].iloc[split_idx:]
    current_prices_test = data_known['Copper_Price'].iloc[split_idx:]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    print("Training Random Forest Model...")
    rf = train_rf_model(X_train, y_train)
    
    print("\nModel Evaluation:")
    rf_prices = evaluate_and_plot(rf, X_train, y_train, X_test, y_test, prices_test, current_prices_test, "Random Forest")
    
    # Feature Importance (New)
    plot_feature_importance(rf, X_train.columns)
    
    plt.figure(figsize=(12, 6))
    plt.plot(prices_test.index, prices_test, label='Actual Price', color='black', linewidth=2)
    plt.plot(prices_test.index, rf_prices, label='RF Forecast', linestyle='--', color='green')
    plt.title('Copper Price Forecast (3-Month Horizon)')
    plt.ylabel('Price ($/lb)')
    plt.legend()
    plt.grid(True)
    
    output_fig = 'outputs/figures/model_comparison.png'
    os.makedirs(os.path.dirname(output_fig), exist_ok=True)
    plt.savefig(output_fig)
    print(f"\nValidation plot saved to {output_fig}")
    
    # Future Inference
    if not data_future.empty:
        X_future = data_future.drop(columns=drop_cols)
        current_prices_future = data_future['Copper_Price']
        
        # Predict Returns
        future_ret_rf = rf.predict(X_future)
        
        # Convert to Price
        future_price_rf = current_prices_future * (1 + future_ret_rf)
        
        forecast = pd.DataFrame({
            'Current_Price': current_prices_future,
            'RF_Return': future_ret_rf,
            'RF_Price_Forecast': future_price_rf
        })
        
        print("\n--- Future Forecast (Next 3 Months) ---")
        print(forecast[['RF_Price_Forecast']])
        
        output_csv = 'outputs/reports/forecast_3m.csv'
        forecast.to_csv(output_csv)
        print(f"Forecast saved to {output_csv}")
    else:
        print("No future data available.")

if __name__ == "__main__":
    main()
