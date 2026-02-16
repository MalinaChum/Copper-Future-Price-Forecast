# Copper Price Forecast

This project uses machine learning (Random Forest) to forecast future copper prices based on macroeconomic indicators.

## Features:
- **Data Ingestion**: Fetches historical Copper, Gold, Oil, DXY, and macro data from Yahoo Finance & FRED.
- **Feature Engineering**: Calculates rolling volatility, momentum, and cross-asset ratios.
- **Model**: Random Forest Regressor targeting 3-month returns.
- **Evaluation**: Backtested on 2024 data to ensure robustness.

## Setup:
1. Clone the repository:
   ```bash
   git clone https://github.com/MalinaChum/Copper-Future-Price-Forecast.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the pipeline:
   ```bash
   python src/data_loader.py  # Fetch data
   python src/features.py     # Process features
   python src/models.py       # Train & Forecast
   ```

## Results:
- Includes detailed backtest plots for 2024.
- Feature importance analysis (e.g., AUD/USD impact).
