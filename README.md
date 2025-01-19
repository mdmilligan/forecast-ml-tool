# Market Forecasting Tool

A Python-based tool for fetching market data, calculating technical indicators, training machine learning models, and generating price forecasts.

## Features

- **Data Fetching**: Pulls historical market data from Interactive Brokers
  - Supports multiple symbols (SPY, VIX, UUP)
  - Configurable timeframes (1m, 5m, 15m, 30m, 1h, 1D)
  - Customizable date ranges
- **Technical Analysis**:
  - Moving Averages (SMA 20, SMA 50)
  - RSI
  - MACD
  - Bollinger Bands
  - Volatility
  - ADMF (Accumulation/Distribution Money Flow)
- **Machine Learning**:
  - Random Forest Regressor model
  - Feature scaling and preprocessing
  - Confidence scoring for predictions
- **Visualization**:
  - Interactive Plotly charts
  - Multiple indicator subplots
  - Recent 100-day view
- **Forecasting**:
  - Next-day price predictions
  - Confidence-based position sizing
  - Signal generation

## Requirements

Python 3.8+ with the following packages:
- pandas
- numpy
- scikit-learn
- plotly
- ib_insync
- joblib
- sqlite3

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/market-forecast-tool.git
   cd market-forecast-tool
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Interactive Brokers connection:
   - Ensure IB Gateway or TWS is running
   - Configure connection settings in `scripts/fetch_ibkrdata.py`

## Usage

### Fetching Data
```bash
python scripts/fetch_ibkrdata.py -s SPY -b "5 mins" -m 36
```

### Training Model
```bash
python scripts/train_model.py
```

### Generating Forecasts
```bash
python scripts/forecast_price.py
```

### Viewing Indicators
```bash
python scripts/view_indicators.py
```

## File Structure

```
market-forecast-tool/
├── data/
│   ├── marketdata.db          # SQLite database for market data
│   ├── model.pkl              # Trained model
│   ├── scaler.pkl             # Feature scaler
│   └── feature_columns.pkl    # Feature column names
├── scripts/
│   ├── fetch_ibkrdata.py      # Data fetching from IBKR
│   ├── train_model.py         # Model training
│   ├── forecast_price.py      # Price forecasting
│   ├── view_indicators.py     # Indicator visualization
│   └── indicator_admf.py      # ADMF indicator calculation
└── README.md                  # This file
```

## Configuration

Key configuration options:

- Database path: `DB_PATH` in `fetch_ibkrdata.py`
- IBKR connection: `ib.connect()` parameters in `fetch_ibkrdata.py`
- Model parameters: `RandomForestRegressor` in `train_model.py`
- Indicator parameters: Various functions in `train_model.py` and `indicator_admf.py`