# Market Forecasting and Trading Strategy System

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![ML](https://img.shields.io/badge/machine%20learning-random%20forest-orange)

A comprehensive Python-based system for market data analysis, machine learning forecasting, and trading strategy development. This project demonstrates advanced skills in financial data engineering, machine learning, and quantitative analysis.

## Key Features

### Data Pipeline
- **Automated Data Collection**: Fetches historical market data from Interactive Brokers API
- **Multi-Asset Support**: SPY, VIX, UUP with configurable timeframes (1m to 1D)
- **Data Storage**: SQLite database for efficient data management

### Technical Analysis
- **Advanced Indicators**: 
  - Ultimate RSI with signal line
  - Fisher Transform
  - Accumulation/Distribution Money Flow (ADMF)
  - Bollinger Bands %B and Bandwidth
  - Volatility and ATR
- **Custom Calculations**: 
  - Distance to Moving Averages
  - 5D SMA Slope
  - Rate of Change (ROC)

### Machine Learning
- **Predictive Modeling**:
  - Random Forest Regressor with 300 estimators
  - Feature importance analysis
  - Confidence-based predictions
- **Model Evaluation**:
  - Backtesting framework with performance metrics
  - Position sizing based on prediction confidence
  - Win rate and Sharpe ratio tracking

### Visualization
- **Interactive Dashboards**:
  - 11-panel technical indicator view
  - Confidence score visualization
  - Backtest performance charts
- **Custom Plotly Charts**:
  - Price and position sizing overlay
  - Cumulative returns comparison
  - Feature importance plots

## Tech Stack

- **Core**: Python 3.8+
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, Joblib
- **Visualization**: Plotly, Matplotlib
- **Database**: SQLite
- **Broker Integration**: IBKR API (ib_insync)

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/market-forecast-tool.git
cd market-forecast-tool

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage Examples

### Data Collection
```bash
# Fetch historical data with various options
python scripts/fetch_ibkrdata.py -s SYMBOL -b BAR_SIZE [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]

# Example: Fetch 30-minute SPY data from 2013-01-01 to 2013-03-25
python scripts/fetch_ibkrdata.py -s SPY -b "30 mins" --start-date 2013-01-01 --end-date 2013-03-25

# Example: Fetch 1-hour VIX data for last 3 years
python scripts/fetch_ibkrdata.py -s VIX -b "1 hour"

# Arguments:
#   -s, --symbol       Stock symbol to fetch (e.g., SPY, VIX, UUP)
#   -b, --bar-size     Bar size for historical data (1 min, 5 mins, 15 mins, 30 mins, 1 hour, 1 day)
#   --start-date       Start date for historical data (YYYY-MM-DD)
#   --end-date         End date for historical data (YYYY-MM-DD). If not provided, collects up to most recent available data
```

### Model Training
```bash
# Train model with default parameters
python scripts/train_model.py
```

### Strategy Backtesting
```bash
# Run backtest with confidence-based position sizing
python scripts/forecast_price.py
```

### Technical Analysis
```bash
# View interactive technical indicators
python scripts/view_indicators.py
```

## Project Structure

```
market-forecast-tool/
├── data/                   # Persistent data storage
│   ├── marketdata.db       # Market data database
│   ├── model.pkl           # Trained ML model
│   ├── scaler.pkl          # Feature scaler
│   └── feature_columns.pkl # Feature names
│
├── scripts/                # Core application code
│   ├── fetch_ibkrdata.py   # Data collection
│   ├── train_model.py      # Model training
│   ├── forecast_price.py   # Prediction & backtesting
│   ├── view_indicators.py  # Technical analysis
│   ├── backtest.py         # Backtesting framework
│   └── data_processing.py  # Feature engineering
│
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Key Metrics Tracked

- **Model Performance**:
  - R² Score
  - Mean Squared Error
  - Feature Importance
- **Strategy Performance**:
  - Annualized Return
  - Sharpe Ratio
  - Max Drawdown
  - Win Rate
  - Average Trade Duration

## Configuration

Customize system behavior through these key parameters:

- **Data Collection**:
  - `DB_PATH` in `fetch_ibkrdata.py`
  - IBKR connection settings
  - Historical data range

- **Model Training**:
  - Random Forest parameters in `train_model.py`
  - Feature engineering settings in `data_processing.py`

- **Trading Strategy**:
  - Confidence thresholds
  - Position sizing rules
  - Risk management parameters

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Interactive Brokers for market data API
- Scikit-learn for machine learning framework
- Plotly for interactive visualizations
