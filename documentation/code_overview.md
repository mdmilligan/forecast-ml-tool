# Code Overview

This document provides a high-level overview of the codebase, its main components, data flow, key classes/functions, and entry points.

## 1. Main Components/Modules and Their Purpose

The system is organized into several Python scripts, primarily located in the `scripts` directory, along with a main execution script.

*   **`main.py`**: Serves as the primary entry point for orchestrating the overall workflow, likely calling functions from other modules.
*   **`scripts/data_fetch.py`**: Responsible for fetching historical market data (e.g., from Interactive Brokers) and storing it, typically in the `data/marketdata.db` SQLite database.
*   **`scripts/data_process.py`**: Handles loading raw market data, cleaning it, calculating various technical indicators (e.g., EMA, RSI), and preparing it for model training and analysis.
*   **`scripts/model_train.py`**: Contains logic for training machine learning models. This includes feature preparation, model definition, training routines, and saving trained models (e.g., `model.pkl`, `scaler.pkl`, `feature_columns.pkl` in the `models/` or `data/` directory).
*   **`scripts/model_forecast.py`**: Uses trained models to generate predictions or forecasts on new data. Predictions are often saved (e.g., to `data/test_predictions.csv`).
*   **`scripts/strategy.py`**: Defines the trading strategy logic. The `MLStrategy` class, for instance, uses model predictions to generate trading signals (buy/sell/hold) and may include logic for confidence scores and holding periods.
*   **`scripts/backtest.py`**: Implements a custom backtesting engine (`Backtester` class) to simulate the trading strategy on historical data, calculate performance metrics (returns, Sharpe ratio, etc.), and plot results, potentially with an interactive dashboard.
*   **`scripts/backtest_bt.py`**: Provides an alternative backtesting implementation using the `backtesting` (bt) library. The `MLTradingStrategy` class integrates predictions from `data/test_predictions.csv`.
*   **`scripts/backtest_vectorbt.py`**: Offers another backtesting approach, using the `vectorbt` library for fast, vectorized backtesting.
*   **`scripts/visualization.py`**: (Mentioned as read-only) Contains functions for plotting data, such as technical indicators.

## 2. Data Flow

The typical data flow through the system is as follows:

1.  **Data Acquisition**: `scripts/data_fetch.py` retrieves raw market data (OHLCV) and saves it to `data/marketdata.db`.
2.  **Data Processing & Feature Engineering**: `scripts/data_process.py` loads data from the database, calculates technical indicators, and potentially other features. This processed data is then used for training and prediction.
3.  **Model Training**: `scripts/model_train.py` takes the processed data, splits it into training and testing sets, trains an ML model, and saves the model artifacts (e.g., `model.pkl`, `scaler.pkl`) to the `models/` or `data/` directory.
4.  **Prediction Generation**:
    *   `scripts/model_train.py` (via `export_predictions`) or `scripts/model_forecast.py` loads a trained model and uses it to generate predictions on a test set or new data.
    *   These predictions are often saved to a CSV file like `data/test_predictions.csv`.
5.  **Signal Generation**: `scripts/strategy.py` (specifically the `MLStrategy` class) takes the model's predictions and applies strategy logic (e.g., thresholds, holding periods) to generate trading signals.
6.  **Backtesting**:
    *   `scripts/backtest.py`, `scripts/backtest_bt.py`, or `scripts/backtest_vectorbt.py` use the historical market data and the generated trading signals (or predictions directly) to simulate trades.
    *   They calculate performance metrics and can generate reports or visualizations (e.g., `data/feature_importance.png`).

## 3. Key Classes and Functions

*   **Classes**:
    *   `scripts.backtest.Backtester`: Custom backtesting engine.
        *   `calculate_returns()`: Computes strategy and benchmark returns.
        *   `calculate_metrics()`: Calculates various performance metrics.
        *   `plot_results()`: Visualizes backtest results using Dash.
    *   `scripts.backtest_bt.MLTradingStrategy`: Strategy class for use with the `backtesting` (bt) library.
        *   `init()`: Initializes the strategy, loads predictions.
        *   `next()`: Defines logic for each trading step.
    *   `scripts.model_train.MLStrategy` / `scripts.strategy.MLStrategy`: (Note: `MLStrategy` appears in `model_train.py` for confidence calculation during export and in `strategy.py` for signal generation. The one in `strategy.py` is more comprehensive for ongoing signal generation).
        *   `__init__()`: Initializes with model, scaler, features.
        *   `calculate_confidence_score()`: Computes confidence for predictions.
        *   `generate_signals()` (in `scripts.strategy.MLStrategy`): Generates trading signals based on model output and other criteria.
        *   `enforce_hold_period()` (in `scripts.model_train.MLStrategy`): Enforces a minimum holding period for trades.

*   **Functions**:
    *   `main.py:main()`: Orchestrates the overall execution based on command-line arguments.
    *   `scripts.data_fetch.py:get_all_historical_data()`: Core function for fetching historical data from IBKR.
    *   `scripts.data_process.py:load_market_data()`: Loads data from the SQLite database.
    *   `scripts.data_process.py:calculate_technical_indicators()`: Central function for adding a comprehensive set of TA indicators to the data.
    *   `scripts.model_train.py:train_model()`: Trains the LightGBM model using `RandomizedSearchCV`.
    *   `scripts.model_train.py:export_predictions()`: Generates and saves predictions from a trained model, primarily for the test set.
    *   `scripts.model_forecast.py:generate_predictions()`: Generates predictions for a specified date range using a trained model.
    *   `scripts.backtest*.py:run_backtest()`: Main functions in each backtesting script to execute a backtest and display results.
    *   `scripts.backtest.py:load_and_backtest()`: Helper to load predictions and run the custom backtest.

## 4. Entry Points

How to run and use the tool:

*   **Main Workflow**: The primary way to run the entire pipeline (or selected parts of it) is through `main.py`.
    ```bash
    python main.py --train --backtest --predict --start-date YYYY-MM-DD --end-date YYYY-MM-DD
    ```
    Use `--help` to see all available options.

*   **Individual Scripts**: Specific tasks can often be run by executing individual scripts directly:
    *   **Fetch Data**:
        ```bash
        python scripts/data_fetch.py --symbol SPY --start-date YYYY-MM-DD --end-date YYYY-MM-DD --bar-size "30 mins"
        ```
    *   **Train Model**: (This script also handles data loading and processing internally if not run via `main.py`)
        ```bash
        python scripts/model_train.py
        ```
        (Can also use `--export-only` or `--skip-training` flags)
    *   **Run Backtest**: (After model training and prediction generation)
        ```bash
        python scripts/backtest.py
        # or
        python scripts/backtest_bt.py
        # or
        python scripts/backtest_vectorbt.py
        ```
    *   **Generate Forecasts**: (This script also handles data loading and processing)
        ```bash
        python scripts/model_forecast.py
        ```
        (The script currently uses hardcoded dates for `start_date` and `end_date` in its `if __name__ == "__main__":` block, which might need adjustment or parameterization for flexible use).

The `requirements.txt` file lists necessary dependencies that should be installed in the Python environment (e.g., using `pip install -r requirements.txt`).
The `data/` directory stores input data (`marketdata.db`, `nyse-holidays.csv`), intermediate results (like `test_predictions.csv`, `feature_importance.csv`), and outputs (`feature_importance.png`).
The `models/` directory stores serialized machine learning models (`model.pkl`, `scaler.pkl`, `feature_columns.pkl`).
