# Market Forecast Tool
A Python system for market data ingestion, technical feature engineering, ML forecasting and backtesting.

## Project name and one-line description
Market Forecast Tool — end-to-end pipeline to fetch market data, create features, train forecasting models and backtest trading strategies.

## Prerequisites
- Python 3.8+
- SQLite available (used by scripts/data_fetch.py)
- Optional: Interactive Brokers Gateway / TWS if using live data (scripts/data_fetch.py uses ib_insync)
- A web browser for Dash backtest dashboard

## Install
1. Clone the repository:
   - git clone <repo-url>
   - cd MarketForecastTool
2. Create and activate a virtual environment:
   - python -m venv venv
   - Windows: venv\Scripts\activate
   - macOS/Linux: source venv/bin/activate
3. Install dependencies:
   - pip install -r requirements.txt

## Installation notes
- The dependencies are listed in [`requirements.txt`](requirements.txt:1).
- Heavy packages: lightgbm, ib_insync, dash; install with system package managers if wheel builds fail.

## Entrypoints and scripts
- Top-level orchestrator: [`main.py`](main.py:1)
- Core scripts (in repository):
  - [`scripts/data_fetch.py`](scripts/data_fetch.py:1) — fetch historical data from IBKR and write to `data/marketdata.db`
  - [`scripts/model_train.py`](scripts/model_train.py:1) — full training pipeline and artifact export
  - [`scripts/model_forecast.py`](scripts/model_forecast.py:1) — generate predictions using saved artifacts
  - [`scripts/backtest.py`](scripts/backtest.py:1) — run backtest and launch Dash dashboard
  - [`scripts/config.py`](scripts/config.py:1) — runtime configuration values
  - [`scripts/visualization.py`](scripts/visualization.py:1) — helper plotting/visualization utilities

## How to run (Usage)
All commands run from repository root.

### 1) Fetch historical market data (Interactive Brokers)
Example (30-minute bars for SPY):
   python scripts/data_fetch.py --symbol SPY --bar-size "30 mins" --start-date 2019-01-01 --end-date 2021-01-01
Notes:
- Output: writes to SQLite database `data/marketdata.db` and returns a DataFrame when run as module.
- Requires IB Gateway/TWS running and accessible on ports 7497 / 4001 / 4002.

### 2) Train the model (end-to-end)
   python scripts/model_train.py
What this does:
- Loads market data via `load_market_data()` and computes indicators
- Prepares features and trains a LightGBM regressor
- Saves artifacts to `models/` and predictions to `data/test_predictions.csv`
Optional flags:
- `--export-only` — export predictions using existing artifacts without retraining
- `--skip-training` — skip training and export feature importance (requires existing model artifacts)

### 3) Generate forecasts (use existing artifacts)
   python scripts/model_forecast.py
Output:
- `data/test_predictions.csv`
- `data/test_features.csv` (if feature_columns present)

### 4) Backtest and dashboard
- Run backtest (reads `data/test_predictions.csv` and `data/marketdata.db`):
   python scripts/backtest.py
- Orchestrator option:
   python main.py --backtest
Expected:
- Console prints summary metrics and a Dash dashboard launches at http://127.0.0.1:8050/
- Stop server with Ctrl+C

### 5) Orchestrator CLI
The top-level [`main.py`](main.py:1) supports:
- --train
- --backtest
- --predict
Note: some internal function signatures differ between `main.py` and `scripts/*` modules; prefer running the scripts directly if `main.py` produces errors.

## Important file locations
- data/: persistent data and outputs (e.g., `data/marketdata.db`, `data/test_predictions.csv`)
- models/: saved model artifacts (`models/model.pkl`, `models/scaler.pkl`, `models/feature_columns.pkl`)
- documentation/: design notes and guides (`documentation/code_overview.md`, `documentation/training.md`)

## Example minimal flows
1) Quick backtest (data already present)
   python scripts/backtest.py

2) End-to-end example (requires IBKR)
   python scripts/data_fetch.py --symbol SPY --bar-size "30 mins" --start-date 2019-01-01 --end-date 2021-01-01
   python scripts/model_train.py
   python scripts/backtest.py

3) Generate predictions only (model artifacts already present)
   python scripts/model_forecast.py

## Troubleshooting & common issues
- "Model files not found": ensure `models/model.pkl`, `models/scaler.pkl`, `models/feature_columns.pkl` exist. Create them with `python scripts/model_train.py`.
- IBKR connection failures: ensure IB Gateway/TWS is running and API connections are enabled; confirm port (7497/4001/4002) and firewall rules.
- Timezone or index mismatches: scripts attempt to align timezone-aware indices; check timestamps in `data/test_predictions.csv` and `data/marketdata.db` if alignment errors occur.
- Dashboard not opening: open http://127.0.0.1:8050/ manually.
- Missing tests or license: no `tests/` folder and no `LICENSE` file detected.

## Tests
- No automated tests present. To add tests:
  - Create a `tests/` directory with pytest tests
  - Add `pytest` to `requirements.txt`
  - Run `pytest`

## Configuration
- Edit [`scripts/config.py`](scripts/config.py:1) for defaults:
  - `MIN_CONFIDENCE`, `RETURN_THRESHOLD_PERCENTILE`, `MIN_HOLD_BARS`
  - `START_DATE`, `END_DATE`
  - `TEST_SPLIT_RATIO`, `INITIAL_CAPITAL`, `COMMISSION`

## Notes on mismatches found
- README referenced scripts with different names than present. Use the actual script filenames in `scripts/`.
- Model artifacts are saved to `models/`, not `data/` (README previously said otherwise).
- `main.py` calls internal functions with signatures that may not match script definitions (e.g., `generate_predictions`). Use script runners for reliability.
- No `tests/` or `LICENSE` present in root (README previously referenced them).
- Total mismatches/outdated items found: 6

## License
- No license file detected. Add one (e.g., MIT) at repository root as `LICENSE`.

## Where to look next
- [`documentation/code_overview.md`](documentation/code_overview.md:1)
- [`documentation/training.md`](documentation/training.md:1)
- [`documentation/backtesting_roadmap.md`](documentation/backtesting_roadmap.md:1)

--- 
End of README
