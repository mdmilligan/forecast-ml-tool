# Model Training Documentation

This guide provides a practical overview of how to train, evaluate, and use the machine learning models in this project.

## 1. What Data the Model Uses

The model is trained on historical market data and a rich set of derived features.

**Input Data:**
The training process starts by loading data from the `data/marketdata.db` SQLite database. Specifically, it uses 30-minute bar data for:
*   **SPY**: The primary instrument for price action (Open, High, Low, Close, Volume).
*   **VIX**: The CBOE Volatility Index, used as a feature.
*   **UUP**: The Invesco DB US Dollar Index Bullish Fund, used as a feature.

**Features (Inputs):**
The raw data is processed by `scripts/data_process.py` to generate a wide array of technical indicators and statistical features. These include:
*   **Price and Volume:** Raw OHLCV for SPY, and closing prices for VIX and UUP.
*   **Moving Averages & Slopes:** Various EMAs and SMAs, their slopes, and the price's distance to them.
*   **Volatility Indicators:** Bollinger Bands (`%B`, bandwidth), ATR, and historical volatility.
*   **Momentum Indicators:** ADMF, Rate of Change (ROC), and a custom Ultimate RSI.
*   **Oscillators:** Fisher Transform.
*   **Candle & Pattern Features:** Wick sizes, body size, and bounce detection off key levels (SMAs, Bollinger Bands).
*   **Statistical Features:** Autocorrelation, skewness, z-scores, and entropy over different time windows.
*   **Market State:** A derived feature that classifies the market as 'trending', 'choppy', 'transitioning', or 'uncertain'.

**Targets (Outputs):**
The model in `scripts/model_train.py` is trained to predict multiple outputs simultaneously, although the current implementation focuses on the primary return target for training:
1.  **Return:** The next 30-minute period's logarithmic return of SPY's closing price. This is the primary regression target.
2.  **Exit Long:** A binary flag indicating if a long position should be exited (e.g., stop-loss or take-profit hit).
3.  **Exit Short:** A binary flag indicating if a short position should be exited.
4.  **Market State:** The market state in the next period.

## 2. How to Run the Training Process



You can train the model using the main script or by running the training script directly.

                                                                                                                                                
                                                                                                                                                
**Recommended Method (via `main.py`):**                                                                                                         
                                                                                                                                                
Use the main entry point to ensure the entire data pipeline is run consistently. Train the model using data from the specified date range

python main.py --train --start-date 2013-01-01 --end-date 2024-12-31


**Direct Script Execution:**

If you only want to run the training part, you can execute `scripts/model_train.py`. This script handles its own data loading and processing.   


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Run the entire training and evaluation pipeline                                                                                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

python scripts/model_train.py

                                                                                                                                                
                                                                                                                                                
                                                                                                                                                
The script splits the data, using the first 90% for training and the final 10% for testing.                                                     
                                                                                                                                                
                                                                                                                                                
                                                                                                                                                
## 3. Key Training Parameters



Parameters can be adjusted in two main places: feature calculation and model training.

                                                                                                                                                
                                                                                                                                                
**Feature Parameters (`scripts/data_process.py`):**

Inside the `calculate_technical_indicators` function, a `params` dictionary controls how features are generated. Key parameters include:        

*   `admf_length`: Lookback period for the ADMF momentum indicator.

*   `roc_period`: Lookback for the Rate of Change indicator.                                                                                    
                                                                                                                                                
*   `fisher_length`: Lookback for the Fisher Transform oscillator.                                                                              
                                                                                                                                                
*   `lookback`: Period for calculating the slope of moving averages.



**Model Hyperparameters (`scripts/model_train.py`):**

The `train_model` function uses `RandomizedSearchCV` to find the best hyperparameters for the `LightGBM` model. The search space is defined in  
`param_dist`. Key hyperparameters include:

*   `n_estimators`: The number of boosting rounds (trees) to build.

*   `learning_rate`: Controls the step size at each iteration. Lower values require more estimators.

*   `num_leaves`: The maximum number of leaves in one tree. A key parameter for controlling model complexity.                                   
                                                                                                                                                
*   `max_depth`: The maximum depth of a tree.



The training process also uses a custom scoring function, `position_aware_score`, which penalizes the model for generating signals that would   
lead to frequent trading (high turnover).
                                                                                                                                                
                                                                                                                                                
                                                                                                                                                
## 4. How to Tell if Training is Working Well                                                                                                   
                                                                                                                                                
                                                                                                                                                
                                                                                                                                                
The script provides several outputs to help you assess training performance.                                                                    
                                                                                                                                                


**During Training:**

*   **Console Output:** The script will print progress for data loading, indicator calculation, and model training.

*   **Best CV Score:** After the hyperparameter search, you will see a "Best CV score". This score is the result of the `position_aware_score`  
(negative MSE penalized by turnover). A score closer to zero is better.

                                                                                                                                                
                                                                                                                                                
**After Training (Evaluation on Test Set):**

*   **Model Evaluation Metrics:** The script prints `Mean Squared Error` and `R-squared` for the model's predictions on the unseen 10% test set.

    *   **MSE:** Lower is better.

    *   **R-squared:** Closer to 1.0 is better. Positive values indicate the model is better than a simple mean-based prediction.

*   **Feature Importance:**

    *   A plot is saved to `data/feature_importance.png`. This helps you see which features the model found most useful.

    *   A CSV with detailed importance scores is saved to `data/feature_importance.csv`.

*   **Performance History:**

    *   Key metrics from the training run are appended to `data/performance_history.csv`. This is the best way to track model performance over  
time as you make changes.
                                                                                                                                                
                                                                                                                                                
                                                                                                                                                
## 5. Where Trained Models Are Saved                                                                                                            
                                                                                                                                                
                                                                                                                                                
                                                                                                                                                
After a successful training run, the following essential files (model artifacts) are saved to the `data/` directory:



*   `data/model.pkl`: The serialized, trained LightGBM model object.

*   `data/scaler.pkl`: The `StandardScaler` object used to normalize the features. This is required to process new data in the same way.        

*   `data/feature_columns.pkl`: The list of feature names the model was trained on. This ensures that the same features are used for future     
predictions.
                                                                                                                                                
                                                                                                                                                
                                                                                                                                                
## 6. How to Use a Newly Trained Model                                                                                                          
                                                                                                                                                
                                                                                                                                                
                                                                                                                                                
The training script is designed to flow directly into the next steps of the workflow.                                                           



1.  **Automatic Prediction Generation:** After saving the model, the script automatically uses it to generate predictions on the test set. These
predictions, along with signals and confidence scores, are saved to `data/test_predictions.csv`.



2.  **Run a Backtest:** With new predictions available, the immediate next step is to run a backtest to evaluate the strategy's trading
performance. You can do this using one of the backtesting scripts.

                                                                                                                                                
                                                                                                                                                
    ```bash                                                                                                                                     
                                                                                                                                                
    # Run the custom backtest with an interactive dashboard                                                                                     
                                                                                                                                                
    python scripts/backtest.py                                                                                                                  
                                                                                                                                                
                                                                                                                                                
                                                                                                                                                
    # Or run the backtest via the main script                                                                                                   
                                                                                                                                                
    python main.py --backtest                                                                                                                   
                                                                                                                                                
    ```



3.  **Exporting Predictions Separately:** If you want to re-generate predictions using an already-trained model without re-training, use the    
`--export-only` flag:
                                                                                                                                                
    ```bash                                                                                                                                     
                                                                                                                                                
    python scripts/model_train.py --export-only                                                                                                 
                                                                                                                                                
    ```      