import os
import time
import logging
import traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    explained_variance_score
)
from sklearn.exceptions import ConvergenceWarning
from data_process import load_market_data, calculate_technical_indicators

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_features(df):
    """Prepare features for model training using all technical indicators"""
    feature_columns = [
        # Price and Volume
        'spy_open', 'spy_high', 'spy_low', 'spy_close', 
        'spy_volume', 'vix_close', 'uup_close',
        
        # Moving Averages
        # 'EMA21', 'EMA50', 'SMA5', 'SMA10', 'SMA20', 'SMA50', 'SMA100', 'SMA150', 'SMA200',
        
        # Bollinger Bands
        'bb_percent_b', 'bb_bandwidth', 'bb_1d_percent_b',
        
        # Volatility
        'volatility', 'atr', 'atr_20',
        
        # Momentum Indicators
        'admf', 'roc', 'ultimate_rsi', 'ultimate_rsi_signal', 
        
        # Fisher Transform
        'fisher', 'fisher_trigger', 
        
        # Distance to MAs
        'dist_to_EMA21', 'dist_to_EMA50', 'dist_to_5D_SMA', 'dist_to_20D_SMA', 
        'dist_to_50D_SMA', 'dist_to_100D_SMA', 'dist_to_150D_SMA', 'dist_to_200D_SMA',
        
        # Slope
        '5D_Slope', 'EMA21_Slope', 'EMA50_Slope', '20D_Slope', '50D_Slope',
        
        # Donchian Channel
        'donchian_position', 'donchian_width',
        
        # Candle Features
        'candle_body', 'upper_wick', 'lower_wick', 'candle_relative_position', 
        # 'candle_direction',
        
        # Bounce Features
        # 'bounce_SMA5', 'bounce_SMA10', 'bounce_SMA20', 'bounce_SMA50', 'bounce_SMA100',
        # 'bounce_bb_1d_upper', 'bounce_bb_1d_lower',
        'bounce_strength_SMA5', 'bounce_strength_SMA10', 'bounce_strength_SMA20',
        'bounce_strength_SMA50', 'bounce_strength_SMA100',
        'bounce_strength_bb_1d_upper', 'bounce_strength_bb_1d_lower',
        
        # Touch Counts
        'touch_count_SMA5', 'touch_count_SMA10', 'touch_count_SMA20', 
        'touch_count_SMA50', 'touch_count_SMA100',
        'touch_count_bb_1d_upper', 'touch_count_bb_1d_lower',
        
        # Proximity Features
        'proximity_SMA5', 'proximity_SMA10', 'proximity_SMA20', 
        'proximity_SMA50', 'proximity_SMA100',
        'proximity_bb_1d_upper', 'proximity_bb_1d_lower',
        
        # Statistical Features
        'skewness_1d', 'skewness_5d',
        'z_score_1d', 'z_score_5d',
        'percentile_1d', 'percentile_5d',
        'entropy_1d', 'entropy_5d'
        
        # # Market State
        # 'market_state'
    ]

    # Create feature matrix
    X = df[feature_columns]
    
    # Create multi-output target
    # Next period's log return (original target)
    y_return = np.log(df['spy_close'].shift(-1) / df['spy_close'])
    
    # Create exit targets (1 = exit, 0 = hold)
    y_exit_long = ((df['spy_low'].shift(-1) <= df['stop_loss_long']) | 
                  (df['spy_high'].shift(-1) >= df['take_profit_long'])).astype(int)
    y_exit_short = ((df['spy_high'].shift(-1) >= df['stop_loss_short']) | 
                   (df['spy_low'].shift(-1) <= df['take_profit_short'])).astype(int)
    
    # Add market state as additional target
    y_market_state = pd.get_dummies(df['market_state'].shift(-1), prefix='market')
    
    # Combine into multi-output target
    y = pd.concat([
        pd.DataFrame({
            'return': y_return,
            'exit_long': y_exit_long,
            'exit_short': y_exit_short
        }),
        y_market_state
    ], axis=1)
    
    # Drop rows with missing values
    valid_idx = X.notna().all(axis=1) & y.notna().all(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Convert market_state to categorical codes
    if 'market_state' in X.columns:
        X['market_state'] = X['market_state'].astype('category').cat.codes
    
    scaler = StandardScaler()
    X_scaled_values = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_values, columns=feature_columns, index=X.index)
    
    return X_scaled, y, scaler, feature_columns, valid_idx

class MLStrategy:
    def __init__(self, model, scaler, feature_columns, min_hold_bars=3):
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.min_hold_bars = min_hold_bars
        self.last_signal_time = None
        self.current_position = 0
        
    def calculate_confidence_score(self, X_scaled):
        """Calculate confidence score based on prediction variance"""
        # Get predictions from LightGBM
        predictions = self.model.predict(X_scaled)
        
        # Calculate confidence using prediction magnitude
        confidence = np.abs(predictions) / np.max(np.abs(predictions))
        return confidence
        
    def enforce_hold_period(self, current_time, signals):
        """Enforce minimum hold period between signals"""
        if self.last_signal_time is None:
            # First signal, allow it
            self.last_signal_time = current_time
            return signals
            
        # Calculate bars since last signal
        bars_since_last = (current_time - self.last_signal_time).total_seconds() / (30 * 60)  # 30 min bars
        
        if bars_since_last < self.min_hold_bars:
            # Within hold period, maintain current position
            return self.current_position
        else:
            # Hold period expired, allow new signal
            self.last_signal_time = current_time
            self.current_position = signals
            return signals

def export_predictions(df, model_path='models/model.pkl', scaler_path='models/scaler.pkl', 
                      feature_columns_path='models/feature_columns.pkl'):
    """Export predictions using existing trained model"""
    try:
        # Validate model files exist
        for path in [model_path, scaler_path, feature_columns_path]:
            if not Path(path).exists():
                raise FileNotFoundError(f"Model artifact not found: {path}")
                
        # Load trained model and artifacts with validation
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            feature_columns = joblib.load(feature_columns_path)
        except Exception as e:
            raise ValueError(f"Error loading model artifacts: {str(e)}")
        
        # Prepare features
        X, y, _, _, valid_idx = prepare_features(df)
        
        # Get test set (last 10%)
        split_idx = int(len(X) * 0.9)
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        test_indices = df.index[valid_idx][split_idx:]
        
        # Generate predictions
        y_pred = model.predict(X_test)
        confidence_scores = MLStrategy(model, scaler, feature_columns).calculate_confidence_score(X_test)
        
        # Generate signals using only the return predictions
        min_confidence = 0.5
        return_predictions = y_pred
        return_threshold = np.percentile(return_predictions, 75)  # Use 75th percentile as threshold
        
        signals = pd.Series(
            np.where(
                (return_predictions > return_threshold) & (confidence_scores > min_confidence), 
                1, 
                np.where(
                    (return_predictions < -return_threshold) & (confidence_scores > min_confidence), 
                    -1, 
                    0
                )
            ),
            index=test_indices
        )
        
        # Save minimal prediction data
        test_results = pd.DataFrame({
            'timestamp': test_indices,
            'signal': signals,
            'confidence': confidence_scores,
            'predicted_return': y_pred,
            'actual_return': y_test.iloc[:, 0]
        }, index=test_indices)

        # Save features separately
        features_df = df[feature_columns].loc[test_indices]
        features_df.to_csv('data/test_features.csv', index=True)

        # Save to file
        file_path = Path('data/test_predictions.csv')
        test_results.to_csv(file_path, index=True, date_format='%Y-%m-%d %H:%M:%S%z')
        print(f"\nSuccessfully exported predictions to {file_path.absolute()}")

        return test_results

    except MemoryError:
        logger.error("Out of memory - reduce data size or use smaller model")
        raise
    except ConvergenceWarning:
        logger.warning("Model failed to converge - check learning rate")
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise
    except ValueError as e:
        logger.error(f"Data validation error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error exporting predictions: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def train_model(X_train, y_train):
    """Train and save the model with turnover penalty"""
    from sklearn.model_selection import RandomizedSearchCV
    # Extract primary target (return) for regression
    y_train = y_train.iloc[:, 0]
    
    # Custom scoring function to penalize frequent trading
    def position_aware_score(estimator, X, y):
        y_pred = estimator.predict(X)
        y_true = y
        
        # Calculate base MSE
        mse = mean_squared_error(y_true, y_pred)
        
        # Simulate position sizing based on trading frequency
        recent_trade_count = 0
        trade_decay = 0.9
        position_sizes = []
        
        for pred in y_pred:
            position_reduction = 1 / (1 + 0.2 * recent_trade_count)
            if abs(pred) > 0.001:
                position_sizes.append(position_reduction)
                recent_trade_count += 1
            else:
                position_sizes.append(0)
            
            # Decay recent trade count
            recent_trade_count *= trade_decay
        
        # Calculate average position size
        avg_position_size = np.mean(position_sizes) if position_sizes else 0
        
        # Reward larger average position sizes (less frequent trading)
        position_reward = avg_position_size
        
        # Combine metrics
        return -(mse - 0.2 * position_reward)  # Negative since we want to maximize
    
    # Base model - LightGBM
    base_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
    
    # Hyperparameter grid for LightGBM
    param_dist = {
        'num_leaves': [31, 63, 127],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 500],
        'max_depth': [5, 10, 20],
        'min_child_samples': [20, 50, 100],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0.0, 0.1, 0.5],
        'reg_lambda': [0.0, 0.1, 0.5]
    }
    
    # Randomized search with custom scoring
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring=position_aware_score,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    search.fit(X_train, y_train)
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV score: {-search.best_score_:.4f}")
    
    return search.best_estimator_
    
    # Fit the best model from search with progress tracking
    best_model = search.best_estimator_
    
    # Add progress tracking for RandomForest
    if hasattr(best_model.estimator, 'n_estimators'):
        from tqdm import tqdm
        n_trees = best_model.estimator.n_estimators
        start_time = time.time()
        
        # Use warm start to track progress
        best_model.estimator.set_params(warm_start=True)
        for i in tqdm(range(10, n_trees + 1, 10), desc="Training trees", unit="trees"):
            best_model.estimator.set_params(n_estimators=i)
            best_model.fit(X_train, y_train)
            
            # Calculate progress
            elapsed = time.time() - start_time
            trees_per_sec = i / elapsed
            remaining = (n_trees - i) / trees_per_sec
            
            # Update progress bar description
            tqdm.write(f"Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s")
        
        print()  # New line after progress
    else:
        # Fallback for non-tree models
        start_time = time.time()
        best_model.fit(X_train, y_train)
        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed:.1f} seconds")
    
    return best_model

if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--export-only', action='store_true',
                       help='Export predictions without retraining')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and only export feature importance')
    args = parser.parse_args()
    
    # Fetch and prepare data with progress tracking
    print("Loading market data...")
    start_load = time.time()
    df = load_market_data()
    load_time = time.time() - start_load
    print(f"Market data loaded in {load_time:.1f} seconds")
    
    print("Calculating technical indicators...")
    start_indicators = time.time()
    df = calculate_technical_indicators(df)
    indicators_time = time.time() - start_indicators
    print(f"Technical indicators calculated in {indicators_time:.1f} seconds")
    
    if args.export_only:
        print("Exporting predictions from existing model...")
        export_predictions(df)
        sys.exit(0)
        
    if args.skip_training:
        print("Skipping training, exporting feature importance only...")
        # Load existing model and artifacts
        model = joblib.load('data/model.pkl')
        scaler = joblib.load('data/scaler.pkl')
        feature_columns = joblib.load('data/feature_columns.pkl')
        
        # Prepare features
        X, y, _, _, valid_idx = prepare_features(df)
        
        # Calculate feature importance
        importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
        std = np.std([est.feature_importances_ for est in model.estimators_], axis=0)
        feature_importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importances,
            'Std': std
        }).sort_values('Importance', ascending=False)
        
        # Save feature importance
        feature_importance_df.to_csv('data/feature_importance.csv', index=False)
        print("Feature importance exported to data/feature_importance.csv")
        sys.exit(0)
        
    print("Training model...")
    start_time = datetime.now()
    X, y, scaler, feature_columns, valid_idx = prepare_features(df)
    
    # Split data by position - train on first 90%, test on last 10%
    split_idx = int(len(X) * 0.9)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Get test set date range and info
    test_dates = df.index[valid_idx][split_idx:]
    print(f"\nUsing 90/10 split")
    print(f"Test set date range: {test_dates[-1]} to {test_dates[0]}")  # Oldest to newest
    print(f"Test set size: {len(X_test)} samples (10% of total data)")
    print(f"Training set size: {len(X_train)} samples (90% of total data)")
    
    # Train and save model with progress tracking
    print("\nTraining model...")
    start_train = time.time()
    model = train_model(X_train, y_train)
    train_time = time.time() - start_train
    print(f"Model training completed in {train_time:.1f} seconds")
    
    print("Saving model artifacts...")
    start_save = time.time()
    joblib.dump(model, 'data/model.pkl')
    joblib.dump(scaler, 'data/scaler.pkl')
    joblib.dump(feature_columns, 'data/feature_columns.pkl')
    save_time = time.time() - start_save
    print(f"Model artifacts saved in {save_time:.1f} seconds")
    
    # Enhanced model evaluation
    y_pred = model.predict(X_test)
    
    # Calculate multiple metrics
    metrics = {
        'train_date': datetime.now().isoformat(),
        'mse': mean_squared_error(y_test.iloc[:, 0], y_pred),
        'mae': mean_absolute_error(y_test.iloc[:, 0], y_pred),
        'r2': r2_score(y_test.iloc[:, 0], y_pred),
        'explained_variance': explained_variance_score(y_test.iloc[:, 0], y_pred),
        'num_features': len(feature_columns),
        'test_set_size': len(X_test)
    }
    
    # Update performance history with validation
    performance_file = Path('data/performance_history.csv')
    try:
        if performance_file.exists():
            performance_history = pd.read_csv(performance_file)
        else:
            performance_history = pd.DataFrame(columns=metrics.keys())
            
        # Convert metrics to DataFrame and concatenate
        metrics_df = pd.DataFrame([metrics])
        performance_history = pd.concat([performance_history, metrics_df], ignore_index=True)
        
        # Validate before saving
        if not all(col in performance_history.columns for col in metrics.keys()):
            raise ValueError("Performance history columns mismatch")
            
        performance_history.to_csv(performance_file, index=False)
        
        # Verify file was written
        if not performance_file.exists():
            raise RuntimeError("Failed to save performance history")
            
        print(f"\nPerformance history updated at {performance_file}")
        
    except Exception as e:
        print(f"\nError updating performance history: {str(e)}")
        raise
    
    print(f"\nModel Evaluation on Test Set:")
    print(f"Mean Squared Error: {metrics['mse']:.4f}")
    print(f"R-squared: {metrics['r2']:.4f}")
    
    # Feature importance for LightGBM
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Keep all features
    print(f"\nUsing all {len(feature_columns)} features")
    
    # Enhanced feature importance visualization
    plt.figure(figsize=(12, 8))
    sorted_features = feature_importance_df.sort_values('Importance', ascending=True)
    
    # Create horizontal bar plot
    plt.barh(sorted_features['Feature'], 
             sorted_features['Importance'], 
             alpha=0.7)
    
    # Add labels and title
    plt.title('Feature Importance with Standard Deviation', fontsize=14)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add vertical line at median importance
    median_importance = np.median(sorted_features['Importance'])
    plt.axvline(median_importance, color='red', linestyle='--', alpha=0.5)
    plt.text(median_importance + 0.01, len(sorted_features)*0.9, 
             f'Median: {median_importance:.3f}', 
             color='red', fontsize=10)
    
    # Save high-quality version
    plt.tight_layout()
    plt.savefig('data/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Export feature importance to CSV
    feature_importance_df.to_csv('data/feature_importance.csv', index=False)
    
    # Print top 20 features with formatting
    print("\nTop 20 Feature Importances:")
    top_20 = feature_importance_df.nlargest(20, 'Importance')
    print(top_20[['Feature', 'Importance']].to_string(index=False))
    print("\nFull feature importance data saved to data/feature_importance.csv")
    
    # Update performance history
    performance_data = {
        'date': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')],
        'model_version': [model.__class__.__name__],
        'feature_count': [len(feature_columns)],
        'train_size': [len(X_train)],
        'test_size': [len(X_test)],
        'mse': [mean_squared_error(y_test.iloc[:, 0], model.predict(X_test))],
        'mae': [mean_absolute_error(y_test.iloc[:, 0], model.predict(X_test))],
        'top_feature_1': [feature_importance_df.iloc[0]['Feature']],
        'top_feature_1_importance': [feature_importance_df.iloc[0]['Importance']],
        'top_feature_2': [feature_importance_df.iloc[1]['Feature']],
        'top_feature_2_importance': [feature_importance_df.iloc[1]['Importance']],
        'top_feature_3': [feature_importance_df.iloc[2]['Feature']],
        'top_feature_3_importance': [feature_importance_df.iloc[2]['Importance']]
    }
    
    # Create or append to performance history
    performance_df = pd.DataFrame(performance_data)
    if os.path.exists('data/performance_history.csv'):
        history_df = pd.read_csv('data/performance_history.csv')
        performance_df = pd.concat([history_df, performance_df], ignore_index=True)
    
    performance_df.to_csv('data/performance_history.csv', index=False)
    print("\nPerformance metrics saved to data/performance_history.csv")
    
    # Create ML strategy instance
    strategy = MLStrategy(model, scaler, feature_columns)
    
    # Calculate confidence scores
    confidence_scores = strategy.calculate_confidence_score(X_test)
    
    # Generate signals with dynamic thresholds
    min_confidence = 0.5  # Lower confidence threshold
    
    # Use only the return predictions for signals
    return_predictions = y_pred
    return_threshold = np.percentile(return_predictions, 75)  # Use 75th percentile as threshold
    
    # Get test set indices from the position-based split
    test_indices = df.index[valid_idx][split_idx:]
    
    # Initialize strategy with 3 bar hold period
    strategy = MLStrategy(model, scaler, feature_columns, min_hold_bars=3)
    
    # Generate raw signals using only return predictions
    raw_signals = np.where(
        (return_predictions > return_threshold) & (confidence_scores > min_confidence), 
        1, 
        np.where(
            (return_predictions < -return_threshold) & (confidence_scores > min_confidence), 
            -1, 
            0
        )
    )
    
    # Apply hold period constraint
    signals = []
    for i, (time, raw_signal) in enumerate(zip(test_indices, raw_signals)):
        enforced_signal = strategy.enforce_hold_period(time, raw_signal)
        signals.append(enforced_signal)
    
    signals = pd.Series(signals, index=test_indices)
    
    # Save test predictions with OHLC data for backtesting
    test_results = df.loc[test_indices].copy()
    test_results['Signal'] = signals
    test_results['Confidence'] = confidence_scores
    test_results['Predicted_Return'] = y_pred
    
    # Ensure data directory exists
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Save predictions with validation
    file_path = data_dir / 'test_predictions.csv'
    try:
        # Save with explicit datetime formatting
        test_results.to_csv(file_path, index=True, date_format='%Y-%m-%d %H:%M:%S%z')
        
        # Verify file was written
        if not file_path.exists():
            raise RuntimeError(f"Failed to create {file_path}")
            
        # Verify file contents
        saved_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        if len(saved_data) != len(test_results):
            raise RuntimeError("Saved data length mismatch")
            
        print(f"\nSuccessfully saved test predictions to {file_path.absolute()}")
        print(f"Saved {len(saved_data)} rows with columns: {list(saved_data.columns)}")
        
    except Exception as e:
        print(f"\nError saving predictions: {str(e)}")
        if file_path.exists():
            file_path.unlink()  # Clean up partial file
        raise
    
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\nModel training pipeline complete!")
    print(f"Total time: {total_time:.1f} seconds")
    print("Breakdown:")
    print(f"- Data loading: {load_time:.1f}s ({load_time/total_time:.1%})")
    print(f"- Indicators: {indicators_time:.1f}s ({indicators_time/total_time:.1%})")
    print(f"- Model training: {train_time:.1f}s ({train_time/total_time:.1%})")
    print(f"- Model saving: {save_time:.1f}s ({save_time/total_time:.1%})")
    
    # Verify all outputs exist
    required_files = [
        'data/model.pkl',
        'data/scaler.pkl',
        'data/feature_columns.pkl',
        'data/test_predictions.csv',
        'data/feature_importance.png',
        'data/feature_importance.csv',
        'data/performance_history.csv'
    ]
    
    print("\nOutput verification:")
    missing_files = []
    for file in required_files:
        if Path(file).exists():
            print(f"- {file} exists")
        else:
            missing_files.append(file)
            print(f"- {file} MISSING")
    
    if missing_files:
        print("\nWarning: Some output files are missing!")
        print("Please check the logs for errors")
    else:
        print("\nAll output files created successfully")
    
    print("\nYou can now run backtest.py to evaluate performance")
