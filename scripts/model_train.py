import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from data_process import load_market_data, calculate_technical_indicators

def prepare_features(df):
    """Prepare features for model training using all technical indicators"""
    feature_columns = [
        # Price and Volume
        'spy_close', 'spy_volume', 'vix_close', 'uup_close',
        
        # Moving Averages
        'EMA21', 'EMA50', 'SMA20', 'SMA50',
        
        # Bollinger Bands
        'bb_percent_b', 'bb_bandwidth',
        
        # Volatility
        'volatility', 'atr',
        
        # Momentum Indicators
        'admf', 'roc', 
        
        # Fisher Transform
        'fisher', 'fisher_trigger',
        'fisher_cross_above', 'fisher_cross_below',
        
        # Distance to MAs
        'dist_to_EMA21', 'dist_to_EMA50', 'dist_to_5day_SMA',
        
        # Slope
        'slope',
        
        # Ultimate RSI
        'ultimate_rsi', 'ultimate_rsi_signal'
    ]
    
    # Add crossover features
    df['fisher_cross_above'] = ((df['fisher'] > df['fisher_trigger']) & 
                               (df['fisher'].shift() <= df['fisher_trigger'].shift())).astype(int)
    df['fisher_cross_below'] = ((df['fisher'] < df['fisher_trigger']) & 
                               (df['fisher'].shift() >= df['fisher_trigger'].shift())).astype(int)
    df['ursi_cross_above'] = ((df['ultimate_rsi'] > df['ultimate_rsi_signal']) & 
                             (df['ultimate_rsi'].shift() <= df['ultimate_rsi_signal'].shift())).astype(int)
    df['ursi_cross_below'] = ((df['ultimate_rsi'] < df['ultimate_rsi_signal']) & 
                             (df['ultimate_rsi'].shift() >= df['ultimate_rsi_signal'].shift())).astype(int)

    # Add crossover features to feature columns
    feature_columns.extend(['fisher_cross_above', 'fisher_cross_below', 
                          'ursi_cross_above', 'ursi_cross_below'])

    # Create feature matrix and target
    X = df[feature_columns]
    # Use log returns as target for better numerical stability
    y = np.log(df['spy_close'].shift(-1) / df['spy_close'])  # Next period's log return
    
    # Drop rows with missing values (from indicator calculations)
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_columns, valid_idx

class MLStrategy:
    def __init__(self, model, scaler, feature_columns):
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns
        
    def calculate_confidence_score(self, X_scaled):
        """Calculate confidence score based on prediction variance"""
        # Get predictions from all trees
        predictions = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])
        
        # Calculate confidence as inverse of standard deviation
        std_dev = np.std(predictions, axis=0)
        confidence = 1 / (1 + std_dev)
        return confidence

def train_model(X_train, y_train):
    """Train and save the model with improved parameters"""
    from sklearn.model_selection import RandomizedSearchCV
    
    # Base model
    model = RandomForestRegressor(random_state=42, n_jobs=-1, verbose=1)
    
    # Hyperparameter grid
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 0.5, 0.8],
        'max_samples': [0.6, 0.8, 1.0]
    }
    
    # Randomized search with 3-fold CV
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV score: {-search.best_score_:.4f}")
    
    return search.best_estimator_
    
    # Fit model with early stopping if possible
    try:
        model.set_params(warm_start=True)
        for i in range(10, 201, 10):
            model.set_params(n_estimators=i)
            model.fit(X_train, y_train)
            print(f"Trained with {i} estimators")
    except:
        # Fallback to normal training if early stopping not supported
        model.set_params(warm_start=False, n_estimators=200)
        model.fit(X_train, y_train)
    
    return model

if __name__ == "__main__":
    print("Training model...")
    start_time = datetime.now()
    
    # Fetch and prepare data
    df = load_market_data()
    df = calculate_technical_indicators(df)
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
    
    # Train and save model
    model = train_model(X_train, y_train)
    joblib.dump(model, 'data/model.pkl')
    joblib.dump(scaler, 'data/scaler.pkl')
    joblib.dump(feature_columns, 'data/feature_columns.pkl')
    
    # Evaluate model on test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Evaluation on Test Set:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")
    
    # Feature selection based on importance
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances,
        'Std': std
    }).sort_values('Importance', ascending=False)
    
    # Select top 50% of features
    threshold = np.median(feature_importance_df['Importance'])
    selected_features = feature_importance_df[feature_importance_df['Importance'] > threshold]['Feature'].tolist()
    print(f"\nSelected {len(selected_features)} features out of {len(feature_columns)}")
    print(f"Selected features: {selected_features}")
    
    # Update feature columns
    feature_columns = selected_features
    
    # Print all feature importances with formatting
    print("\nAll Feature Importances:")
    print(feature_importance_df.sort_values('Importance', ascending=False).to_string())
    
    # Create ML strategy instance
    strategy = MLStrategy(model, scaler, feature_columns)
    
    # Calculate confidence scores
    confidence_scores = strategy.calculate_confidence_score(X_test)
    
    # Generate signals with dynamic thresholds
    min_confidence = 0.5  # Lower confidence threshold
    return_threshold = np.percentile(y_pred, 75)  # Use 75th percentile as threshold
    
    # Get test set indices from the position-based split
    test_indices = df.index[valid_idx][split_idx:]
    
    signals = pd.Series(
        np.where(
            (y_pred > return_threshold) & (confidence_scores > min_confidence), 
            1, 
            np.where(
                (y_pred < -return_threshold) & (confidence_scores > min_confidence), 
                -1, 
                0
            )
        ),
        index=test_indices
    )
    
    # Initialize backtest metrics
    backtest_metrics = {
        'status': 'no_signals',
        'message': 'No valid signals generated for backtesting'
    }
    
    # Run backtest only if we have valid signals
    if signals.abs().sum() > 0:
        from .backtest import run_backtest
        test_df = df.loc[test_indices]
        backtest_metrics = run_backtest(test_df, signals, confidence_scores)
        
        # Save test predictions
        test_results = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Signal': signals
        }, index=test_indices)
        
        try:
            import os
            from pathlib import Path
            
            # Create data directory if it doesn't exist
            data_dir = Path('data')
            data_dir.mkdir(exist_ok=True)
            
            # Save with explicit path and error handling
            file_path = data_dir / 'test_predictions.csv'
            test_results.to_csv(file_path)
            print(f"\nSuccessfully saved test predictions to {file_path.absolute()}")
            
        except Exception as e:
            print(f"\nError saving predictions: {str(e)}")
            import traceback
            traceback.print_exc()
            backtest_metrics['status'] = 'error'
            backtest_metrics['message'] = f'Could not save predictions: {str(e)}'
    else:
        print("Warning: No valid signals generated for backtesting")
    
    print("\nBacktest Metrics:")
    print(backtest_metrics)
    print("\nSaved test predictions to data/test_predictions.csv")
    
    print(f"\nModel training complete! Time taken: {datetime.now() - start_time}")
