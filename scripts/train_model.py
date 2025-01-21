import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from data_processing import load_market_data, calculate_technical_indicators

def prepare_features(df):
    """Prepare features for model training using all technical indicators"""
    feature_columns = [
        # Price and Volume
        'spy_close', 'spy_volume', 'vix_close', 'uup_close',
        
        # Moving Averages
        'EMA21', 'EMA50', 'SMA20', 'SMA50_daily',
        
        # Bollinger Bands
        'bb_percent_b', 'bb_bandwidth',
        
        # Volatility
        'volatility', 'atr',
        
        # Momentum Indicators
        'admf', 'roc', 
        
        # Fisher Transform
        'fisher', 'fisher_trigger',
        
        # Distance to MAs
        'dist_to_EMA21', 'dist_to_EMA50', 'dist_to_50day_SMA',
        
        # Slope
        'slope',
        
        # Ultimate RSI
        'ultimate_rsi', 'ultimate_rsi_signal'
    ]
    
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
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features=0.5,  # Use half of features for each split
        random_state=42,
        n_jobs=-1,
        verbose=1,
        max_samples=0.8  # Use bootstrap sampling
    )
    
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
    
    # Train/test split
    train_size = int(len(df) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
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
    
    # Enhanced feature importance analysis
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances,
        'Std': std
    }).sort_values('Importance', ascending=False)
    
    # Print all feature importances with formatting
    print("\nAll Feature Importances:")
    print(feature_importance_df.sort_values('Importance', ascending=False).to_string())
    
    # Create ML strategy instance
    strategy = MLStrategy(model, scaler, feature_columns)
    
    # Calculate confidence scores
    confidence_scores = strategy.calculate_confidence_score(X_test)
    
    # Generate signals with confidence threshold
    min_confidence = 0.7  # Minimum confidence threshold
    signals = pd.Series(
        np.where(
            (y_pred > 0.0025) & (confidence_scores > min_confidence), 
            1, 
            np.where(
                (y_pred < -0.0025) & (confidence_scores > min_confidence), 
                -1, 
                0
            )
        ),
        index=df.index[train_size:train_size+len(y_test)]
    )
    
    # Initialize backtest metrics
    backtest_metrics = {
        'status': 'no_signals',
        'message': 'No valid signals generated for backtesting'
    }
    
    # Run backtest only if we have valid signals
    if signals.abs().sum() > 0:
        from backtest import run_backtest
        test_df = df.iloc[train_size:train_size+len(y_test)]
        backtest_metrics = run_backtest(test_df, signals, confidence_scores)
        
        # Save test predictions
        test_results = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Signal': signals
        }, index=df.index[train_size:train_size+len(y_test)])
        
        try:
            test_results.to_csv('data/test_predictions.csv')
        except PermissionError:
            print("Warning: Could not save predictions due to permission error")
            backtest_metrics['status'] = 'error'
            backtest_metrics['message'] = 'Could not save predictions'
    else:
        print("Warning: No valid signals generated for backtesting")
    
    print("\nBacktest Metrics:")
    print(backtest_metrics)
    print("\nSaved test predictions to data/test_predictions.csv")
    
    print(f"\nModel training complete! Time taken: {datetime.now() - start_time}")
