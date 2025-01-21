import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import pandas as pd
from datetime import datetime
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
    y = df['spy_close'].shift(-1) / df['spy_close'] - 1  # Next period's return
    
    # Drop rows with missing values (from indicator calculations)
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_columns, valid_idx

def train_model(X_train, y_train):
    """Train and save the model with improved parameters"""
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        verbose=1
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
    
    # Feature importance analysis
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Feature Importances:")
    print(feature_importance_df.head(10))
    
    # Save test predictions
    test_results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    }, index=df.index[train_size:train_size+len(y_test)])
    
    test_results.to_csv('data/test_predictions.csv')
    print("\nSaved test predictions to data/test_predictions.csv")
    
    print(f"\nModel training complete! Time taken: {datetime.now() - start_time}")
