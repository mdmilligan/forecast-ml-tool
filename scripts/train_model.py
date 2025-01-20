import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
from data_processing import load_market_data, calculate_technical_indicators

def prepare_features(df):
    """Prepare features for model training"""
    feature_columns = [
        'spy_close', 'spy_volume', 'vix_close', 'uup_close',
        'sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal',
        'bb_upper', 'bb_lower', 'volatility'
    ]
    
    X = df[feature_columns]
    y = df['spy_close'].shift(-1) / df['spy_close'] - 1  # Next day's return
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_columns

def train_model(X_train, y_train):
    """Train and save the model"""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    print("Training model...")
    start_time = datetime.now()
    
    # Fetch and prepare data
    df = load_market_data()
    df = calculate_technical_indicators(df)
    X, y, scaler, feature_columns = prepare_features(df)
    
    # Train/test split
    train_size = int(len(df) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train and save model
    model = train_model(X_train, y_train)
    joblib.dump(model, 'data/model.pkl')
    joblib.dump(scaler, 'data/scaler.pkl')
    joblib.dump(feature_columns, 'data/feature_columns.pkl')
    
    print(f"Model training complete! Time taken: {datetime.now() - start_time}")
