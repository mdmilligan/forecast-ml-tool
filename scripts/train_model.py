import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import sqlite3
import joblib
from datetime import datetime

def fetch_data(start_date='2010-01-01', end_date='2024-01-01'):
    """Fetch and preprocess historical data"""
    conn = sqlite3.connect('data/marketdata.db')
    
    # Read data from database
    spy = pd.read_sql("SELECT * FROM stock_data_spy WHERE date >= ? AND date <= ?", 
                     conn, params=(start_date, end_date), parse_dates=['date'], index_col='date')
    vix = pd.read_sql("SELECT * FROM stock_data_vix WHERE date >= ? AND date <= ?", 
                     conn, params=(start_date, end_date), parse_dates=['date'], index_col='date')
    uup = pd.read_sql("SELECT * FROM stock_data_uup WHERE date >= ? AND date <= ?", 
                     conn, params=(start_date, end_date), parse_dates=['date'], index_col='date')
    
    # Create main dataframe with SPY data
    df = pd.DataFrame(index=spy.index)
    df['spy_close'] = spy['close']
    df['spy_open'] = spy['open']
    df['spy_high'] = spy['high']
    df['spy_low'] = spy['low']
    df['spy_volume'] = spy['volume']
    df['vix_close'] = vix['close']
    df['uup_close'] = uup['close']
    
    return df.dropna()

def calculate_technical_indicators(df):
    """Calculate all technical indicators"""
    # Moving averages
    df['sma_20'] = df['spy_close'].rolling(window=20).mean()
    df['sma_50'] = df['spy_close'].rolling(window=50).mean()
    
    # RSI
    delta = df['spy_close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['spy_close'].ewm(span=12, adjust=False).mean()
    exp2 = df['spy_close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['spy_close'].rolling(window=20).mean()
    df['bb_upper'] = df['bb_middle'] + 2 * df['spy_close'].rolling(window=20).std()
    df['bb_lower'] = df['bb_middle'] - 2 * df['spy_close'].rolling(window=20).std()
    
    # Volatility
    df['volatility'] = df['spy_close'].rolling(window=20).std()
    
    return df

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
    df = fetch_data()
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
