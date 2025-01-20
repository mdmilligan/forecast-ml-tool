import pandas as pd
import numpy as np
import sqlite3

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
