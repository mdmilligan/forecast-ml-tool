import pandas as pd
import numpy as np
import sqlite3
from typing import Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_market_data(start_date='2013-01-01', end_date='2025-12-31'):
    """Load and preprocess historical market data from database"""
    try:
        conn = sqlite3.connect('data/marketdata.db')
        
        # Read data from database in ascending date order and ensure proper datetime index
        spy = pd.read_sql("SELECT * FROM stock_data_spy WHERE date >= ? AND date <= ? ORDER BY date ASC", 
                         conn, params=(start_date, end_date))
        spy['date'] = pd.to_datetime(spy['date'], utc=True)
        spy = spy.set_index('date')
        
        vix = pd.read_sql("SELECT * FROM stock_data_vix WHERE date >= ? AND date <= ? ORDER BY date ASC", 
                         conn, params=(start_date, end_date))
        vix['date'] = pd.to_datetime(vix['date'], utc=True)
        vix = vix.set_index('date')
        
        uup = pd.read_sql("SELECT * FROM stock_data_uup WHERE date >= ? AND date <= ? ORDER BY date ASC", 
                         conn, params=(start_date, end_date))
        uup['date'] = pd.to_datetime(uup['date'], utc=True)
        uup = uup.set_index('date')
        
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
    
    except Exception as e:
        logger.error(f"Error loading market data: {str(e)}")
        raise

def calculate_ultimate_rsi(df: pd.DataFrame, length: int = 14, smooth: int = 5) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Ultimate RSI and its signal line.
    
    Args:
        df: DataFrame with price data
        length: Lookback period for RSI calculation
        smooth: Smoothing period for signal line
    
    Returns:
        Tuple containing URSI values and signal line
    """
    try:
        # Input validation
        if length <= 0 or smooth <= 0:
            raise ValueError("Length and smooth parameters must be positive")
        
        source = df['spy_close']
        upper = source.rolling(window=length).max()
        lower = source.rolling(window=length).min()
        price_range = upper - lower
        price_diff = source.diff()
        
        diff = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            if upper.iloc[i] > upper.iloc[i-1]:
                diff.iloc[i] = price_range.iloc[i]
            elif lower.iloc[i] < lower.iloc[i-1]:
                diff.iloc[i] = -price_range.iloc[i]
            else:
                diff.iloc[i] = float(price_diff.iloc[i])
                
        num = diff.ewm(alpha=1/length, adjust=False).mean()
        den = diff.abs().ewm(alpha=1/length, adjust=False).mean()
        
        ultimate_rsi = np.where(
            den != 0,
            (num / den) * 50 + 50,
            50
        )
        
        # Convert ultimate_rsi to Series and calculate EMA
        ultimate_rsi_series = pd.Series(ultimate_rsi, index=df.index)
        signal = ultimate_rsi_series.ewm(span=smooth, adjust=False).mean()
        
        # Fill initial NaN values with the first valid value
        signal.bfill(inplace=True)
        
        return ultimate_rsi_series, signal.rename('ultimate_rsi_signal')
        
    except Exception as e:
        logger.error(f"Error calculating Ultimate RSI: {str(e)}")
        raise

def calculate_technical_indicators(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """
    Calculate all technical indicators
    
    Args:
        df: Input dataframe with OHLCV data
        params: Dictionary of parameters for indicators
    
    Returns:
        DataFrame with added technical indicators
    """
    """
    Calculate all technical indicators
    
    Args:
        df: Input dataframe with OHLCV data
        params: Dictionary of parameters for indicators
    
    Returns:
        DataFrame with added technical indicators
    """
    # Default parameters
    default_params = {
        'admf_weight': 0.6,         # Weight for momentum calculation (0-1): higher = more reactive to recent price changes
        'admf_length': 20,          # Lookback period for ADMF calculation: higher = smoother momentum measurement
        'roc_period': 10,           # Rate of Change calculation period: higher = longer-term price changes
        'fisher_length': 10,        # Fisher Transform lookback period: higher = smoother oscillator
        'price_column': 'spy_close', # Price data to use for calculations (close/open/high/low)
        'sma_period': 5,            # Short-term moving average period: lower = more sensitive to price changes
        'lookback': 5,              # Slope calculation lookback period: higher = smoother slope measurement
        'admf_price_enable': True   # Whether to use price-weighted volume in ADMF calculation
    }
    
    params = {**default_params, **(params or {})}
    
    try:
        # Moving averages (converted to daily periods - 26 periods per day)
        df['EMA21'] = df[params['price_column']].ewm(span=21*13, adjust=False).mean()
        df['EMA50'] = df[params['price_column']].ewm(span=50*13, adjust=False).mean()
        # Daily SMAs (26 periods per day for 30-minute data)
        df['SMA5'] = df[params['price_column']].rolling(window=5*13).mean()  # 5-day SMA
        df['SMA10'] = df[params['price_column']].rolling(window=10*13).mean()  # 10-day SMA
        df['SMA20'] = df[params['price_column']].rolling(window=20*13).mean()  # 20-day SMA
        df['SMA50'] = df[params['price_column']].rolling(window=50*13).mean()  # 50-day SMA
        df['SMA100'] = df[params['price_column']].rolling(window=100*13).mean()  # 100-day SMA
        df['SMA150'] = df[params['price_column']].rolling(window=150*13).mean()  # 150-day SMA
        df['SMA200'] = df[params['price_column']].rolling(window=200*13).mean()  # 200-day SMA
        
        # Distance to MAs
        df['dist_to_EMA21'] = ((df[params['price_column']] - df['EMA21']) / df['EMA21']) * 100
        df['dist_to_EMA50'] = ((df[params['price_column']] - df['EMA50']) / df['EMA50']) * 100
        df['dist_to_5D_SMA'] = ((df[params['price_column']] - df['SMA5']) / df['SMA5']) * 100
        df['dist_to_20D_SMA'] = ((df[params['price_column']] - df['SMA20']) / df['SMA20']) * 100
        df['dist_to_50D_SMA'] = ((df[params['price_column']] - df['SMA50']) / df['SMA50']) * 100
        df['dist_to_100D_SMA'] = ((df[params['price_column']] - df['SMA100']) / df['SMA100']) * 100
        df['dist_to_150D_SMA'] = ((df[params['price_column']] - df['SMA150']) / df['SMA150']) * 100
        df['dist_to_200D_SMA'] = ((df[params['price_column']] - df['SMA200']) / df['SMA200']) * 100
        
        # Slope calculations
        rad2degree = 180/3.14159265359
        lookback_periods = params['lookback']*13
        
        # 5D SMA Slope
        df['5D_Slope'] = rad2degree * np.arctan(
            (df['SMA5'] - df['SMA5'].shift(lookback_periods)) / lookback_periods
        )
        
        # EMA21 Slope
        df['EMA21_Slope'] = rad2degree * np.arctan(
            (df['EMA21'] - df['EMA21'].shift(lookback_periods)) / lookback_periods
        )
        
        # EMA50 Slope
        df['EMA50_Slope'] = rad2degree * np.arctan(
            (df['EMA50'] - df['EMA50'].shift(lookback_periods)) / lookback_periods
        )
        
        # 20D SMA Slope
        df['20D_Slope'] = rad2degree * np.arctan(
            (df['SMA20'] - df['SMA20'].shift(lookback_periods)) / lookback_periods
        )
        
        # 50D SMA Slope
        df['50D_Slope'] = rad2degree * np.arctan(
            (df['SMA50'] - df['SMA50'].shift(lookback_periods)) / lookback_periods
        )
        
        # Bollinger Bands
        df['bb_middle'] = df['spy_close'].rolling(window=20).mean()
        bb_std = df['spy_close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        
        # Bollinger Band indicators
        df['bb_percent_b'] = np.where(
            (df['bb_upper'] - df['bb_lower']) != 0,
            (df['spy_close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']),
            0
        )
        
        # Distance to 30-min BB
        df['dist_to_bb_upper'] = ((df['spy_close'] - df['bb_upper']) / df['bb_upper']) * 100
        df['dist_to_bb_lower'] = ((df['spy_close'] - df['bb_lower']) / df['bb_lower']) * 100
        
        # 1-Day Equivalent Bollinger Bands
        df['bb_1d_middle'] = df['spy_close'].rolling(window=13).mean()
        bb_1d_std = df['spy_close'].rolling(window=13).std()
        df['bb_1d_upper'] = df['bb_1d_middle'] + 2 * bb_1d_std
        df['bb_1d_lower'] = df['bb_1d_middle'] - 2 * bb_1d_std
        
        df['bb_1d_percent_b'] = np.where(
            (df['bb_1d_upper'] - df['bb_1d_lower']) != 0,
            (df['spy_close'] - df['bb_1d_lower']) / (df['bb_1d_upper'] - df['bb_1d_lower']),
            0
        )
        
        # Distance to 1D BB
        df['dist_to_bb_1d_upper'] = ((df['spy_close'] - df['bb_1d_upper']) / df['bb_1d_upper']) * 100
        df['dist_to_bb_1d_lower'] = ((df['spy_close'] - df['bb_1d_lower']) / df['bb_1d_lower']) * 100
        
        df['bb_bandwidth'] = np.where(
            df['bb_middle'] != 0,
            ((df['bb_upper'] - df['bb_lower']) / df['bb_middle']) * 100,
            0
        )
        
        # Volatility
        df['volatility'] = df['spy_close'].rolling(window=20*2).std()
        
        # True Range and ATR
        high_low = df['spy_high'] - df['spy_low']
        high_close = abs(df['spy_high'] - df['spy_close'].shift(1))
        low_close = abs(df['spy_low'] - df['spy_close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.ewm(span=14, adjust=False).mean()
        
        # ADMF Calculation
        df['ad_ratio'] = df['spy_close'].diff() / tr
        df['ad_ratio'] = df['ad_ratio'].fillna(0)
        df['ad_ratio'] = (1 - params['admf_weight']) * df['ad_ratio'] + np.sign(df['ad_ratio']) * params['admf_weight']
        
        if params['admf_price_enable']:
            df['hlc3'] = (df['spy_high'] + df['spy_low'] + df['spy_close']) / 3
            volume_factor = df['spy_volume'] * df['hlc3']
        else:
            volume_factor = df['spy_volume']
            
        alpha = 1 / params['admf_length']
        df['admf'] = (volume_factor * df['ad_ratio']).ewm(alpha=alpha, adjust=False).mean()
        
        # Rate of Change
        df['roc'] = ((df['spy_close'] - df['spy_close'].shift(params['roc_period'])) / 
                     df['spy_close'].shift(params['roc_period'])) * 100
        
        # Fisher Transform
        df['hl2'] = (df['spy_high'] + df['spy_low']) / 2
        period_min = df['hl2'].rolling(params['fisher_length']).min()
        period_max = df['hl2'].rolling(params['fisher_length']).max()
        
        # Calculate normalized price
        normalized_price = 0.66 * ((df['hl2'] - period_min) / 
                                 (period_max - period_min + 1e-9) - 0.5)
        
        # Calculate value with EMA smoothing
        value = normalized_price.ewm(alpha=0.33, adjust=False).mean()
        value = value.clip(-0.999, 0.999)
        
        # Calculate Fisher Transform
        df['fisher'] = 0.5 * np.log((1 + value) / (1 - value))
        df['fisher'] = df['fisher'].ewm(alpha=0.5, adjust=False).mean()
        df['fisher_trigger'] = df['fisher'].shift(1)
              
        # Calculate Ultimate RSI
        df['ultimate_rsi'], df['ultimate_rsi_signal'] = calculate_ultimate_rsi(df)
        
        # Calculate stop-loss and take-profit levels
        df['atr_20'] = df['atr'].rolling(window=20).mean()
        df['stop_loss_long'] = df['spy_low'].rolling(window=5).min() - 1.5 * df['atr_20']
        df['stop_loss_short'] = df['spy_high'].rolling(window=5).max() + 1.5 * df['atr_20']
        df['take_profit_long'] = df['spy_close'] + 2.5 * df['atr_20']
        df['take_profit_short'] = df['spy_close'] - 2.5 * df['atr_20']
        
        # Donchian Channel (20-period)
        df['donchian_upper'] = df['spy_high'].rolling(window=20).max()
        df['donchian_lower'] = df['spy_low'].rolling(window=20).min()
        df['donchian_middle'] = (df['donchian_upper'] + df['donchian_lower']) / 2
        
        # Donchian Channel Width
        df['donchian_width'] = df['donchian_upper'] - df['donchian_lower']
        
        # Donchian Channel Position
        df['donchian_position'] = np.where(
            df['donchian_width'] != 0,
            (df['spy_close'] - df['donchian_lower']) / df['donchian_width'],
            0
        )
        
        # Clean up intermediate columns
        columns_to_drop = ['hl2', 'ad_ratio']
        df = df.drop(columns_to_drop, axis=1, errors='ignore')
        
        return df
        
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {str(e)}")
        raise

if __name__ == "__main__":
    # Test the data processing functions
    print("Testing data processing functions...")
    try:
        # Load sample data
        df = load_market_data(start_date='2024-01-01', end_date='2024-01-10')
        print(f"Loaded {len(df)} rows of data")
        
        # Calculate indicators
        df = calculate_technical_indicators(df)
        print("Calculated indicators:")
        print(df[['spy_close', 'EMA21', 'EMA50', 'volatility']].tail())
        
    except Exception as e:
        print(f"Error in data processing test: {str(e)}")

