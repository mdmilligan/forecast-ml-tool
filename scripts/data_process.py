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
        
        # Add binary zone indicator after ADMF is calculated
        df['admf_above_zero'] = (df['admf'] > 0).astype(int)
        
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
        # Add binary zone indicator
        df['fisher_above_zero'] = (df['fisher'] > 0).astype(int)
              
        # Calculate Ultimate RSI
        df['ultimate_rsi'], df['ultimate_rsi_signal'] = calculate_ultimate_rsi(df)
        # Add binary zone indicator
        df['ursi_above_50'] = (df['ultimate_rsi'] > 50).astype(int)
        
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
        
        # Candle body calculations
        df['candle_body'] = df['spy_close'] - df['spy_open']
        df['upper_wick'] = df['spy_high'] - df[['spy_open', 'spy_close']].max(axis=1)
        df['lower_wick'] = df[['spy_open', 'spy_close']].min(axis=1) - df['spy_low']
        df['candle_relative_position'] = np.where(
            (df['spy_high'] - df['spy_low']) != 0,
            (df['spy_close'] - df['spy_low']) / (df['spy_high'] - df['spy_low']),
            0.5
        )
        df['candle_direction'] = (df['spy_close'] > df['spy_open']).astype(int)
        
        # Support/Resistance Features
        def detect_bounces(df, level_col, threshold_pct=0.01):
            """
            Detect bounces near key levels (support/resistance)
            
            Args:
                df: DataFrame with price data
                level_col: Column name of the level to check (e.g. 'SMA20', 'bb_lower')
                threshold_pct: Percentage threshold to consider a bounce (default 1%)
            
            Returns:
                Series with bounce signals (1 = bounce up, -1 = bounce down, 0 = no bounce)
            """
            level = df[level_col]
            threshold = level * threshold_pct
            
            # Detect bounces up (price approaches from below then moves up)
            bounce_up = (
                (df['spy_low'] <= level + threshold) & 
                (df['spy_close'] > level + threshold)
            )
            
            # Detect bounces down (price approaches from above then moves down)
            bounce_down = (
                (df['spy_high'] >= level - threshold) & 
                (df['spy_close'] < level - threshold)
            )
            
            return bounce_up.astype(int) - bounce_down.astype(int)
        
        # Add bounce detection for key levels (SMA and 1D BB)
        df['bounce_SMA5'] = detect_bounces(df, 'SMA5')
        df['bounce_SMA10'] = detect_bounces(df, 'SMA10')
        df['bounce_SMA20'] = detect_bounces(df, 'SMA20') 
        df['bounce_SMA50'] = detect_bounces(df, 'SMA50')
        df['bounce_SMA100'] = detect_bounces(df, 'SMA100')
        df['bounce_bb_1d_upper'] = detect_bounces(df, 'bb_1d_upper')
        df['bounce_bb_1d_lower'] = detect_bounces(df, 'bb_1d_lower')
        
        # Add strength of bounce (how far price moved away)
        df['bounce_strength_SMA5'] = np.where(
            df['bounce_SMA5'] != 0,
            (df['spy_close'] - df['SMA5']) / df['SMA5'],
            0
        )
        df['bounce_strength_SMA10'] = np.where(
            df['bounce_SMA10'] != 0,
            (df['spy_close'] - df['SMA10']) / df['SMA10'],
            0
        )
        df['bounce_strength_SMA20'] = np.where(
            df['bounce_SMA20'] != 0,
            (df['spy_close'] - df['SMA20']) / df['SMA20'],
            0
        )
        df['bounce_strength_SMA50'] = np.where(
            df['bounce_SMA50'] != 0,
            (df['spy_close'] - df['SMA50']) / df['SMA50'],
            0
        )
        df['bounce_strength_SMA100'] = np.where(
            df['bounce_SMA100'] != 0,
            (df['spy_close'] - df['SMA100']) / df['SMA100'],
            0
        )
        df['bounce_strength_bb_1d_upper'] = np.where(
            df['bounce_bb_1d_upper'] != 0,
            (df['spy_close'] - df['bb_1d_upper']) / df['bb_1d_upper'],
            0
        )
        df['bounce_strength_bb_1d_lower'] = np.where(
            df['bounce_bb_1d_lower'] != 0,
            (df['spy_close'] - df['bb_1d_lower']) / df['bb_1d_lower'],
            0
        )
        
        # Add recent touch counts (how many times price has touched level recently)
        for level in ['SMA5', 'SMA10', 'SMA20', 'SMA50', 'SMA100', 'bb_1d_upper', 'bb_1d_lower']:
            df[f'touch_count_{level}'] = (
                (df['spy_low'] <= df[level] * 1.01) & 
                (df['spy_high'] >= df[level] * 0.99)
            ).rolling(window=20).sum()
        
        # Add proximity to levels
        for level in ['SMA5', 'SMA10', 'SMA20', 'SMA50', 'SMA100', 'bb_1d_upper', 'bb_1d_lower']:
            df[f'proximity_{level}'] = (df['spy_close'] - df[level]) / df[level]
        
        # Statistical calculations
        # Autocorrelation (1-period lag)
        df['autocorr_30m'] = df['spy_close'].autocorr(lag=1)
        
        # Skewness (daily and weekly)
        df['skewness_1d'] = df['spy_close'].rolling(window=26).skew()  # 1 trading day
        df['skewness_5d'] = df['spy_close'].rolling(window=130).skew()  # 5 trading days
        
        # Z-score (daily and weekly)
        rolling_mean_1d = df['spy_close'].rolling(window=26).mean()
        rolling_std_1d = df['spy_close'].rolling(window=26).std()
        df['z_score_1d'] = (df['spy_close'] - rolling_mean_1d) / rolling_std_1d
        
        rolling_mean_5d = df['spy_close'].rolling(window=130).mean()
        rolling_std_5d = df['spy_close'].rolling(window=130).std()
        df['z_score_5d'] = (df['spy_close'] - rolling_mean_5d) / rolling_std_5d
        
        # Rolling percentile (daily and weekly)
        def rolling_percentile(x):
            return (x.rank(pct=True).iloc[-1] * 100)
        df['percentile_1d'] = df['spy_close'].rolling(window=26).apply(rolling_percentile)
        df['percentile_5d'] = df['spy_close'].rolling(window=130).apply(rolling_percentile)
        
        # Entropy (daily and weekly)
        def rolling_entropy(x):
            value_counts = x.value_counts(normalize=True)
            return -(value_counts * np.log(value_counts)).sum()
        df['entropy_1d'] = df['spy_close'].rolling(window=26).apply(rolling_entropy)
        df['entropy_5d'] = df['spy_close'].rolling(window=130).apply(rolling_entropy)
        
        # Market State Classification
        def classify_market_state(df):
            # Calculate composite scores using most relevant indicators
            df['trend_score'] = (
                (1 - df['entropy_1d']) +         # Low entropy favors trending
                df['autocorr_30m'].clip(0, 1) +  # High autocorrelation favors trending
                df['z_score_1d'].abs() +         # Extreme z-scores favor trending
                ((df['percentile_1d'] - 50).abs() / 50)  # Extreme percentiles favor trending
            ) / 4  # Normalize to 0-1 range
            
            df['chop_score'] = (
                df['entropy_1d'] +               # High entropy favors choppy
                (1 - df['autocorr_30m'].abs()) + # Low autocorrelation favors choppy
                (1 - df['z_score_1d'].abs()) +   # Z-scores near zero favor choppy
                (1 - ((df['percentile_1d'] - 50).abs() / 50))  # Percentile near 50 favors choppy
            ) / 4  # Normalize to 0-1 range
            
            # Classify market state with clear thresholds
            conditions = [
                (df['trend_score'] > 0.7) & (df['chop_score'] < 0.3),  # Strong trend
                (df['chop_score'] > 0.7) & (df['trend_score'] < 0.3),  # Strong chop
                (df['trend_score'].between(0.4, 0.7)) |               # Transitioning
                (df['chop_score'].between(0.4, 0.7))                  # Transitioning
            ]
            choices = ['trending', 'choppy', 'transitioning']
            
            df['market_state'] = np.select(conditions, choices, default='uncertain')
            return df
        
        # Apply market state classification
        df = classify_market_state(df)
        
        # Clean up intermediate columns
        columns_to_drop = ['hl2', 'ad_ratio', 'trend_score', 'chop_score']
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

