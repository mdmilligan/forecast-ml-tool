def calculate_admf(df, length=14, price_enable=True, ad_weight=0.0):
    """
    Calculate Accumulation/Distribution Money Flow indicator
    
    Parameters:
    df: DataFrame with OHLCV data
    length: Length for the RMA calculation (default 14)
    price_enable: Whether to factor in price (default True)
    ad_weight: A/D weight parameter (default 0.0)
    """
    # Calculate True Range
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': abs(df['High'] - df['Close'].shift(1)),
        'lc': abs(df['Low'] - df['Close'].shift(1))
    }).max(axis=1)
    
    # Calculate A/D ratio
    price_change = df['Close'].diff()
    ad_ratio = price_change / tr
    ad_ratio = ad_ratio.fillna(0)  # Replace NaN with 0 (equivalent to Pine's nz())
    
    # Apply A/D weight
    ad_ratio = (1 - ad_weight) * ad_ratio + np.sign(ad_ratio) * ad_weight
    
    # Calculate volume component
    if price_enable:
        hlc3 = (df['High'] + df['Low'] + df['Close']) / 3
        volume_factor = df['Volume'] * hlc3
    else:
        volume_factor = df['Volume']
    
    # Calculate final ADMF
    admf = volume_factor * ad_ratio
    
    # Apply RMA (equivalent to Pine's rma())
    alpha = 1 / length
    admf_rma = admf.ewm(alpha=alpha, adjust=False).mean()
    
    return admf_rma

# To add this to your existing code, modify the fetch_data function by adding these lines 
# after your existing technical indicators:

def fetch_data(start_date='2010-01-01', end_date='2024-01-01'):
    """
    Fetch historical data for SPY and indicators
    """
    # [Previous code remains the same until technical indicators section]
    
    # Add ADMF indicator
    df['admf'] = calculate_admf(
        pd.DataFrame({
            'High': spy['High'],
            'Low': spy['Low'],
            'Close': spy['Close'],
            'Volume': spy['Volume']
        }),
        length=14,
        price_enable=True,
        ad_weight=0.0
    )
    
    # Add to feature_columns in prepare_features function:
    feature_columns = [
        'spy_close', 'spy_volume', 'vix_close',
        'sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal',
        'bb_upper', 'bb_lower', 'volatility', 'admf'  # Added 'admf' here
    ]