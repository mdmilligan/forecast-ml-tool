import pandas as pd
import matplotlib.pyplot as plt
from train_model import fetch_data, calculate_technical_indicators

def plot_indicators(df):
    """Plot current technical indicators"""
    fig, axes = plt.subplots(4, 1, figsize=(15, 20))
    
    # Price and Moving Averages
    df[['spy_close', 'sma_20', 'sma_50']].plot(ax=axes[0])
    axes[0].set_title('Price and Moving Averages')
    
    # RSI
    df['rsi'].plot(ax=axes[1])
    axes[1].axhline(70, color='r', linestyle='--')
    axes[1].axhline(30, color='g', linestyle='--')
    axes[1].set_title('RSI')
    
    # MACD
    df[['macd', 'macd_signal']].plot(ax=axes[2])
    axes[2].set_title('MACD')
    
    # Bollinger Bands
    df[['spy_close', 'bb_upper', 'bb_lower']].plot(ax=axes[3])
    axes[3].set_title('Bollinger Bands')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Fetching and displaying current indicators...")
    
    # Fetch and calculate indicators
    df = fetch_data()
    df = calculate_technical_indicators(df)
    
    # Plot current indicators
    plot_indicators(df[-100:])  # Show last 100 days
