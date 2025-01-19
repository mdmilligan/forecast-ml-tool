import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from train_model import fetch_data, calculate_technical_indicators

def plot_indicators(df):
    """Plot current technical indicators using Plotly"""
    # Create subplots
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                       vertical_spacing=0.05,
                       subplot_titles=('Price and Moving Averages', 'RSI', 'MACD', 'Bollinger Bands'))
    
    # Price and Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['spy_close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['sma_50'], name='SMA 50'), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], name='Signal'), row=3, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['spy_close'], name='Price'), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], name='Upper Band'), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_lower'], name='Lower Band'), row=4, col=1)
    
    # Update layout
    fig.update_layout(height=1000, width=1200, 
                     title_text="Technical Indicators",
                     showlegend=True)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=4, col=1)
    
    fig.show()

if __name__ == "__main__":
    print("Fetching and displaying current indicators...")
    
    # Fetch and calculate indicators
    df = fetch_data()
    df = calculate_technical_indicators(df)
    
    # Plot current indicators
    plot_indicators(df[-100:])  # Show last 100 days
