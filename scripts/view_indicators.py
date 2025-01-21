import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_processing import load_market_data, calculate_technical_indicators

def plot_indicators(df):
    """Plot current technical indicators using Plotly"""
    # Create subplots with 30-min interval indicators
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                       vertical_spacing=0.1,  # Increased vertical spacing
                       row_heights=[0.4, 0.2, 0.2, 0.4, 0.2],  # Adjusted row heights
                       subplot_titles=(
                           'Price and Moving Averages',
                           'RSI',
                           'MACD',
                           'Bollinger Bands',
                           'Volatility'
                       ))
    
    # Price and Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['spy_close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['sma_50'], name='SMA 50'), row=1, col=1)
    
    # RSI with fixed range
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)  # Fix RSI range
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], name='Signal'), row=3, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_middle'], name='Middle Band'), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], name='Upper Band'), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_lower'], name='Lower Band'), row=4, col=1)
    
    # Update layout to only show periods with data
    fig.update_layout(
        height=1600,  # Increased height to prevent overlap
        width=1400,
        title_text="Technical Indicators - 30 Minute Intervals",
        showlegend=True,
        xaxis=dict(
            tickformat='%Y-%m-%d %H:%M',
            range=[df.index.min(), df.index.max()],  # Set range to actual data bounds
            autorange=False
        ),
        margin=dict(l=50, r=50, t=100, b=100)  # Increased bottom margin
    )
    
    # Remove gaps in time series
    fig.update_xaxes(
        rangebreaks=[
            # Hide weekends
            dict(bounds=["sat", "mon"]),
            # Hide hours outside trading times
            dict(bounds=[16, 9.5], pattern="hour")  # 4pm to 9:30am
        ]
    )
    
    # Add volatility plot
    fig.add_trace(go.Scatter(x=df.index, y=df['volatility'], name='Volatility'), row=5, col=1)
    
    # Update y-axes titles
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=4, col=1)
    fig.update_yaxes(title_text="Volatility", row=5, col=1)
    
    fig.show()

if __name__ == "__main__":
    print("Fetching and displaying current indicators...")
    
    # Fetch and calculate indicators
    df = load_market_data()
    df = calculate_technical_indicators(df)
    
    # Plot current indicators
    plot_indicators(df[-200:])  # Show last 200 periods (100 hours)
