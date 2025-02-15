import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_process import load_market_data, calculate_technical_indicators

def plot_indicators(df):
    """Plot current technical indicators using Plotly"""
    # Create subplots with requested indicators
    fig = make_subplots(rows=11, cols=1, shared_xaxes=True,
                       vertical_spacing=0.05,
                       row_heights=[0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                       subplot_titles=(
                           'Price and Moving Averages',
                           'BB %B',
                           'BB Bandwidth',
                           'Volatility',
                           'ATR',
                           'ADMF',
                           'ROC',
                           'Fisher Transform',
                           'Distance to MAs',
                           '5D SMA Slope',
                           'Ultimate RSI'
                       ))
    
    # Price and Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['spy_close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA5'], name='5D SMA', line=dict(color='blue', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='20D SMA', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='50D SMA', line=dict(color='green', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA100'], name='100D SMA', line=dict(color='purple', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA150'], name='150D SMA', line=dict(color='brown', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], name='200D SMA', line=dict(color='pink', width=1)), row=1, col=1)
    
    # BB %B
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_percent_b'], name='BB %B'), row=2, col=1)
    fig.add_hline(y=1, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="green", row=2, col=1)
    
    # BB Bandwidth
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_bandwidth'], name='BB Bandwidth'), row=3, col=1)
    
    # Volatility
    fig.add_trace(go.Scatter(x=df.index, y=df['volatility'], name='Volatility'), row=4, col=1)
    
    # ATR
    fig.add_trace(go.Scatter(x=df.index, y=df['atr'], name='ATR'), row=5, col=1)
    
    # ADMF
    fig.add_trace(go.Scatter(x=df.index, y=df['admf'], name='ADMF'), row=6, col=1)
    
    # ROC
    fig.add_trace(go.Scatter(x=df.index, y=df['roc'], name='ROC'), row=7, col=1)
    
    # Fisher Transform
    fig.add_trace(go.Scatter(x=df.index, y=df['fisher'], name='Fisher'), row=8, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['fisher_trigger'], name='Trigger'), row=8, col=1)
    
    # Distance to MAs
    fig.add_trace(go.Scatter(x=df.index, y=df['dist_to_EMA21'], name='Dist to EMA21'), row=9, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['dist_to_EMA50'], name='Dist to EMA50'), row=9, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['dist_to_5day_SMA'], name='Dist to 5D SMA'), row=9, col=1)
    
    # 5D SMA Slope
    fig.add_trace(go.Scatter(x=df.index, y=df['slope'], name='5D SMA Slope'), row=10, col=1)
    
    # Ultimate RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['ultimate_rsi'], name='Ultimate RSI'), row=11, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ultimate_rsi_signal'], name='RSI Signal', line=dict(dash='dot')), row=11, col=1)
    
    # Update layout to only show periods with data
    fig.update_layout(
        height=2200,  # Increased height to accommodate extra row
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
    
    # Update y-axes titles
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="BB %B", row=2, col=1)
    fig.update_yaxes(title_text="BB Bandwidth", row=3, col=1)
    fig.update_yaxes(title_text="Volatility", row=4, col=1)
    fig.update_yaxes(title_text="ATR", row=5, col=1)
    fig.update_yaxes(title_text="ADMF", row=6, col=1)
    fig.update_yaxes(title_text="ROC", row=7, col=1)
    fig.update_yaxes(title_text="Fisher", row=8, col=1)
    fig.update_yaxes(title_text="Distance %", row=9, col=1)
    fig.update_yaxes(title_text="Slope", row=10, col=1)
    fig.update_yaxes(title_text="RSI", row=11, col=1)
    
    fig.show()

if __name__ == "__main__":
    print("Fetching and displaying current indicators...")
    
    # Fetch and calculate indicators for most recent data
    df = load_market_data(start_date='2024-12-01', end_date='2025-01-20')
    df = calculate_technical_indicators(df)
    
    # Plot current indicators
    plot_indicators(df)  # Show all available recent data
