import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from data_process import load_market_data

class MLTradingStrategy(Strategy):
    def init(self):
        # Load predictions from CSV with timezone handling
        self.predictions = pd.read_csv('data/test_predictions.csv', 
                                     index_col=0, 
                                     parse_dates=True)
        # Convert predictions index to timezone-naive datetime
        self.predictions.index = self.predictions.index.tz_localize(None)
        
        # Ensure predictions align with data
        self.predictions = self.predictions.reindex(self.data.index)
        
        # Access signals and confidence scores
        self.signals = self.predictions['Signal']
        self.confidence = self.predictions['Confidence']
        
    def next(self):
        # Get current position and signal
        # Access current bar's data using backtesting.py format
        current_bar = self.data.df.iloc[-1]
        current_signal = self.signals.loc[current_bar.name]
        confidence = self.confidence.loc[current_bar.name]
        
        # Only trade if confidence exceeds threshold
        if confidence > 0.7:
            if current_signal > 0 and not self.position:
                # Long entry
                self.buy(size=1)
            elif current_signal < 0 and not self.position:
                # Short entry
                self.sell(size=1)
            elif current_signal == 0 and self.position:
                # Exit position
                self.position.close()

def run_backtest():
    """Run backtest using backtesting.py"""
    try:
        # Load market data
        df = load_market_data()
        
        # Get test period data only
        predictions = pd.read_csv('data/test_predictions.csv', index_col=0, parse_dates=True)
        predictions.index = pd.to_datetime(predictions.index).tz_localize(None)
        
        # Convert df index to timezone-naive datetime for comparison
        df.index = df.index.tz_localize(None)
        
        # Filter market data to match prediction period
        start_date = predictions.index.min()
        end_date = predictions.index.max()
        df = df.loc[start_date:end_date]
        
        # Format data for backtesting.py
        data = pd.DataFrame({
            'Open': df['spy_open'],
            'High': df['spy_high'],
            'Low': df['spy_low'],
            'Close': df['spy_close'],
            'Volume': df['spy_volume']
        }, index=df.index)
        
        # Initialize and run backtest
        bt = Backtest(data, 
                     MLTradingStrategy,
                     cash=100000,
                     commission=0.0005,
                     exclusive_orders=True)
        
        # Run optimization
        stats = bt.run()
        
        # Plot results
        bt.plot()
        
        print("\nBacktest Results:")
        print(f"Total Return: {stats['Return [%]']:.2f}%")
        print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
        print(f"# Trades: {stats['# Trades']}")
        print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
        
        return stats
        
    except FileNotFoundError:
        print("Error: test_predictions.csv not found. Please run model_train.py first.")
    except Exception as e:
        print(f"Error running backtest: {str(e)}")
        raise

if __name__ == "__main__":
    run_backtest()
