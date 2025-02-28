"""
Backtesting using vectorbt library
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from datetime import datetime
import matplotlib.pyplot as plt
from data_process import load_market_data

def load_and_prepare_data():
    """Load and prepare data for backtesting"""
    try:
        # Load market data
        print("Loading market data...")
        df = load_market_data()
        
        # Load predictions
        print("Loading predictions...")
        predictions = pd.read_csv('data/test_predictions.csv', 
                                index_col=0, 
                                parse_dates=True)
        predictions.index = predictions.index.tz_localize(None)
        
        # Filter market data to match prediction period
        start_date = predictions.index.min()
        end_date = predictions.index.max()
        df = df.loc[start_date:end_date]
        
        # Align predictions with market data
        predictions = predictions.reindex(df.index)
        
        return df, predictions
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("Please run model_train.py first to generate predictions.")
        raise
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def run_backtest(initial_capital=100000, commission=0.0005):
    """Run vectorbt backtest"""
    try:
        # Load and prepare data
        df, predictions = load_and_prepare_data()
        
        # Create portfolio
        portfolio = vbt.Portfolio.from_signals(
            close=df['spy_close'],
            entries=predictions['Signal'] > 0,  # Long signals
            exits=predictions['Signal'] < 0,   # Exit signals
            size=1.0,  # Full position size
            fees=commission,
            freq='30T',  # 30 minute bars
            init_cash=initial_capital
        )
        
        # Get performance stats
        stats = portfolio.stats()
        
        # Print detailed results
        print("\nBacktest Results:")
        print(f"Start Date: {stats['Start']}")
        print(f"End Date: {stats['End']}")
        print(f"Total Return: {stats['Total Return [%]']:.2f}%")
        print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown: {stats['Max Drawdown [%]']:.2f}%")
        print(f"Total Trades: {stats['Total Trades']}")
        print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
        print(f"Profit Factor: {stats['Profit Factor']:.2f}")
        print(f"Expectancy: {stats['Expectancy']:.2f}")
        
        # Plot results
        print("\nPlotting results...")
        fig = plt.figure(figsize=(16, 12))
        
        # Portfolio value
        ax1 = fig.add_subplot(311)
        portfolio.value().vbt.plot(ax=ax1, title='Portfolio Value')
        
        # Drawdowns
        ax2 = fig.add_subplot(312)
        portfolio.drawdown().vbt.plot(ax=ax2, title='Drawdown')
        
        # Trades
        ax3 = fig.add_subplot(313)
        portfolio.trades.plot_pnl(ax=ax3, title='Trade PnL')
        
        plt.tight_layout()
        plt.show()
        
        return portfolio, stats
        
    except Exception as e:
        print(f"Error running backtest: {str(e)}")
        raise

if __name__ == "__main__":
    # Run with default parameters
    portfolio, stats = run_backtest()
    
    # Optionally save results
    save_results = input("\nSave results to CSV? (y/n): ").lower() == 'y'
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        portfolio.trades.records_readable.to_csv(f'data/backtest_results_{timestamp}.csv')
        print(f"Results saved to data/backtest_results_{timestamp}.csv")
