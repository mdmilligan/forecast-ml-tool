import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class Backtester:
    def __init__(self, initial_capital=100000, commission=0.0005):
        self.initial_capital = initial_capital
        self.commission = commission
        
    def calculate_returns(self, df, signals):
        """Calculate strategy returns"""
        # Initialize portfolio values
        portfolio_value = self.initial_capital
        position = 0
        returns = []
        trade_count = 0
        
        # Calculate daily returns
        for i in range(1, len(df)):
            current_price = df['spy_close'].iloc[i]
            prev_price = df['spy_close'].iloc[i-1]
            
            # Calculate position change
            signal = signals.iloc[i]
            position_change = signal - position
            
            # Calculate return with commission
            if position_change != 0:
                trade_count += 1
                commission_cost = abs(position_change) * current_price * self.commission
                portfolio_value -= commission_cost
                
            # Update position
            position = signal
            
            # Calculate daily return
            daily_return = (current_price - prev_price) / prev_price * position
            portfolio_value *= (1 + daily_return)
            returns.append(portfolio_value)
            
        return np.array(returns), trade_count
    
    def calculate_metrics(self, returns, benchmark_returns, trade_count):
        """Calculate performance metrics"""
        # Calculate returns
        total_return = returns[-1] / self.initial_capital - 1
        annualized_return = (1 + total_return) ** (252/len(returns)) - 1
        
        # Calculate volatility
        daily_returns = np.diff(returns) / returns[:-1]
        annualized_vol = daily_returns.std() * np.sqrt(252)
        
        # Calculate Sharpe ratio with safety checks
        risk_free_rate = 0.0  # Can be adjusted
        if annualized_vol > 0:
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol
        else:
            sharpe_ratio = 0.0
        
        # Calculate max drawdown
        cumulative_max = np.maximum.accumulate(returns)
        drawdown = (cumulative_max - returns) / cumulative_max
        max_drawdown = drawdown.max()
        
        # Calculate benchmark metrics
        benchmark_total_return = benchmark_returns[-1] / benchmark_returns[0] - 1
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'benchmark_return': benchmark_total_return,
            'trade_count': trade_count
        }
    
    def plot_results(self, strategy_returns, benchmark_returns, metrics):
        """Plot backtest results"""
        plt.figure(figsize=(12, 8))
        
        # Plot strategy vs benchmark
        plt.plot(strategy_returns, label='Strategy')
        plt.plot(benchmark_returns, label='Benchmark (Buy & Hold)')
        plt.title(f"Backtest Results\nAnnualized Return: {metrics['annualized_return']:.2%}")
        plt.xlabel('Days')
        plt.ylabel('Portfolio Value')
        plt.legend()
        
        # Add metrics text box
        metrics_text = "\n".join([
            f"Total Return: {metrics['total_return']:.2%}",
            f"Annualized Vol: {metrics['annualized_volatility']:.2%}",
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}",
            f"Max Drawdown: {metrics['max_drawdown']:.2%}",
            f"Trades: {metrics['trade_count']}"
        ])
        plt.text(0.05, 0.95, metrics_text, 
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.show()

def run_backtest(df, signals):
    """Run complete backtest"""
    backtester = Backtester()
    
    # Calculate strategy returns
    strategy_returns, trade_count = backtester.calculate_returns(df, signals)
    
    # Calculate benchmark returns (buy & hold)
    benchmark_returns = df['spy_close'].values / df['spy_close'].iloc[0] * backtester.initial_capital
    
    # Calculate performance metrics
    metrics = backtester.calculate_metrics(strategy_returns, benchmark_returns, trade_count)
    
    # Plot results
    backtester.plot_results(strategy_returns, benchmark_returns, metrics)
    
    return metrics
