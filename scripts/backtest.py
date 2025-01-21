import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class Backtester:
    def __init__(self, initial_capital=100000, commission=0.0005):
        self.initial_capital = initial_capital
        self.commission = commission
        
    def calculate_returns(self, df, signals, confidence_scores=None):
        """Calculate strategy returns with confidence-based position sizing"""
        # Initialize portfolio values
        portfolio_value = self.initial_capital
        position = 0
        returns = []
        trade_count = 0
        winning_trades = 0
        losing_trades = 0
        total_position_size = 0
        
        # If no confidence scores provided, use full position
        if confidence_scores is None:
            confidence_scores = np.ones(len(signals))
            
        # Track trade statistics
        trade_stats = {
            'entry_dates': [],
            'exit_dates': [],
            'returns': [],
            'durations': [],
            'position_sizes': []
        }
        
        # Initialize previous price
        prev_price = df['spy_close'].iloc[0]
        
        # Calculate daily returns
        for i in range(1, len(df)):
            current_price = df['spy_close'].iloc[i]
            prev_price = df['spy_close'].iloc[i-1]
            
            # Calculate position change with confidence-based sizing
            signal = signals.iloc[i]
            confidence = confidence_scores[i] if i < len(confidence_scores) else 1.0
            target_position = signal * confidence
            position_change = target_position - position
            
            # Calculate daily return
            daily_return = (current_price - prev_price) / prev_price * position
            portfolio_value *= (1 + daily_return)
            returns.append(portfolio_value)
            
            # Track trade statistics when entering/exiting positions
            if position_change != 0:
                if position != 0:  # Exiting a position
                    trade_stats['exit_dates'].append(df.index[i])
                    trade_stats['returns'].append(daily_return)
                    trade_stats['durations'].append(len(trade_stats['entry_dates']) - len(trade_stats['exit_dates']))
                    trade_stats['position_sizes'].append(abs(position))
                    
                    if daily_return > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1
                
                if target_position != 0:  # Entering a position
                    trade_stats['entry_dates'].append(df.index[i])
                
                total_position_size += abs(position_change)
            
            # Calculate return with commission
            if position_change != 0:
                trade_count += 1
                commission_cost = abs(position_change) * current_price * self.commission
                portfolio_value -= commission_cost
                
            # Update position
            position = signal
            
        return (
            np.array(returns), 
            trade_count,
            winning_trades,
            losing_trades,
            total_position_size,
            trade_stats
        )
    
    def calculate_metrics(self, returns, benchmark_returns, trade_count, winning_trades, losing_trades, total_position_size, trade_stats):
        """Calculate enhanced performance metrics"""
        # Calculate returns
        total_return = returns[-1] / self.initial_capital - 1
        annualized_return = (1 + total_return) ** (252/len(returns)) - 1
        
        # Calculate trade statistics
        win_rate = winning_trades / trade_count if trade_count > 0 else 0
        avg_trade_duration = np.mean(trade_stats['durations']) if trade_stats['durations'] else 0
        avg_position_size = total_position_size / trade_count if trade_count > 0 else 0
        avg_winning_return = np.mean([r for r in trade_stats['returns'] if r > 0]) if winning_trades > 0 else 0
        avg_losing_return = np.mean([r for r in trade_stats['returns'] if r < 0]) if losing_trades > 0 else 0
        
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
            'trade_count': trade_count,
            'win_rate': win_rate,
            'avg_trade_duration': avg_trade_duration,
            'avg_position_size': avg_position_size,
            'avg_winning_return': avg_winning_return,
            'avg_losing_return': avg_losing_return,
            'profit_factor': abs(avg_winning_return / avg_losing_return) if avg_losing_return != 0 else 0
        }
    
    def plot_results(self, strategy_returns, benchmark_returns, metrics, confidence_scores=None):
        """Plot backtest results with confidence visualization"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot strategy vs benchmark
        ax1.plot(strategy_returns, label='Strategy')
        ax1.plot(benchmark_returns, label='Benchmark (Buy & Hold)')
        ax1.set_title(f"Backtest Results\nAnnualized Return: {metrics['annualized_return']:.2%}")
        ax1.set_ylabel('Portfolio Value')
        ax1.legend()
        ax1.grid(True)
        
        # Add confidence visualization if available
        if confidence_scores is not None:
            ax2.plot(confidence_scores, color='purple', alpha=0.6, label='Confidence')
            ax2.axhline(y=0.7, color='red', linestyle='--', label='Confidence Threshold')
            ax2.set_ylabel('Confidence')
            ax2.set_xlabel('Days')
            ax2.legend()
            ax2.grid(True)
        
        # Add metrics text box
        metrics_text = "\n".join([
            f"Total Return: {metrics['total_return']:.2%}",
            f"Annualized Vol: {metrics['annualized_volatility']:.2%}",
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}",
            f"Max Drawdown: {metrics['max_drawdown']:.2%}",
            f"Trades: {metrics['trade_count']}",
            f"Win Rate: {metrics['win_rate']:.1%}",
            f"Avg Trade Duration: {metrics['avg_trade_duration']:.1f} days",
            f"Avg Position Size: {metrics['avg_position_size']:.1%}",
            f"Profit Factor: {metrics['profit_factor']:.2f}"
        ])
        ax1.text(0.02, 0.98, metrics_text, 
                transform=ax1.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

def run_backtest(df, signals, confidence_scores=None):
    """Run complete backtest with optional confidence scores"""
    backtester = Backtester()
    
    # Calculate strategy returns
    # Calculate strategy returns and get trade statistics
    strategy_returns, trade_count, winning_trades, losing_trades, total_position_size, trade_stats = (
        backtester.calculate_returns(df, signals, confidence_scores)
    )
    
    # Calculate benchmark returns (buy & hold)
    benchmark_returns = df['spy_close'].values / df['spy_close'].iloc[0] * backtester.initial_capital
    
    # Calculate performance metrics with all required parameters
    metrics = backtester.calculate_metrics(
        strategy_returns, 
        benchmark_returns, 
        trade_count,
        winning_trades,
        losing_trades,
        total_position_size,
        trade_stats
    )
    
    # Plot results
    backtester.plot_results(strategy_returns, benchmark_returns, metrics, confidence_scores)
    
    return metrics
