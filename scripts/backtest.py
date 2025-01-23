import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from data_process import load_market_data

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
            
        # Track trade statistics with dates
        trade_stats = {
            'entry_dates': [],
            'exit_dates': [],
            'entry_prices': [],
            'exit_prices': [],
            'returns': [],
            'durations': [],
            'position_sizes': [],
            'trade_dates': []  # Track dates for each trade
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
                    trade_stats['exit_prices'].append(current_price)
                    trade_stats['returns'].append(daily_return)
                    trade_stats['durations'].append(len(trade_stats['entry_dates']) - len(trade_stats['exit_dates']))
                    trade_stats['position_sizes'].append(abs(position))
                    trade_stats['trade_dates'].append({
                        'entry': trade_stats['entry_dates'][-1],
                        'exit': df.index[i]
                    })
                    
                    if daily_return > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1
                
                if target_position != 0:  # Entering a position
                    trade_stats['entry_dates'].append(df.index[i])
                    trade_stats['entry_prices'].append(current_price)
                
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
        
        # Add date-based metrics
        if trade_stats['trade_dates']:
            first_trade_date = trade_stats['trade_dates'][0]['entry']
            last_trade_date = trade_stats['trade_dates'][-1]['exit']
            trade_duration = (last_trade_date - first_trade_date).days
        else:
            first_trade_date = last_trade_date = trade_duration = None
        
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
        
        metrics = {
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
            'profit_factor': abs(avg_winning_return / avg_losing_return) if avg_losing_return != 0 else 0,
            'first_trade_date': first_trade_date,
            'last_trade_date': last_trade_date,
            'trade_duration_days': trade_duration,
            'trade_dates': trade_stats['trade_dates']
        }
        
        return metrics
    
    def create_interactive_dashboard(self, df, strategy_returns, benchmark_returns, metrics, confidence_scores=None):
        """Create interactive Plotly dashboard"""
        import dash
        from dash import dcc, html
        import dash_bootstrap_components as dbc
        import plotly.graph_objects as go
        from dash.dependencies import Input, Output

        # Create figures
        equity_fig = go.Figure()
        equity_fig.add_trace(go.Scatter(
            x=df.index,
            y=strategy_returns,
            name='Strategy',
            line=dict(color='royalblue', width=2)
        ))
        equity_fig.add_trace(go.Scatter(
            x=df.index,
            y=benchmark_returns,
            name='Benchmark',
            line=dict(color='gray', width=1, dash='dot')
        ))
        equity_fig.update_layout(
            title='Equity Curve',
            xaxis_title='Date',
            yaxis_title='Portfolio Value',
            hovermode='x unified'
        )

        # Create drawdown figure
        cumulative_max = np.maximum.accumulate(strategy_returns)
        drawdown = (cumulative_max - strategy_returns) / cumulative_max
        drawdown_fig = go.Figure()
        drawdown_fig.add_trace(go.Scatter(
            x=df.index,
            y=drawdown,
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='red', width=1),
            name='Drawdown'
        ))
        drawdown_fig.update_layout(
            title='Drawdown',
            xaxis_title='Date',
            yaxis_title='Drawdown',
            yaxis_tickformat='.1%'
        )

        # Create confidence figure if available
        if confidence_scores is not None:
            confidence_fig = go.Figure()
            confidence_fig.add_trace(go.Scatter(
                x=df.index,
                y=confidence_scores,
                name='Confidence',
                line=dict(color='purple', width=1)
            ))
            confidence_fig.add_trace(go.Scatter(
                x=df.index,
                y=[0.7]*len(confidence_scores),
                name='Threshold',
                line=dict(color='red', width=1, dash='dash')
            ))
            confidence_fig.update_layout(
                title='Confidence Scores',
                xaxis_title='Date',
                yaxis_title='Confidence'
            )

        # Create metrics cards
        metrics_cards = dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Returns"),
                dbc.CardBody([
                    html.H4(f"{metrics['total_return']:.2%}", className="card-title"),
                    html.P(f"Annualized: {metrics['annualized_return']:.2%}")
                ])
            ]), width=3),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Risk"),
                dbc.CardBody([
                    html.H4(f"{metrics['max_drawdown']:.2%}", className="card-title"),
                    html.P(f"Volatility: {metrics['annualized_volatility']:.2%}")
                ])
            ]), width=3),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Trades"),
                dbc.CardBody([
                    html.H4(f"{metrics['trade_count']}", className="card-title"),
                    html.P(f"Win Rate: {metrics['win_rate']:.1%}")
                ])
            ]), width=3),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Performance"),
                dbc.CardBody([
                    html.H4(f"{metrics['sharpe_ratio']:.2f}", className="card-title"),
                    html.P(f"Profit Factor: {metrics['profit_factor']:.2f}")
                ])
            ]), width=3)
        ])

        # Create app layout
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        app.layout = dbc.Container([
            dbc.Row(dbc.Col(html.H1("Backtest Results Dashboard"))),
            metrics_cards,
            dbc.Row(dbc.Col(dcc.Graph(figure=equity_fig))),
            dbc.Row(dbc.Col(dcc.Graph(figure=drawdown_fig))),
            dbc.Row(dbc.Col(dcc.Graph(figure=confidence_fig))) if confidence_scores is not None else None
        ], fluid=True)

        return app

    def plot_results(self, strategy_returns, benchmark_returns, metrics, confidence_scores=None):
        """Plot backtest results with interactive dashboard"""
        import threading
        import webbrowser
        from waitress import serve
        
        # Create a minimal date range index
        date_range = pd.date_range(
            start=metrics['first_trade_date'] if metrics['first_trade_date'] else pd.Timestamp.now() - pd.Timedelta(days=len(strategy_returns)),
            periods=len(strategy_returns),
            freq='D'
        )
        
        # Create dashboard with minimal data
        app = self.create_interactive_dashboard(
            pd.DataFrame(index=date_range),
            strategy_returns,
            benchmark_returns,
            metrics,
            confidence_scores
        )
        
        # Configure server for Windows
        def run_server():
            serve(app.server, host='127.0.0.1', port=8050)
            
        # Start server in a separate thread
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()
        
        # Open browser after short delay
        def open_browser():
            webbrowser.open_new('http://127.0.0.1:8050/')
            
        threading.Timer(1.0, open_browser).start()
        
        # Keep main thread alive while server runs
        try:
            while server_thread.is_alive():
                server_thread.join(1)
        except KeyboardInterrupt:
            print("\nShutting down server...")

def run_backtest(df, signals, confidence_scores=None):
    """Run complete backtest with optional confidence scores"""
    backtester = Backtester()
    
    # Calculate strategy returns
    strategy_returns, trade_count, winning_trades, losing_trades, total_position_size, trade_stats = (
        backtester.calculate_returns(df, signals, confidence_scores)
    )
    
    # Calculate benchmark returns (buy & hold)
    benchmark_returns = df['spy_close'].values / df['spy_close'].iloc[0] * backtester.initial_capital
    
    # Calculate performance metrics
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

def load_and_backtest():
    """Load predictions and run backtest"""
    try:
        # Load test predictions
        test_results = pd.read_csv('data/test_predictions.csv', index_col=0, parse_dates=True)
        
        # Load market data for the prediction period
        df = load_market_data()
        test_df = df.loc[test_results.index]
        
        # Run backtest with saved predictions
        metrics = run_backtest(
            test_df,
            test_results['Signal'],
            test_results.get('Confidence', None)
        )
        
        print("\nBacktest Metrics:")
        print(metrics)
        
        return metrics
        
    except FileNotFoundError:
        print("Error: test_predictions.csv not found. Please run model_train.py first.")
    except Exception as e:
        print(f"Error running backtest: {str(e)}")
        raise

if __name__ == "__main__":
    load_and_backtest()
