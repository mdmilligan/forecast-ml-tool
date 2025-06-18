import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from data_process import load_market_data
import config

class Backtester:
    def __init__(self, initial_capital=config.INITIAL_CAPITAL, commission=config.COMMISSION):
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
        
        # Add entry and exit markers for plotting
        metrics['entry_markers'] = [{'date': d, 'price': p} for d, p in zip(trade_stats['entry_dates'], trade_stats['entry_prices'])]
        metrics['exit_markers'] = [{'date': d, 'price': p} for d, p in zip(trade_stats['exit_dates'], trade_stats['exit_prices'])]

        return metrics
    
    def create_interactive_dashboard(self, plot_index_df, full_market_data_df, strategy_returns, benchmark_returns, metrics, confidence_scores=None):
        """Create interactive Plotly dashboard"""
        import dash
        from dash import dcc, html
        import dash_bootstrap_components as dbc
        import plotly.graph_objects as go
        from dash.dependencies import Input, Output

        # Create figures
        equity_fig = go.Figure()
        equity_fig.add_trace(go.Scatter(
            x=plot_index_df.index,
            y=strategy_returns,
            name='Strategy',
            line=dict(color='royalblue', width=2)
        ))
        equity_fig.add_trace(go.Scatter(
            x=plot_index_df.index,
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
        drawdown_calc = (cumulative_max - strategy_returns) / cumulative_max # Renamed to avoid conflict
        drawdown_fig = go.Figure()
        drawdown_fig.add_trace(go.Scatter(
            x=plot_index_df.index,
            y=drawdown_calc,
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
        confidence_fig = None # Initialize to None
        if confidence_scores is not None:
            # Ensure confidence_scores aligns with plot_index_df.index for plotting
            # This might involve slicing or reindexing if lengths/indices differ.
            # For this example, we assume confidence_scores is a NumPy array or list
            # that is already aligned or will be correctly handled by Plotly's broadcasting.
            # A safer approach would be to explicitly align it:
            # if isinstance(confidence_scores, pd.Series):
            #    confidence_scores_plot = confidence_scores.reindex(plot_index_df.index).values
            # elif len(confidence_scores) == len(full_market_data_df.index) and len(plot_index_df.index) == len(full_market_data_df.index) -1:
            #    confidence_scores_plot = confidence_scores[1:len(plot_index_df.index)+1] # Align if it matches full data and returns are one shorter
            # else:
            #    confidence_scores_plot = confidence_scores # Hope for the best or ensure it's pre-aligned
            
            # Simplified: assume confidence_scores is passed correctly aligned for plot_index_df
            # or that its length matches plot_index_df.index
            # If confidence_scores is a pd.Series, it's better to pass .values
            
            # Let's try to align confidence_scores if it's a numpy array and its length matches full_market_data_df
            # and strategy_returns is one shorter (common pattern)
            confidence_scores_for_plot = confidence_scores
            if isinstance(confidence_scores, (np.ndarray, list)) and \
               len(confidence_scores) == len(full_market_data_df.index) and \
               len(plot_index_df.index) == len(full_market_data_df.index) - 1:
                confidence_scores_for_plot = confidence_scores[1:]
            elif isinstance(confidence_scores, (np.ndarray, list)) and \
                 len(confidence_scores) != len(plot_index_df.index):
                # If lengths still don't match, it's problematic.
                # For now, we'll proceed, but this could lead to plotting issues.
                # Consider logging a warning or handling this case more robustly.
                pass


            confidence_fig = go.Figure()
            confidence_fig.add_trace(go.Scatter(
                x=plot_index_df.index, 
                y=confidence_scores_for_plot, # Use the potentially aligned version
                name='Confidence',
                line=dict(color='purple', width=1)
            ))
            confidence_fig.add_trace(go.Scatter(
                x=plot_index_df.index, 
                y=[0.7]*len(plot_index_df.index), # Threshold should match the x-axis length
                name='Threshold',
                line=dict(color='red', width=1, dash='dash')
            ))
            confidence_fig.update_layout(
                title='Confidence Scores',
                xaxis_title='Date',
                yaxis_title='Confidence'
            )

        # Create trade markers plot
        trade_markers_fig = go.Figure()
        trade_markers_fig.add_trace(go.Scatter(
            x=full_market_data_df.index,
            y=full_market_data_df['spy_close'], # Assuming 'spy_close' is the relevant price
            name='Price',
            line=dict(color='skyblue', width=1)
        ))

        entry_dates = [marker['date'] for marker in metrics.get('entry_markers', [])]
        entry_prices = [marker['price'] for marker in metrics.get('entry_markers', [])]
        trade_markers_fig.add_trace(go.Scatter(
            x=entry_dates,
            y=entry_prices,
            mode='markers',
            name='Entry',
            marker=dict(color='green', symbol='triangle-up', size=10),
            text=[f"Entry: {d.strftime('%Y-%m-%d')}<br>Price: {p:.2f}" for d, p in zip(entry_dates, entry_prices)],
            hoverinfo='text'
        ))

        exit_dates = [marker['date'] for marker in metrics.get('exit_markers', [])]
        exit_prices = [marker['price'] for marker in metrics.get('exit_markers', [])]
        trade_markers_fig.add_trace(go.Scatter(
            x=exit_dates,
            y=exit_prices,
            mode='markers',
            name='Exit',
            marker=dict(color='red', symbol='triangle-down', size=10),
            text=[f"Exit: {d.strftime('%Y-%m-%d')}<br>Price: {p:.2f}" for d, p in zip(exit_dates, exit_prices)],
            hoverinfo='text'
        ))

        trade_markers_fig.update_layout(
            title='Price Chart with Trade Markers',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='closest' # 'closest' is often better for scatter with tooltips
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
        
        layout_components = [
            dbc.Row(dbc.Col(html.H1("Backtest Results Dashboard"))),
            metrics_cards,
            dbc.Row(dbc.Col(dcc.Graph(figure=equity_fig))),
            dbc.Row(dbc.Col(dcc.Graph(figure=drawdown_fig))),
            dbc.Row(dbc.Col(dcc.Graph(figure=trade_markers_fig))),
        ]
        
        if confidence_fig: # Will be None if confidence_scores was None or alignment failed
            layout_components.append(dbc.Row(dbc.Col(dcc.Graph(figure=confidence_fig))))
        
        app.layout = dbc.Container(layout_components, fluid=True)

        return app

    def plot_results(self, df, strategy_returns, benchmark_returns, metrics, confidence_scores=None):
        """Plot backtest results with interactive dashboard"""
        import threading
        import webbrowser
        from waitress import serve
        
        # Ensure strategy_returns is a NumPy array for consistent processing
        if not isinstance(strategy_returns, np.ndarray):
            strategy_returns = np.array(strategy_returns)

        # Determine date_range for plots based on strategy_returns
        # This plot_index_df is primarily for equity curve, drawdown, and potentially confidence scores
        if len(strategy_returns) > 0:
            # If first_trade_date is available and makes sense for the start of returns
            start_plot_date = metrics.get('first_trade_date')
            # strategy_returns typically starts from the first change, so df might be one longer
            # If df.index[0] is the actual start of the data period for returns, use it.
            # This needs careful alignment. Let's assume df.index aligns with the period of signals.
            # strategy_returns are usually one shorter than the price series they were derived from.
            
            # If we have trade dates, try to use them.
            # The number of periods for date_range should match len(strategy_returns)
            if metrics.get('first_trade_date') and metrics.get('last_trade_date') and len(strategy_returns) > 0:
                # Try to create a date range based on actual trade activity if possible,
                # but ensure it has the same number of points as strategy_returns.
                # This can be tricky if trade dates don't perfectly span the returns.
                # A simpler approach: use the relevant slice of df.index that corresponds to strategy_returns.
                # If strategy_returns is derived from df[1:] vs df[:-1], then df.index[1:] is appropriate.
                if len(df.index) == len(strategy_returns) + 1:
                    plot_index_for_returns = df.index[1:]
                elif len(df.index) == len(strategy_returns): # If already aligned
                     plot_index_for_returns = df.index
                else: # Fallback or error: lengths don't match expected pattern
                    # Default to a generic range if alignment is unclear
                    if metrics.get('first_trade_date'):
                        plot_index_for_returns = pd.date_range(start=metrics['first_trade_date'], periods=len(strategy_returns), freq=pd.infer_freq(df.index) or 'B')
                    else:
                        plot_index_for_returns = pd.date_range(start=df.index[0], periods=len(strategy_returns), freq=pd.infer_freq(df.index) or 'B')

            elif len(strategy_returns) > 0 : # If no specific trade dates, but returns exist
                 if len(df.index) == len(strategy_returns) + 1:
                    plot_index_for_returns = df.index[1:]
                 elif len(df.index) == len(strategy_returns):
                     plot_index_for_returns = df.index
                 else: # Fallback: create a generic index
                    plot_index_for_returns = pd.RangeIndex(start=0, stop=len(strategy_returns), step=1)

            else: # No strategy returns, create a dummy index for dashboard structure
                plot_index_for_returns = pd.to_datetime(['2000-01-01']) if len(df.index) == 0 else df.index[:1]
        else: # No strategy returns
             plot_index_for_returns = pd.to_datetime(['2000-01-01']) if len(df.index) == 0 else df.index[:1]


        plot_index_df = pd.DataFrame(index=plot_index_for_returns)
        
        # Prepare confidence_scores for the dashboard
        # It should align with plot_index_df
        final_confidence_scores_for_plot = None
        if confidence_scores is not None:
            if isinstance(confidence_scores, pd.Series):
                # If it's a series, try to reindex to the plot_index_df's index
                # This assumes confidence_scores.index is compatible (e.g., datetime)
                try:
                    final_confidence_scores_for_plot = confidence_scores.reindex(plot_index_df.index).values
                except Exception: # Fallback if reindexing fails
                    if len(confidence_scores.values) == len(plot_index_df.index):
                        final_confidence_scores_for_plot = confidence_scores.values
                    elif len(confidence_scores.values) == len(df.index) and len(plot_index_df.index) == len(df.index) -1:
                         final_confidence_scores_for_plot = confidence_scores.values[1:]

            elif isinstance(confidence_scores, (np.ndarray, list)):
                if len(confidence_scores) == len(plot_index_df.index):
                    final_confidence_scores_for_plot = np.array(confidence_scores)
                elif len(confidence_scores) == len(df.index) and len(plot_index_df.index) == len(df.index) -1 : # Common case for returns
                    final_confidence_scores_for_plot = np.array(confidence_scores[1:])
                # If still not aligned, it might be an issue, plot as is or None
                # else: final_confidence_scores_for_plot = None # Or log warning

        # Create dashboard
        app = self.create_interactive_dashboard(
            plot_index_df,    # DataFrame with index for equity, drawdown, confidence plots
            df,               # Full market data DataFrame for price chart with trade markers
            strategy_returns,
            benchmark_returns,
            metrics,
            final_confidence_scores_for_plot # Pass the aligned/processed confidence scores
        )
        
        # Configure server
        def run_server():
            serve(app.server, host='127.0.0.1', port=8050)
            
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()
        
        def open_browser():
            webbrowser.open_new('http://127.0.0.1:8050/')
            
        threading.Timer(1.0, open_browser).start()
        
        try:
            while server_thread.is_alive():
                server_thread.join(1)
        except KeyboardInterrupt:
            print("\nShutting down server...")

def run_backtest(df, signals, confidence_scores=None):
    """Run complete backtest with optional confidence scores"""
    backtester = Backtester()
    
    # Ensure signals is a pd.Series for .iloc access, if it's not already
    if not isinstance(signals, pd.Series):
        signals = pd.Series(signals, index=df.index[:len(signals)]) # Align with df index

    # Ensure confidence_scores is a numpy array or list if provided, or None
    # The calculate_returns function expects confidence_scores to be indexable like an array/list
    # or be None. If it's a pd.Series, convert to numpy array.
    if isinstance(confidence_scores, pd.Series):
        confidence_scores_input = confidence_scores.values
    else:
        confidence_scores_input = confidence_scores # Can be None or np.array/list

    strategy_returns, trade_count, winning_trades, losing_trades, total_position_size, trade_stats = (
        backtester.calculate_returns(df, signals, confidence_scores_input)
    )
    
    benchmark_returns = df['spy_close'].values / df['spy_close'].iloc[0] * backtester.initial_capital
    
    metrics = backtester.calculate_metrics(
        strategy_returns, 
        benchmark_returns, 
        trade_count,
        winning_trades,
        losing_trades,
        total_position_size,
        trade_stats
    )
    
    # Pass the original confidence_scores (as Series or array) to plot_results, 
    # it will handle alignment for plotting.
    backtester.plot_results(df, strategy_returns, benchmark_returns, metrics, confidence_scores) 
    
    return metrics

def load_and_backtest():
    """Load predictions and run backtest"""
    try:
        test_results = pd.read_csv('data/test_predictions.csv', index_col=0, parse_dates=True)
        
        df_market = load_market_data() # Renamed to avoid conflict with 'df' in some scopes
        
        # Ensure indices are compatible for slicing (e.g., timezone awareness)
        if df_market.index.tz is not None and test_results.index.tz is None:
            test_results.index = test_results.index.tz_localize(df_market.index.tz)
        elif df_market.index.tz is None and test_results.index.tz is not None:
            # Making test_results timezone naive to match df_market if df_market is naive
            test_results.index = test_results.index.tz_localize(None)
        
        # Intersect indices to ensure we only use dates present in both DataFrames
        common_index = df_market.index.intersection(test_results.index)
        if common_index.empty:
            print("Error: No common dates between market data and test predictions.")
            print(f"Market data index range: {df_market.index.min()} to {df_market.index.max()}")
            print(f"Test results index range: {test_results.index.min()} to {test_results.index.max()}")
            return None

        test_df = df_market.loc[common_index]
        test_results_aligned = test_results.loc[common_index]

        signals_input = test_results_aligned['Signal']
        confidence_input = test_results_aligned.get('Confidence', None) # This will be a Series or None
        
        metrics_output = run_backtest(
            test_df,
            signals_input, # pd.Series
            confidence_input # pd.Series or None
        )
        
        if metrics_output:
            print("\nBacktest Metrics:")
            for k, v in metrics_output.items():
                if isinstance(v, list) and (k.endswith('_markers') or k == 'trade_dates'):
                    print(f"{k}: (data list)")
                elif isinstance(v, pd.Timestamp):
                     print(f"{k}: {v.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    print(f"{k}: {v}")
        
        return metrics_output
        
    except FileNotFoundError:
        print("Error: data/test_predictions.csv not found. Please run model_train.py first.")
    except KeyError as e:
        print(f"KeyError during data processing: {e}. Check column names in CSV and data loading.")
        raise
    except Exception as e:
        print(f"Error running backtest: {str(e)}")
        raise

if __name__ == "__main__":
    load_and_backtest()
