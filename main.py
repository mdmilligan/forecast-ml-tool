import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sqlite3

def fetch_data(start_date='2010-01-01', end_date='2024-01-01'):
    """
    Fetch historical data for SPY, VIX and UUP from database
    """
    # Connect to database
    conn = sqlite3.connect('data/marketdata.db')
    
    # Read data from database
    spy = pd.read_sql("SELECT * FROM stock_data_spy WHERE date >= ? AND date <= ?", 
                     conn, params=(start_date, end_date), parse_dates=['date'], index_col='date')
    vix = pd.read_sql("SELECT * FROM stock_data_vix WHERE date >= ? AND date <= ?", 
                     conn, params=(start_date, end_date), parse_dates=['date'], index_col='date')
    uup = pd.read_sql("SELECT * FROM stock_data_uup WHERE date >= ? AND date <= ?", 
                     conn, params=(start_date, end_date), parse_dates=['date'], index_col='date')
    
    # Create main dataframe with SPY data
    df = pd.DataFrame(index=spy.index)
    
    # Add basic price and volume data
    df['spy_close'] = spy['close']
    df['spy_open'] = spy['open']
    df['spy_high'] = spy['high']
    df['spy_low'] = spy['low']
    df['spy_volume'] = spy['volume']
    df['vix_close'] = vix['close']
    df['uup_close'] = uup['close']
    
    # Calculate technical indicators
    # Moving averages
    df['sma_20'] = df['spy_close'].rolling(window=20).mean()
    df['sma_50'] = df['spy_close'].rolling(window=50).mean()
    
    # RSI
    delta = df['spy_close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['spy_close'].ewm(span=12, adjust=False).mean()
    exp2 = df['spy_close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['spy_close'].rolling(window=20).mean()
    df['bb_upper'] = df['bb_middle'] + 2 * df['spy_close'].rolling(window=20).std()
    df['bb_lower'] = df['bb_middle'] - 2 * df['spy_close'].rolling(window=20).std()
    
    # Volatility
    df['volatility'] = df['spy_close'].rolling(window=20).std()
    
    # Add target variable (next day's return)
    df['target'] = df['spy_close'].shift(-1) / df['spy_close'] - 1
    
    return df.dropna()

def prepare_features(df):
    """
    Prepare features for model training
    """
    feature_columns = [
        'spy_close', 'spy_volume', 'vix_close', 'uup_close',
        'sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal',
        'bb_upper', 'bb_lower', 'volatility'
    ]
    
    X = df[feature_columns]
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_columns

class MLStrategy:
    def __init__(self, model, scaler, feature_columns):
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns
        
    def calculate_confidence_score(self, X_scaled):
        """
        Calculate confidence score based on:
        1. Prediction magnitude
        2. Tree agreement in Random Forest
        3. Historical volatility
        """
        # Get predictions from all trees
        predictions = np.array([tree.predict(X_scaled) 
                              for tree in self.model.estimators_])
        
        # Calculate base prediction
        mean_pred = predictions.mean(axis=0)
        
        # Calculate tree agreement (lower std = higher agreement)
        tree_std = predictions.std(axis=0)
        agreement_score = 1 / (1 + tree_std)
        
        # Normalize prediction magnitude
        magnitude_score = np.abs(mean_pred) / np.std(mean_pred)
        
        # Combine scores
        confidence = (agreement_score * magnitude_score)
        
        # Normalize to 0-1 range
        confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min())
        
        return confidence, mean_pred
        
    def generate_signals(self, df, max_position=1.0, min_confidence=0.2):
        """
        Generate trading signals with position sizing based on confidence
        """
        X = df[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        # Get confidence scores and predictions
        confidence_scores, predictions = self.calculate_confidence_score(X_scaled)
        
        # Generate base signals
        signals = pd.Series(index=df.index, data=0.0)
        
        # Apply position sizing based on confidence
        for i in range(len(predictions)):
            if predictions[i] > 0.001:  # Long signal
                if confidence_scores[i] > min_confidence:
                    signals.iloc[i] = max_position * confidence_scores[i]
            elif predictions[i] < -0.001:  # Short signal
                if confidence_scores[i] > min_confidence:
                    signals.iloc[i] = -max_position * confidence_scores[i]
        
        return signals, predictions, confidence_scores

def backtest_strategy(df, signals, confidence_scores):
    """
    Enhanced backtest with position sizing
    """
    # Calculate daily returns
    daily_returns = df['spy_close'].pct_change()
    
    # Calculate strategy returns with position sizing
    strategy_returns = signals.shift(1) * daily_returns
    
    # Calculate cumulative returns
    cumulative_market_returns = (1 + daily_returns).cumprod()
    cumulative_strategy_returns = (1 + strategy_returns).cumprod()
    
    # Calculate metrics
    total_trades = np.sum(np.abs(signals.diff() != 0))
    winning_trades = np.sum((strategy_returns > 0) & (signals.shift(1) != 0))
    losing_trades = np.sum((strategy_returns < 0) & (signals.shift(1) != 0))
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Calculate average position size
    avg_position_size = np.mean(np.abs(signals[signals != 0]))
    
    # Calculate Sharpe Ratio (assuming risk-free rate of 2%)
    excess_returns = strategy_returns - 0.02/252
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    # Maximum drawdown
    portfolio_value = cumulative_strategy_returns
    rolling_max = portfolio_value.expanding().max()
    drawdowns = (portfolio_value - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    return {
        'cumulative_market_returns': cumulative_market_returns,
        'cumulative_strategy_returns': cumulative_strategy_returns,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'daily_returns': strategy_returns,
        'avg_position_size': avg_position_size
    }

def plot_results(df, backtest_results, predictions, confidence_scores, signals, model, feature_columns):
    """
    Enhanced plotting with position sizing visualization
    """
    fig = plt.figure(figsize=(15, 15))
    
    # 1. Price and Position Size
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(df.index, df['spy_close'], label='SPY Price')
    ax1_twin = ax1.twinx()
    ax1_twin.fill_between(df.index, signals, alpha=0.3, label='Position Size')
    ax1.set_title('SPY Price and Position Size')
    ax1.set_ylabel('Price ($)')
    ax1_twin.set_ylabel('Position Size')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True)
    
    # 2. Cumulative Returns Comparison
    ax2 = plt.subplot(4, 1, 2)
    ax2.plot(backtest_results['cumulative_market_returns'], 
             label='Buy & Hold Returns')
    ax2.plot(backtest_results['cumulative_strategy_returns'], 
             label='Strategy Returns')
    ax2.set_title('Cumulative Returns Comparison')
    ax2.set_ylabel('Cumulative Returns')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Predictions and Confidence
    ax3 = plt.subplot(4, 1, 3)
    scatter = ax3.scatter(df.index, predictions, 
                         c=confidence_scores, cmap='viridis', 
                         alpha=0.6, label='Predictions')
    plt.colorbar(scatter, label='Confidence Score')
    ax3.set_title('Predictions with Confidence Scores')
    ax3.set_ylabel('Predicted Return')
    ax3.grid(True)
    
    # 4. Feature Importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    ax4 = plt.subplot(4, 1, 4)
    ax4.barh(feature_importance['feature'], feature_importance['importance'])
    ax4.set_title('Feature Importance')
    ax4.set_xlabel('Importance')
    
    plt.tight_layout()
    plt.show()

def main():
    # Fetch and prepare data
    df = fetch_data()
    X, y, scaler, feature_columns = prepare_features(df)
    
    # Split data into training and testing sets
    train_size = int(len(df) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Create strategy
    strategy = MLStrategy(model, scaler, feature_columns)
    
    # Generate signals and run backtest
    signals, predictions, confidence_scores = strategy.generate_signals(
        df, max_position=1.0, min_confidence=0.2
    )
    backtest_results = backtest_strategy(df, signals, confidence_scores)
    
    # Print backtest results
    print("\nBacktest Results:")
    print(f"Total Trades: {backtest_results['total_trades']}")
    print(f"Win Rate: {backtest_results['win_rate']:.2%}")
    print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {backtest_results['max_drawdown']:.2%}")
    print(f"Average Position Size: {backtest_results['avg_position_size']:.2%}")
    
    # Plot results
    plot_results(df, backtest_results, predictions, confidence_scores, signals, model, feature_columns)
    
    # Make prediction for next day
    latest_data = df[feature_columns].iloc[-1:]
    X_latest = scaler.transform(latest_data)
    confidence, prediction = strategy.calculate_confidence_score(X_latest)
    print(f"\nNext day prediction: {prediction[0]:.4%}")
    print(f"Confidence score: {confidence[0]:.2%}")
    print(f"Recommended position size: {confidence[0] * np.sign(prediction[0]):.2%}")

if __name__ == "__main__":
    main()
