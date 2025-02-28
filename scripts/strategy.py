import numpy as np
import pandas as pd
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class MLStrategy:
    def __init__(self, model, scaler, feature_columns):
        """
        Initialize ML-based trading strategy
        
        Args:
            model: Trained machine learning model
            scaler: Feature scaler used during training
            feature_columns: List of feature column names
        """
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.current_position = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        
    def calculate_confidence_score(self, X_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence score based on tree agreement and prediction magnitude
        
        Args:
            X_scaled: Scaled feature matrix
            
        Returns:
            Tuple of (confidence_scores, mean_predictions)
        """
        try:
            # Get predictions from LightGBM
            mean_pred = self.model.predict(X_scaled)
            
            # Calculate confidence as normalized absolute predictions
            confidence = np.abs(mean_pred)
            max_conf = np.max(confidence)
            if max_conf:
                confidence = confidence / max_conf
            else:
                confidence = np.zeros_like(confidence)
            
            return confidence, mean_pred
            
        except Exception as e:
            logger.error(f"Error calculating confidence scores: {str(e)}")
            raise
            
    def generate_signals(self, df: pd.DataFrame, max_position: float = 1.0, 
                        min_confidence: float = 0.2, trade_decay: float = 0.9) -> Tuple[pd.Series, np.ndarray, np.ndarray]:
        """
        Generate trading signals with dynamic position sizing based on recent trading activity
        
        Args:
            df: DataFrame containing features
            max_position: Maximum position size (0-1)
            min_confidence: Minimum confidence threshold
            trade_decay: Decay rate for recent trade count (0-1)
            
        Returns:
            Tuple of (signals, predictions, confidence_scores)
        """
        try:
            X = df[self.feature_columns]
            X_scaled = self.scaler.transform(X)
            
            # Get predictions (returns, exit_long, exit_short)
            predictions = self.model.predict(X_scaled)
            returns, exit_long, exit_short = predictions[:,0], predictions[:,1], predictions[:,2]
            confidence_scores, _ = self.calculate_confidence_score(X_scaled)
            
            signals = pd.Series(index=df.index, data=0.0)
            recent_trade_count = 0  # Track recent trading activity
            
            for i in range(len(returns)):
                # Check for exit conditions
                if self.current_position == 1 and (exit_long[i] > 0.5 or df['spy_low'].iloc[i] <= self.stop_loss):
                    signals.iloc[i] = 0  # Close long position
                    self.current_position = None
                elif self.current_position == -1 and (exit_short[i] > 0.5 or df['spy_high'].iloc[i] >= self.stop_loss):
                    signals.iloc[i] = 0  # Close short position
                    self.current_position = None
                
                # Generate new signals if no position
                if self.current_position is None:
                    position_reduction = 1 / (1 + 0.2 * recent_trade_count)
                    
                    if returns[i] > 0.001:  # Long signal
                        if confidence_scores[i] > min_confidence:
                            signals.iloc[i] = max_position * confidence_scores[i] * position_reduction
                            self.current_position = 1
                            self.entry_price = df['spy_close'].iloc[i]
                            self.stop_loss = df['stop_loss_long'].iloc[i]
                            self.take_profit = df['take_profit_long'].iloc[i]
                            recent_trade_count += 1
                    elif returns[i] < -0.001:  # Short signal
                        if confidence_scores[i] > min_confidence:
                            signals.iloc[i] = -max_position * confidence_scores[i] * position_reduction
                            self.current_position = -1
                            self.entry_price = df['spy_close'].iloc[i]
                            self.stop_loss = df['stop_loss_short'].iloc[i]
                            self.take_profit = df['take_profit_short'].iloc[i]
                            recent_trade_count += 1
                
                # Decay recent trade count over time
                recent_trade_count *= trade_decay
                recent_trade_count = max(0, recent_trade_count)  # Ensure non-negative
                
            return signals, returns, confidence_scores
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise
