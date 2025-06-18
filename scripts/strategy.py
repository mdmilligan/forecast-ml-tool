import numpy as np
import pandas as pd
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class MLStrategy:
    def __init__(self, model, scaler, feature_columns, min_hold_bars=3):
        """
        Initialize ML-based trading strategy.
        
        Args:
            model: Trained machine learning model.
            scaler: Feature scaler used during training.
            feature_columns: List of feature column names.
            min_hold_bars: Minimum number of bars to hold a signal.
        """
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.min_hold_bars = min_hold_bars
        self.last_signal_time = None
        self.current_position = 0
        
    def calculate_confidence_score(self, X_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence score based on prediction variance"""
        # Get predictions from LightGBM
        predictions = self.model.predict(X_scaled)
        
        # Calculate confidence using prediction magnitude
        confidence = np.abs(predictions) / np.max(np.abs(predictions))
        return confidence

    def generate_signals(self, df: pd.DataFrame, max_position: float = 1.0,
                        min_confidence: float = 0.2, trade_decay: float = 0.9) -> Tuple[pd.Series, np.ndarray, np.ndarray]:
        """
        Generate trading signals based on model predictions.
        Expects the model to output three columns:
          - returns (col 0)
          - exit_long trigger (col 1)
          - exit_short trigger (col 2)
        """
        X = df[self.feature_columns]
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)
        if predictions.ndim == 1:
            # Single-output model: only returns available
            returns = predictions
            exit_long = np.zeros_like(predictions)
            exit_short = np.zeros_like(predictions)
        else:
            # Multi-output model: unpack return and exit signals
            returns, exit_long, exit_short = predictions[:, 0], predictions[:, 1], predictions[:, 2]
        confidence_scores = self.calculate_confidence_score(X_scaled)

        signals = pd.Series(index=df.index, data=0.0)
        recent_trade_count = 0

        for i in range(len(returns)):
            # Check for exit conditions based on stop-loss details in the DataFrame
            if self.current_position == 1 and (exit_long[i] > 0.5 or df['spy_low'].iloc[i] <= df['stop_loss_long'].iloc[i]):
                signals.iloc[i] = 0
                self.current_position = None
            elif self.current_position == -1 and (exit_short[i] > 0.5 or df['spy_high'].iloc[i] >= df['stop_loss_short'].iloc[i]):
                signals.iloc[i] = 0
                self.current_position = None

            if self.current_position is None:
                position_reduction = 1 / (1 + 0.2 * recent_trade_count)
                if returns[i] > 0.001:
                    if confidence_scores[i] > min_confidence:
                        signals.iloc[i] = max_position * confidence_scores[i] * position_reduction
                        self.current_position = 1
                        recent_trade_count += 1
                elif returns[i] < -0.001:
                    if confidence_scores[i] > min_confidence:
                        signals.iloc[i] = -max_position * confidence_scores[i] * position_reduction
                        self.current_position = -1
                        recent_trade_count += 1
            recent_trade_count *= trade_decay
            recent_trade_count = max(0, recent_trade_count)
        return signals, returns, confidence_scores
