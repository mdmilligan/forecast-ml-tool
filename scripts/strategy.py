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
        
    def calculate_confidence_score(self, X_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence score based on tree agreement and prediction magnitude
        
        Args:
            X_scaled: Scaled feature matrix
            
        Returns:
            Tuple of (confidence_scores, mean_predictions)
        """
        try:
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
            
        except Exception as e:
            logger.error(f"Error calculating confidence scores: {str(e)}")
            raise
            
    def generate_signals(self, df: pd.DataFrame, max_position: float = 1.0, 
                        base_confidence: float = 0.2) -> Tuple[pd.Series, np.ndarray, np.ndarray]:
        """
        Generate trading signals with dynamic confidence threshold
        that increases with recent trading activity
        
        Args:
            df: DataFrame containing features
            max_position: Maximum position size (0-1)
            base_confidence: Base confidence threshold
            
        Returns:
            Tuple of (signals, predictions, confidence_scores)
        """
        try:
            X = df[self.feature_columns]
            X_scaled = self.scaler.transform(X)
            confidence_scores, predictions = self.calculate_confidence_score(X_scaled)
            
            signals = pd.Series(index=df.index, data=0.0)
            trade_count = 0  # Track recent trades
            
            for i in range(len(predictions)):
                # Calculate dynamic confidence threshold
                dynamic_threshold = base_confidence * (1 + 0.1 * trade_count)
                
                if predictions[i] > 0.001:  # Long signal
                    if confidence_scores[i] > dynamic_threshold:
                        # Calculate position size reduction
                        position_reduction = 1 / (1 + 0.2 * trade_count)
                        signals.iloc[i] = max_position * confidence_scores[i] * position_reduction
                        trade_count += 1
                elif predictions[i] < -0.001:  # Short signal
                    if confidence_scores[i] > dynamic_threshold:
                        position_reduction = 1 / (1 + 0.2 * trade_count)
                        signals.iloc[i] = -max_position * confidence_scores[i] * position_reduction
                        trade_count += 1
                
                # Decay trade count over time
                if i % 5 == 0:  # Every 5 periods
                    trade_count = max(0, trade_count - 1)
            
            return signals, predictions, confidence_scores
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise
