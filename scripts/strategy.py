import numpy as np
import pandas as pd
import pickle
import warnings
import os
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
        self.current_position = None
        
    def calculate_confidence_score(self, X_input) -> np.ndarray:
        """
        Calculate confidence score based on prediction variance.
        X_input may be a numpy array or pandas DataFrame; pass-through to model.predict.
        """
        # Get predictions from model using the same input format passed to predict
        preds = self.model.predict(X_input)
        # Calculate confidence using prediction magnitude; guard against division by zero
        max_abs = np.max(np.abs(preds))
        confidence = np.abs(preds) / max_abs if max_abs != 0 else np.zeros_like(preds)
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
        # Load expected feature list from models/feature_columns.pkl to validate inference inputs
        try:
            with open(os.path.join('models', 'feature_columns.pkl'), 'rb') as f:
                expected = pickle.load(f)
        except Exception:
            # Fall back to feature_columns passed at init if loading fails
            expected = list(self.feature_columns) if self.feature_columns is not None else []

        expected = list(expected)

        # Ensure input is a DataFrame
        if isinstance(df, np.ndarray):
            # No column names available on numpy array -> create DataFrame without names
            X = pd.DataFrame(df)
        elif isinstance(df, pd.DataFrame):
            X = df.copy()
        else:
            # Try to coerce to DataFrame
            X = pd.DataFrame(df)

        # Compare expected vs current
        current_cols = list(X.columns)
        missing = [c for c in expected if c not in current_cols]
        extra = [c for c in current_cols if c not in expected]

        if missing:
            # Explicit failure: do not silently fill missing features
            raise ValueError(f"Missing feature columns required for model inference: {missing}")

        if extra:
            warnings.warn(f"Extra columns detected in input and will be dropped: {extra}", UserWarning)
            # Drop extras and preserve expected order below
            X = X.drop(columns=extra, errors='ignore')

        # Reindex to expected order (this will also ensure correct column order)
        X = X.reindex(columns=expected)

        # If scaler is present, apply it ensuring columns are in the expected order
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            # If scaler returns numpy array, convert back to DataFrame to preserve column names for model.predict
            if isinstance(X_scaled, np.ndarray):
                X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=expected)
            else:
                X_scaled_df = X_scaled
        else:
            X_scaled_df = X

        logger.info("Feature validation OK — using features: %s", expected)

        # Get predictions from the model using DataFrame with correct feature names to avoid sklearn warnings
        predictions = self.model.predict(X_scaled_df)
        if predictions.ndim == 1:
            # Single-output model: only returns available
            returns = predictions
            exit_long = np.zeros_like(predictions)
            exit_short = np.zeros_like(predictions)
        else:
            # Multi-output model: unpack return and exit signals
            returns, exit_long, exit_short = predictions[:, 0], predictions[:, 1], predictions[:, 2]
        # Calculate confidence scores using same input to model.predict for consistency
        confidence_scores = self.calculate_confidence_score(X_scaled_df)

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
