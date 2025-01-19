import pandas as pd
import numpy as np
import joblib
from train_model import fetch_data, calculate_technical_indicators
from datetime import datetime

class MLStrategy:
    def __init__(self, model, scaler, feature_columns):
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns
        
    def calculate_confidence_score(self, X_scaled):
        """Calculate confidence score based on tree agreement and prediction magnitude"""
        predictions = np.array([tree.predict(X_scaled) 
                              for tree in self.model.estimators_])
        mean_pred = predictions.mean(axis=0)
        tree_std = predictions.std(axis=0)
        agreement_score = 1 / (1 + tree_std)
        magnitude_score = np.abs(mean_pred) / np.std(mean_pred)
        confidence = (agreement_score * magnitude_score)
        confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min())
        return confidence, mean_pred
        
    def generate_signals(self, df, max_position=1.0, min_confidence=0.2):
        """Generate trading signals with position sizing based on confidence"""
        X = df[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        confidence_scores, predictions = self.calculate_confidence_score(X_scaled)
        
        signals = pd.Series(index=df.index, data=0.0)
        for i in range(len(predictions)):
            if predictions[i] > 0.001:  # Long signal
                if confidence_scores[i] > min_confidence:
                    signals.iloc[i] = max_position * confidence_scores[i]
            elif predictions[i] < -0.001:  # Short signal
                if confidence_scores[i] > min_confidence:
                    signals.iloc[i] = -max_position * confidence_scores[i]
        
        return signals, predictions, confidence_scores

def generate_predictions(df, start_date, end_date):
    """Generate predictions for specified date range"""
    # Load model artifacts
    model = joblib.load('data/model.pkl')
    scaler = joblib.load('data/scaler.pkl')
    feature_columns = joblib.load('data/feature_columns.pkl')
    
    # Filter data for specified date range
    df = df.loc[start_date:end_date]
    
    # Generate predictions
    strategy = MLStrategy(model, scaler, feature_columns)
    signals, predictions, confidence_scores = strategy.generate_signals(
        df, max_position=1.0, min_confidence=0.2
    )
    
    return signals, predictions, confidence_scores

if __name__ == "__main__":
    print("Generating price forecasts...")
    start_time = datetime.now()
    
    # Fetch and prepare data
    df = fetch_data()
    df = calculate_technical_indicators(df)
    
    # Example usage
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    signals, predictions, confidence_scores = generate_predictions(df, start_date, end_date)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'signals': signals,
        'predictions': predictions,
        'confidence_scores': confidence_scores
    }, index=df.loc[start_date:end_date].index)
    predictions_df.to_csv('data/predictions.csv')
    
    print(f"Forecast complete! Time taken: {datetime.now() - start_time}")
    print(f"Predictions saved to data/predictions.csv")
