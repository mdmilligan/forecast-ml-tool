import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from scripts.data_fetch import fetch_data
from scripts.data_process import calculate_technical_indicators
from scripts.strategy import MLStrategy

def generate_predictions(df, start_date, end_date):
    """Generate predictions for specified date range"""
    try:
        # Load model artifacts
        model = joblib.load('models/model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Model files not found. Please ensure you have trained the model first. "
            f"Required files: models/model.pkl, models/scaler.pkl, models/feature_columns.pkl. Error: {str(e)}"
        )
    
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
