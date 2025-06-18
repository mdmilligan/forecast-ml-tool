import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from data_process import load_market_data, calculate_technical_indicators
import config
from strategy import MLStrategy


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
        df, max_position=1.0, min_confidence=config.MIN_CONFIDENCE
    )
    
    # Return signals, predictions, confidence scores, and timestamps
    return signals, predictions, confidence_scores, df.index

if __name__ == "__main__":
    print("Generating price forecasts...")
    start_time = datetime.now()

    # Example usage
    start_date = config.START_DATE
    end_date = config.END_DATE

    # Load data from the database
    df = load_market_data(start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        raise ValueError("No data loaded. Check database and parameters.")

    df = calculate_technical_indicators(df)

    try:
        signals, predictions, confidence_scores, timestamps = generate_predictions(df, start_date, end_date)
    except Exception as e:
        print(f"Error during prediction generation: {str(e)}")
        raise

    # Ensure timestamps are aligned with predictions
    if len(timestamps) != len(signals):
        raise ValueError(
            f"Timestamp mismatch: {len(timestamps)} timestamps, {len(signals)} signals. "
            "Check data processing and prediction steps."
        )

    # Save minimal prediction data
    predictions_df = pd.DataFrame({
        'timestamp': timestamps,
        'signal': signals,
        'prediction': predictions,
        'confidence': confidence_scores
    })
    
    try:
        feature_columns = joblib.load('models/feature_columns.pkl')
        # Save features separately
        features_df = df[feature_columns].loc[start_date:end_date]
        features_df.to_csv('data/test_features.csv', index=True)
    except FileNotFoundError as e:
        print("Feature columns file not found, skipping feature saving.")
    except KeyError as e:
        print("Some feature columns not found in DataFrame, skipping feature saving.")

    predictions_df.to_csv('data/test_predictions.csv', index=False)

    print(f"Forecast complete! Time taken: {datetime.now() - start_time}")
    print(f"Predictions saved to data/test_predictions.csv")
