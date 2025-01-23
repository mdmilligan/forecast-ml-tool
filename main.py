#!/usr/bin/env python3
"""
Main entry point for the trading system
"""
import argparse
import logging
import sys
from scripts.data_process import load_market_data
from scripts.model_train import train_model
from scripts.model_forecast import generate_predictions
from scripts.backtest import run_backtest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Trading System')
    parser.add_argument('--start-date', type=str, default='2013-01-01',
                       help='Start date for training data (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-12-31',
                       help='End date for testing data (YYYY-MM-DD)')
    parser.add_argument('--train', action='store_true',
                       help='Train new model')
    parser.add_argument('--backtest', action='store_true',
                       help='Run backtest')
    parser.add_argument('--predict', action='store_true',
                       help='Make prediction for next trading day')
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    
    # Load market data
    df = load_market_data(args.start_date, args.end_date)
    
    if args.train:
        train_model(df)
        
    if args.backtest:
        from scripts.backtest import load_and_backtest
        load_and_backtest()
        
    if args.predict:
        generate_predictions(df)
        
    if not any([args.train, args.backtest, args.predict]):
        logger.info("No action specified. Use --help for usage information.")

if __name__ == "__main__":
    main()
