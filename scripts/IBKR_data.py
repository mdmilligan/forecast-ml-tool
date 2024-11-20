from ib_insync import *
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def test_historical_data():
    try:
        print("Connecting to IBGateway...")
        ib = IB()
        ib.connect('127.0.0.1', 4001, clientId=123)
        
        # Test with SPY
        contract = Stock('SPY', 'SMART', 'USD')
        print("\nRequesting 1 day of 1-minute data for SPY...")
        
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',  # Empty string means 'now'
            durationStr='1 D',  # Request 1 day of data
            barSizeSetting='1 min',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        
        if bars:
            df = util.df(bars)  # Convert to pandas DataFrame
            print("\nData sample:")
            print(df.head())
            print(f"\nTotal bars received: {len(df)}")
            print(f"Time range: {df['date'].min()} to {df['date'].max()}")
        else:
            print("No data received")
            
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Error type:", type(e).__name__)
        
    finally:
        if 'ib' in locals() and ib.isConnected():
            ib.disconnect()
            print("\nDisconnected successfully")

if __name__ == "__main__":
    test_historical_data()
