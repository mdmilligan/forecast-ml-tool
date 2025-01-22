from ib_insync import *
import pandas as pd
import sqlite3
import logging
import argparse
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)

import os

# Get absolute path to database
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'marketdata.db')
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def create_table_if_not_exists(conn, symbol):
    cursor = conn.cursor()
    table_name = f"stock_data_{symbol.lower()}"
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            date TIMESTAMP,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            average REAL,
            barCount INTEGER,
            PRIMARY KEY (date)
        )
    ''')
    conn.commit()

def get_historical_data_chunk(ib, contract, end_datetime, bar_size, duration='1 M'):
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=end_datetime,
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1
    )
    return bars

def get_all_historical_data(ib, symbol, conn, bar_size='30 mins', months=36):
    try:
        if symbol == 'VIX':
            contract = Index(symbol, 'CBOE', 'USD')
        else:
            contract = Stock(symbol, 'SMART', 'USD')
            
        ib.qualifyContracts(contract)  # Ensure contract is qualified
            
        print(f"\nRequesting historical data for {symbol}...")
        
        # Check if table exists and get latest date
        table_name = f"stock_data_{symbol.lower()}"
        create_table_if_not_exists(conn, symbol)
        
        # Get existing data's latest date if any
        cursor = conn.cursor()
        cursor.execute(f"SELECT MAX(date) FROM {table_name}")
        latest_date = cursor.fetchone()[0]
        
        # If we have existing data, start from the day after the latest date
        if latest_date:
            end_date = datetime.strptime(latest_date, '%Y-%m-%d %H:%M:%S') + timedelta(days=1)
            print(f"Found existing data up to {latest_date}, starting from {end_date}")
        else:
            end_date = datetime.now()
            print("No existing data found, starting from current date")
        
        all_bars = []
        chunks_received = 0
        
        # Request data in 1-month chunks for the specified number of months
        for _ in range(months):
            try:
                print(f"Requesting chunk ending {end_date.strftime('%Y-%m-%d')}")
                bars = get_historical_data_chunk(
                    ib, 
                    contract, 
                    end_date.strftime('%Y%m%d %H:%M:%S'), 
                    bar_size
                )
                
                if bars:
                    all_bars = bars + all_bars  # Prepend new bars
                    chunks_received += 1
                    print(f"Received {len(bars)} bars for chunk {chunks_received}")
                else:
                    print(f"No data received for chunk ending {end_date.strftime('%Y-%m-%d')}")
                
                # Move end_date back by 1 month
                end_date = end_date - timedelta(days=30)
                
                # Add a delay between requests to avoid pacing violations
                ib.sleep(3)
                
            except Exception as e:
                print(f"Error getting chunk: {str(e)}")
                ib.sleep(10)  # Longer delay after an error
                continue
        
        if all_bars:
            df = util.df(all_bars)
            df = df.drop_duplicates(subset=['date'])  # Remove any duplicate dates
            df = df.sort_values('date')
            
            print(f"\nTotal bars collected for {symbol}: {len(df)}")
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
            
            # Save to database - append if data exists
            df.to_sql(table_name, conn, if_exists='append', index=False)
            print(f"Data appended to database table: {table_name}")
            
            return df
            
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        return None

def main():
    try:
        # Set up argument parsing
        parser = argparse.ArgumentParser(description='Fetch historical data from IBKR')
        parser.add_argument('--symbol', '-s', type=str, required=True,
                          help='Stock symbol to fetch (e.g., SPY)')
        parser.add_argument('--bar-size', '-b', type=str, default='30 mins',
                          choices=['1 min', '5 mins', '15 mins', '30 mins', '1 hour', '1 day'],
                          help='Bar size for historical data')
        parser.add_argument('--months', '-m', type=int, default=36,
                          help='Number of months of historical data to fetch')
        
        args = parser.parse_args()
        
        conn = sqlite3.connect(DB_PATH)
        print(f"Connected to database: {DB_PATH}")
        
        print("Connecting to IBGateway...")
        ib = IB()
        # Try common IBKR ports
        # Try common IBKR ports with timeout
        ports = [7497, 4001, 4002]
        timeout = 10  # seconds
        for port in ports:
            try:
                print(f"Attempting connection on port {port}...")
                ib.connect('127.0.0.1', port, clientId=1, timeout=timeout)
                if ib.isConnected():
                    print(f"Successfully connected to IBKR on port {port}")
                    # Verify connection by requesting account data
                    accounts = ib.managedAccounts()
                    if accounts:
                        print(f"Connected to account: {accounts[0]}")
                        break
                    else:
                        print("No accounts found - disconnecting")
                        ib.disconnect()
                else:
                    print("Connection failed")
            except Exception as e:
                print(f"Failed to connect on port {port}: {str(e)}")
                ib.sleep(1)  # Short delay between attempts
        else:
            raise ConnectionError(f"Could not connect to IBKR on any port after {len(ports)} attempts")
        
        print(f"\nFetching {args.months} months of {args.bar_size} data for {args.symbol}...")
        df = get_all_historical_data(ib, args.symbol, conn, bar_size=args.bar_size, months=args.months)
            
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        
    finally:
        if 'conn' in locals():
            conn.close()
            print("Database connection closed")
            
        if 'ib' in locals() and ib.isConnected():
            ib.disconnect()
            print("\nDisconnected from IBGateway")

if __name__ == "__main__":
    main()
