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

def get_all_historical_data(ib, symbol, conn, bar_size='30 mins', start_date=None, end_date=None):
    try:
        if symbol == 'VIX':
            contract = Index(symbol, 'CBOE', 'USD')
        else:
            contract = Stock(symbol, 'SMART', 'USD')
            
        ib.qualifyContracts(contract)  # Ensure contract is qualified
            
        print(f"\nRequesting historical data for {symbol}...")
        
        # Check if table exists
        table_name = f"stock_data_{symbol.lower()}"
        create_table_if_not_exists(conn, symbol)
        
        # Convert string dates to datetime objects
        if end_date:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_date = datetime.now()
            
        if start_date:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            # Default to 3 years ago if no start date provided
            start_date = end_date - timedelta(days=365*3)
            
        # Calculate total months between dates
        total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        
        all_bars = []
        current_end = end_date
        chunks_received = 0
        
        print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        while current_end > start_date:
            try:
                chunk_start = current_end - timedelta(days=30)
                print(f"Requesting chunk {chunks_received + 1}: {chunk_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
                bars = get_historical_data_chunk(
                    ib, 
                    contract, 
                    current_end.strftime('%Y%m%d %H:%M:%S'), 
                    bar_size
                )
                
                if bars:
                    all_bars = bars + all_bars  # Prepend new bars
                    chunks_received += 1
                    print(f"Received {len(bars)} bars for chunk {chunks_received}")
                else:
                    print(f"No data received for chunk {chunks_received + 1}")
                
                # Move current_end back by 1 month
                current_end = current_end - timedelta(days=30)
                
                # Add a delay between requests to avoid pacing violations
                ib.sleep(3)
                
            except Exception as e:
                print(f"Error getting chunk: {str(e)}")
                ib.sleep(10)  # Longer delay after an error
                continue
        
        if all_bars:
            df = util.df(all_bars)
            df = df.drop_duplicates(subset=['date'])
            df = df.sort_values('date')
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Debug print before filtering
            print("\nRaw data date range:", df['date'].min(), "to", df['date'].max())
            
            # Convert to naive datetime for comparison
            df['date_naive'] = df['date'].dt.tz_localize(None)
            
            # Debug print naive dates
            print("Naive date range:", df['date_naive'].min(), "to", df['date_naive'].max())
            
            # Filter with inclusive end date
            mask = (df['date_naive'] >= start_date) & (df['date_naive'] <= end_date + timedelta(days=1))
            df_filtered = df[mask]
            
            # Debug print filtered dates
            print("Filtered date range:", df_filtered['date_naive'].min(), "to", df_filtered['date_naive'].max())
            
            # Check if end date exists
            if end_date not in df_filtered['date_naive'].dt.date.values:
                print(f"\nWarning: End date {end_date.date()} not found in data")
                # Try to fetch the missing day specifically
                try:
                    print(f"Attempting to fetch missing date: {end_date.date()}")
                    missing_bars = get_historical_data_chunk(
                        ib, 
                        contract, 
                        (end_date + timedelta(days=1)).strftime('%Y%m%d %H:%M:%S'), 
                        bar_size,
                        duration='1 D'
                    )
                    if missing_bars:
                        missing_df = util.df(missing_bars)
                        missing_df['date'] = pd.to_datetime(missing_df['date'])
                        missing_df['date_naive'] = missing_df['date'].dt.tz_localize(None)
                        df_filtered = pd.concat([df_filtered, missing_df])
                        print(f"Added {len(missing_df)} bars for {end_date.date()}")
                except Exception as e:
                    print(f"Error fetching missing date: {str(e)}")
            
            # Drop the temporary naive date column
            df = df_filtered.drop(columns=['date_naive'])
            
            print(f"\nTotal bars collected for {symbol}: {len(df)}")
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
            
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
        parser.add_argument('--start-date', type=str,
                          help='Start date for historical data (YYYY-MM-DD)')
        parser.add_argument('--end-date', type=str,
                          help='End date for historical data (YYYY-MM-DD)')
        
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
        
        # Calculate and display the timeframe
        start_dt = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
        end_dt = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None
        
        if start_dt and end_dt:
            timeframe = f"from {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}"
        elif start_dt:
            timeframe = f"from {start_dt.strftime('%Y-%m-%d')} to present"
        elif end_dt:
            timeframe = f"up to {end_dt.strftime('%Y-%m-%d')}"
        else:
            timeframe = "for the last 3 years"
            
        print(f"\nFetching {args.bar_size} data for {args.symbol} {timeframe}...")
        df = get_all_historical_data(
            ib, 
            args.symbol, 
            conn, 
            bar_size=args.bar_size,
            start_date=args.start_date,
            end_date=args.end_date
        )
            
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
