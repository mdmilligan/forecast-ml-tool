

import sqlite3
import yfinance as yf
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_database():
    try:
        with sqlite3.connect('marketdata.db') as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS stock_prices
                         (ticker text, date text, open real, high real, low real, close real, volume real,
                          UNIQUE(ticker, date) ON CONFLICT REPLACE)''')
            conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

def fetch_stock_prices(tickers, start_date, end_date):
    stock_prices = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            if data.empty:
                logging.warning(f"No data found for ticker {ticker}")
            else:
                stock_prices[ticker] = data
        except Exception as e:
            logging.error(f"Error fetching data for {ticker}: {e}")
    return stock_prices

def insert_data(stock_prices):
    try:
        with sqlite3.connect('marketdata.db') as conn:
            cur = conn.cursor()
            for ticker, data in stock_prices.items():
                for index, row in data.iterrows():
                    date = index.strftime('%Y-%m-%d')
                    open_price = row['Open']
                    high_price = row['High']
                    low_price = row['Low']
                    close_price = row['Close']
                    volume = row['Volume']

                    cur.execute('''
                        INSERT OR REPLACE INTO stock_prices (ticker, date, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (ticker, date, open_price, high_price, low_price, close_price, volume))
            conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

def main():
    create_database()  # Create the database and table
    tickers = ['SPY', '^VIX', 'DX-Y.NYB', 'QQQ', 'HG=F']
    start_date = '2010-01-01'
    end_date = '2024-10-31'
    stock_prices = fetch_stock_prices(tickers, start_date, end_date)
    insert_data(stock_prices)

if __name__ == '__main__':
    main()