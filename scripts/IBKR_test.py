from ib_insync import *
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)

def test_connection():
    try:
        print("Attempting to connect to IBGateway...")
        ib = IB()
        # Note: Using clientId=1 might conflict if you have other connections
        # It's good practice to use different clientIds for different applications
        ib.connect('127.0.0.1', 4001, clientId=123)  
        
        # Add a small delay to ensure connection is established
        ib.sleep(1)
        
        if ib.isConnected():
            print("\nConnection Status:")
            print(f"Connected: {ib.isConnected()}")
            print(f"Client ID: {ib.client.clientId}")
            print(f"TWS Time: {ib.reqCurrentTime()}")
            
            print("\nTesting API functionality:")
            # Try to get account info
            account = ib.accountSummary()
            print("✓ Successfully retrieved account info")
            
            # Try to get contract info
            contract = Stock('SPY', 'SMART', 'USD')
            ib.qualifyContracts(contract)
            print("✓ Successfully retrieved contract info")
            
        else:
            print("Connection failed!")
            
    except ConnectionRefusedError:
        print("\nError: Connection refused!")
        print("Please check if:")
        print("1. IBGateway is running")
        print("2. You're logged in to IBGateway")
        print("3. API connections are enabled in IBGateway settings")
        
    except Exception as e:
        print(f"\nUnexpected error occurred: {str(e)}")
        print("Error type:", type(e).__name__)
        
    finally:
        if 'ib' in locals() and ib.isConnected():
            ib.disconnect()
            print("\nDisconnected successfully")

if __name__ == "__main__":
    print("Starting connection test...")
    test_connection()
