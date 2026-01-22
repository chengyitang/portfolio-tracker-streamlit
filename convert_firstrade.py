import pandas as pd
import os

def convert_firstrade_to_ledger(input_path, output_path):
    print(f"Reading {input_path}...")
    try:
        # Firstrade CSVs sometimes have trailing commas or weird encoding, but read_csv usually handles standard ones.
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading raw CSV: {e}")
        return

    # Filter for relevant rows
    # We only want rows that have a Symbol (Stocks/ETFs)
    # This filters out 'ACH DEPOSIT', 'Interest', etc. unless they are assigned to a symbol (Interest sometimes isn't)
    df = df[df['Symbol'].notna() & (df['Symbol'] != '')].copy()
    
    # Map 'Action' to 'Type'
    def map_action(action):
        action = str(action).upper().strip()
        if 'BUY' in action:
            return 'Buy'
        if 'SELL' in action:
            return 'Sell'
        if 'DIVIDEND' in action:
            return 'Dividend'
        if 'REINVEST' in action: # DRIP
            return 'Buy' 
        return 'Other'

    df['Type'] = df['Action'].apply(map_action)
    
    # Filter out 'Other' types if any remain (e.g. maybe 'Journal' or 'Merger' if unsupported?)
    # Keeping them might break logic if Type isn't recognized, but let's keep strict for now.
    df = df[df['Type'].isin(['Buy', 'Sell', 'Dividend'])]

    # Prepare Columns
    # Target: Date, Ticker, Stock Name, Type, Shares, Price, Fee, Cash Flow
    
    # Date -> TradeDate
    df['Date'] = pd.to_datetime(df['TradeDate']).dt.strftime('%Y-%m-%d')
    
    # Ticker -> Symbol
    df['Ticker'] = df['Symbol'].str.strip()
    
    # Stock Name -> Description (Clean it up)
    # Remove "UNSOLICITED", "***", etc using regex or simple replace
    df['Stock Name'] = df['Description'].astype(str).str.replace(r'UNSOLICITED.*', '', regex=True).str.replace(r'\*\*\*', '', regex=True).str.strip()
    
    # Shares -> Quantity (Absolute value)
    df['Shares'] = df['Quantity'].abs()
    
    # Price -> Price
    # Fill NaN price with 0 for Dividends
    df['Price'] = df['Price'].fillna(0.0)
    
    # Fee -> Commission + Fee
    df['Fee'] = df['Commission'].fillna(0.0) + df['Fee'].fillna(0.0)
    
    # Cash Flow -> Amount
    # Firstrade Amount: Buy is negative, Sell/Div is positive. 
    # This matches exactly what we want for "Cash Flow" column if we use it directly.
    # Logic note: If we populate 'Cash Flow', app.py uses it.
    df['Cash Flow'] = df['Amount']

    # Select and Reorder
    final_df = df[['Date', 'Ticker', 'Stock Name', 'Type', 'Shares', 'Price', 'Fee', 'Cash Flow']]
    
    # Sort by Date
    final_df = final_df.sort_values(by='Date')
    
    print(f"Converted {len(final_df)} transactions.")
    print(final_df.head())
    
    # Save
    final_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    convert_firstrade_to_ledger("FT_CSV_90175212.csv", "us-ledger.csv")
