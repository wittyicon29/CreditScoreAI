import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
from tqdm import tqdm

file_paths = [
    'Data\\compoundV2_transactions_ethereum_chunk_1.json',
    'Data\\compoundV2_transactions_ethereum_chunk_2.json',
    'Data\\compoundV2_transactions_ethereum_chunk_4.json'
]

def load_and_process_json(file_path):
    """Load JSON data and convert to DataFrame"""
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r') as f:
        data = json.load(f)

    deposits = []
    borrows = []
    repays = []
    withdraws = []

    if 'deposits' in data:
        deposits.extend(data['deposits'])
    if 'borrows' in data:
        borrows.extend(data['borrows'])
    if 'repays' in data:
        repays.extend(data['repays'])
    if 'withdraws' in data:
        withdraws.extend(data['withdraws'])

    dfs = []
    if deposits:
        df_deposits = pd.json_normalize(deposits)
        df_deposits['transaction_type'] = 'deposit'
        dfs.append(df_deposits)
    
    if borrows:
        df_borrows = pd.json_normalize(borrows)
        df_borrows['transaction_type'] = 'borrow'
        dfs.append(df_borrows)
    
    if repays:
        df_repays = pd.json_normalize(repays)
        df_repays['transaction_type'] = 'repay'
        dfs.append(df_repays)
    
    if withdraws:
        df_withdraws = pd.json_normalize(withdraws)
        df_withdraws['transaction_type'] = 'withdraw'
        dfs.append(df_withdraws)
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

def prepare_features(dfs):
    """Combine DataFrames and prepare features"""
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Standardize column names
    combined_df.rename(columns={
        'account.id': 'wallet_address',
        'asset.symbol': 'asset_symbol',
        'asset.id': 'asset_id'
    }, inplace=True)
    
    # Convert timestamp to datetime
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'].astype(int), unit='s')
    
    # Convert amount columns to numeric with high precision
    # First convert to string to handle very large numbers
    combined_df['amount'] = combined_df['amount'].astype(str)
    combined_df['amountUSD'] = combined_df['amountUSD'].astype(str)
    
    # Then convert to float64 using pd.to_numeric with errors='coerce'
    combined_df['amount'] = pd.to_numeric(combined_df['amount'], errors='coerce')
    combined_df['amountUSD'] = pd.to_numeric(combined_df['amountUSD'], errors='coerce')
    
    return combined_df

def main():
    # Load and process each JSON file
    dataframes = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            df = load_and_process_json(file_path)
            if not df.empty:
                dataframes.append(df)
        else:
            print(f"File not found: {file_path}")
    
    if not dataframes:
        print("No data files were processed successfully.")
        return

    combined_df = prepare_features(dataframes)

    # Save separate CSV files for each transaction type
    for tx_type in ['deposit', 'borrow', 'repay', 'withdraw']:
        df_filtered = combined_df[combined_df['transaction_type'] == tx_type]
        if not df_filtered.empty:
            df_filtered.to_csv(f'processed_compound_v2_{tx_type}s.csv', index=False)
            print(f"Saved {tx_type} data with {len(df_filtered)} records")

    print("\nBasic statistics:")
    print("Transaction counts by type:")
    print(combined_df['transaction_type'].value_counts())
    print(f"\nUnique wallets: {combined_df['wallet_address'].nunique()}")
    print(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    print(f"Assets: {combined_df['asset_symbol'].unique()}")

if __name__ == "__main__":
    main()