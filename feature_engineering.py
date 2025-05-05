import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def engineer_wallet_features(df):
    """
    Extract wallet-level features from transaction data
    """
    print("Engineering features for each wallet...")
    
    # Group by wallet
    wallet_groups = df.groupby('wallet_address')
    
    # Initialize lists to store wallet features
    wallet_features = []
    
    for wallet, group in tqdm(wallet_groups):
        # Sort transactions by timestamp
        group = group.sort_values('timestamp')
        
        # Basic transaction counts and amounts
        deposits = group[group['transaction_type'] == 'deposits']
        borrows = group[group['transaction_type'] == 'borrows']
        repays = group[group['transaction_type'] == 'repays']
        withdraws = group[group['transaction_type'] == 'withdraws'] if 'withdraws' in df['transaction_type'].unique() else pd.DataFrame()
        liquidations = group[group['transaction_type'] == 'liquidations'] if 'liquidations' in df['transaction_type'].unique() else pd.DataFrame()
        
        # Transaction counts
        num_deposits = len(deposits)
        num_borrows = len(borrows)
        num_repays = len(repays)
        num_withdraws = len(withdraws)
        num_liquidations = len(liquidations)
        total_transactions = len(group)
        
        # Transaction amounts (USD)
        total_deposit_usd = deposits['amountUSD'].sum() if num_deposits > 0 else 0
        total_borrow_usd = borrows['amountUSD'].sum() if num_borrows > 0 else 0
        total_repay_usd = repays['amountUSD'].sum() if num_repays > 0 else 0
        total_withdraw_usd = withdraws['amountUSD'].sum() if num_withdraws > 0 else 0
        total_liquidation_usd = liquidations['amountUSD'].sum() if num_liquidations > 0 else 0
        
        # Calculate ratios
        borrow_to_deposit_ratio = total_borrow_usd / total_deposit_usd if total_deposit_usd > 0 else np.inf
        repay_to_borrow_ratio = total_repay_usd / total_borrow_usd if total_borrow_usd > 0 else np.inf
        withdraw_to_deposit_ratio = total_withdraw_usd / total_deposit_usd if total_deposit_usd > 0 else np.inf
        
        # Asset diversity
        unique_assets = group['asset_symbol'].nunique()
        
        # Time-based features
        first_activity = group['timestamp'].min()
        last_activity = group['timestamp'].max()
        account_age_days = (last_activity - first_activity).days + 1  # Add 1 to avoid division by zero
        
        # Activity patterns
        transactions_per_day = total_transactions / account_age_days
        
        # Transaction size statistics
        avg_transaction_size = group['amountUSD'].mean()
        median_transaction_size = group['amountUSD'].median()
        std_transaction_size = group['amountUSD'].std() if len(group) > 1 else 0
        
        # Transaction timing features
        if len(group) > 1:
            time_diffs = group['timestamp'].diff().dropna()
            avg_time_between_txs = time_diffs.mean().total_seconds() / 3600  # in hours
            std_time_between_txs = time_diffs.std().total_seconds() / 3600 if len(time_diffs) > 1 else 0
            min_time_between_txs = time_diffs.min().total_seconds() / 3600
            max_time_between_txs = time_diffs.max().total_seconds() / 3600
        else:
            avg_time_between_txs = np.nan
            std_time_between_txs = np.nan
            min_time_between_txs = np.nan
            max_time_between_txs = np.nan
        
        # Behavioral alerts
        has_liquidations = 1 if num_liquidations > 0 else 0
        borrow_without_repay = 1 if num_borrows > 0 and num_repays == 0 else 0
        deposit_withdraw_same_day = 0
        
        # Check for same-day deposit-withdraw patterns
        if num_deposits > 0 and num_withdraws > 0:
            deposit_dates = deposits['timestamp'].dt.date
            withdraw_dates = withdraws['timestamp'].dt.date
            common_dates = set(deposit_dates).intersection(set(withdraw_dates))
            deposit_withdraw_same_day = len(common_dates)
        
        # Transaction frequency stability
        if len(group) > 2:
            time_diffs = group['timestamp'].diff().dropna().dt.total_seconds()
            time_diff_variance = time_diffs.var()
            time_diff_stability = 1 / (1 + np.log1p(time_diff_variance)) if time_diff_variance > 0 else 1
        else:
            time_diff_stability = np.nan
            
        # Calculate net position (deposits + repays - borrows - withdraws)
        net_position = total_deposit_usd + total_repay_usd - total_borrow_usd - total_withdraw_usd
        
        wallet_features.append({
            'wallet_address': wallet,
            'total_transactions': total_transactions,
            'num_deposits': num_deposits,
            'num_borrows': num_borrows,
            'num_repays': num_repays,
            'num_withdraws': num_withdraws,
            'num_liquidations': num_liquidations,
            'total_deposit_usd': total_deposit_usd,
            'total_borrow_usd': total_borrow_usd,
            'total_repay_usd': total_repay_usd,
            'total_withdraw_usd': total_withdraw_usd,
            'total_liquidation_usd': total_liquidation_usd,
            'borrow_to_deposit_ratio': borrow_to_deposit_ratio,
            'repay_to_borrow_ratio': repay_to_borrow_ratio,
            'withdraw_to_deposit_ratio': withdraw_to_deposit_ratio,
            'unique_assets': unique_assets,
            'account_age_days': account_age_days,
            'transactions_per_day': transactions_per_day,
            'avg_transaction_size': avg_transaction_size,
            'median_transaction_size': median_transaction_size,
            'std_transaction_size': std_transaction_size,
            'avg_time_between_txs': avg_time_between_txs,
            'std_time_between_txs': std_time_between_txs,
            'min_time_between_txs': min_time_between_txs,
            'max_time_between_txs': max_time_between_txs,
            'has_liquidations': has_liquidations,
            'borrow_without_repay': borrow_without_repay,
            'deposit_withdraw_same_day': deposit_withdraw_same_day,
            'time_diff_stability': time_diff_stability,
            'net_position': net_position,
            'first_activity': first_activity,
            'last_activity': last_activity
        })
    
    # Convert to DataFrame
    wallet_features_df = pd.DataFrame(wallet_features)
    
    # Replace infinities with large numbers
    wallet_features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill missing values
    for col in wallet_features_df.columns:
        if wallet_features_df[col].dtype in [np.float64, np.int64]:
            wallet_features_df[col].fillna(wallet_features_df[col].median(), inplace=True)
    
    print(f"Engineered features for {len(wallet_features_df)} wallets")
    return wallet_features_df

def calculate_derived_features(df):
    """Calculate additional derived features"""
    
    # Healthy repayment behavior
    df['healthy_repayment'] = np.where(
        (df['repay_to_borrow_ratio'] >= 0.95) & (df['repay_to_borrow_ratio'] <= 1.05),
        1, 0
    )
    
    # Bot-like behavior detection
    df['bot_likelihood'] = np.where(
        (df['std_time_between_txs'] < 1) &  # Very consistent timing
        (df['transactions_per_day'] > 5) &  # High frequency
        (df['std_transaction_size'] < 0.01 * df['avg_transaction_size']),  # Very consistent amounts
        1, 0
    )
    
    # Risk factors
    df['high_risk_factors'] = (
        df['has_liquidations'] + 
        df['borrow_without_repay'] +
        np.where(df['borrow_to_deposit_ratio'] > 0.9, 1, 0) +
        np.where(df['deposit_withdraw_same_day'] > 3, 1, 0)
    )
    
    # Longevity and stability
    df['longevity_score'] = np.clip(df['account_age_days'] / 365, 0, 1)  # Normalized to 0-1
    
    # Activity diversity
    df['activity_diversity'] = (
        np.minimum(df['num_deposits'], 1) + 
        np.minimum(df['num_borrows'], 1) + 
        np.minimum(df['num_repays'], 1) + 
        np.minimum(df['num_withdraws'], 1)
    ) / 4
    
    # Transaction size consistency (lower is more consistent)
    df['tx_size_consistency'] = np.where(
        df['avg_transaction_size'] > 0,
        np.minimum(df['std_transaction_size'] / df['avg_transaction_size'], 10),
        10
    )
    
    return df

def main():
    # Load all transaction type files
    transaction_types = ['deposit', 'borrow', 'repay', 'withdraw']
    dataframes = []
    
    for tx_type in transaction_types:
        filename = f'processed_compound_v2_{tx_type}s.csv'
        try:
            df = pd.read_csv(filename)
            dataframes.append(df)
            print(f"Loaded {filename} with {len(df)} records")
        except FileNotFoundError:
            print(f"Warning: {filename} not found, skipping...")
    
    if not dataframes:
        raise ValueError("No transaction data files found!")
    
    # Combine all transaction data
    df = pd.concat(dataframes, ignore_index=True)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Process features
    wallet_features = engineer_wallet_features(df)
    wallet_features = calculate_derived_features(wallet_features)

    # Save results
    wallet_features.to_csv('wallet_features.csv', index=False)
    print(f"\nSaved features for {len(wallet_features)} wallets to wallet_features.csv")

    print("\nFeature statistics:")
    print(wallet_features.describe())

if __name__ == "__main__":
    from tqdm import tqdm
    main()