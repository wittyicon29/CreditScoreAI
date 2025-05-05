import os
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import the custom modules
from data_processing import load_and_process_json, prepare_features, main as process_data
from feature_engineering import engineer_wallet_features, calculate_derived_features, main as engineer_features
from scoring_model import CompoundCreditScorer

def generate_wallet_analysis_document(wallet_analyses):
    """Generate a one-page document analyzing high and low scoring wallets"""
    
    # Create analysis document
    analysis_doc = "# Wallet Behavior Analysis\n\n"
    analysis_doc += f"*Analysis date: {datetime.now().strftime('%Y-%m-%d')}*\n\n"
    
    # Add high-scoring wallets
    analysis_doc += "## Top 5 High-Scoring Wallets\n\n"
    
    for i, wallet in enumerate(wallet_analyses[:5]):
        analysis_doc += f"### {i+1}. Wallet {wallet['wallet_address'][:6]}...{wallet['wallet_address'][-4:]} (Score: {wallet['credit_score']})\n\n"
        
        # Add key strengths
        analysis_doc += "**Key Strengths:**\n"
        for strength in wallet['key_strengths']:
            analysis_doc += f"- {strength}\n"
        analysis_doc += "\n"
        
        # Add transaction behavior
        analysis_doc += f"**Transaction Activity:** {wallet['factors']['transaction_behavior']['transactions']} transactions "
        analysis_doc += f"({wallet['factors']['transaction_behavior']['transactions_per_day']:.2f} per day) "
        analysis_doc += f"using {wallet['factors']['transaction_behavior']['unique_assets']} assets\n\n"
        
        # Add financial health
        analysis_doc += f"**Financial Health:** "
        if wallet['factors']['financial_health']['healthy_repayment']:
            analysis_doc += "Consistently repays borrows. "
        else:
            analysis_doc += "Inconsistent repayment patterns. "
        
        analysis_doc += f"Borrow/Deposit ratio: {wallet['factors']['financial_health']['borrow_to_deposit_ratio']}\n\n"
    
    # Add low-scoring wallets
    analysis_doc += "## Bottom 5 Low-Scoring Wallets\n\n"
    
    for i, wallet in enumerate(wallet_analyses[5:]):
        analysis_doc += f"### {i+1}. Wallet {wallet['wallet_address'][:6]}...{wallet['wallet_address'][-4:]} (Score: {wallet['credit_score']})\n\n"
        
        # Add key weaknesses
        analysis_doc += "**Key Concerns:**\n"
        for weakness in wallet['key_weaknesses']:
            analysis_doc += f"- {weakness}\n"
        analysis_doc += "\n"
        
        # Add risk factors
        analysis_doc += f"**Risk Profile:** "
        if wallet['factors']['risk_factors']['liquidations'] > 0:
            analysis_doc += f"Has been liquidated {wallet['factors']['risk_factors']['liquidations']} times. "
        
        if wallet['factors']['risk_factors']['borrow_without_repay']:
            analysis_doc += "Has outstanding unpaid borrows. "
        
        analysis_doc += f"\n\n"
        
        # Add stability info
        analysis_doc += f"**Account Stability:** "
        analysis_doc += f"Active for {wallet['factors']['stability']['account_age_days']} days. "
        
        if wallet['factors']['stability']['bot_likelihood']:
            analysis_doc += "Shows bot-like activity patterns."
        else:
            analysis_doc += "Shows human-like activity patterns."
        
        analysis_doc += "\n\n"
    
    # Add summary and patterns
    analysis_doc += "## Key Behavioral Patterns\n\n"
    analysis_doc += "**High Score Indicators:**\n"
    analysis_doc += "- Consistent repayment of borrowed funds\n"
    analysis_doc += "- Long-term protocol engagement\n"
    analysis_doc += "- Diverse asset usage\n"
    analysis_doc += "- Responsible borrowing (borrow less than deposit)\n"
    analysis_doc += "- No liquidation events\n\n"
    
    analysis_doc += "**Low Score Indicators:**\n"
    analysis_doc += "- History of liquidations\n"
    analysis_doc += "- Borrowing without repayment\n"
    analysis_doc += "- Bot-like transaction patterns\n"
    analysis_doc += "- Excessive borrowing relative to deposits\n"
    analysis_doc += "- Short engagement with the protocol\n"
    
    return analysis_doc

def main():
    """Execute the complete credit scoring pipeline"""
    start_time = time.time()
    print("Starting Compound V2 Credit Scoring Pipeline...")
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Step 1: Process raw data
    print("\n--- Step 1: Processing Raw Data ---")
    if not os.path.exists('processed_compound_v2_data.csv'):
        process_data()
    else:
        print("Using existing processed data file")
    
    # Step 2: Engineer wallet features
    print("\n--- Step 2: Engineering Wallet Features ---")
    if not os.path.exists('wallet_features.csv'):
        engineer_features()
    else:
        print("Using existing wallet features file")
    
    # Step 3: Score wallets
    print("\n--- Step 3: Scoring Wallets ---")
    wallet_df = pd.read_csv('wallet_features.csv')
    
    # Initialize the credit scorer
    scorer = CompoundCreditScorer()
    
    # Score wallets
    scores_df = scorer.score_wallets(wallet_df)
    
    # Get top 1000 wallets by score
    top_wallets = scores_df.sort_values('credit_score', ascending=False).head(1000)
    
    # Save top wallets to CSV
    top_wallets.to_csv('output/top_1000_wallets.csv', index=False)
    print(f"Saved top 1000 wallets to output/top_1000_wallets.csv")
    
    # Step 4: Generate wallet analyses
    print("\n--- Step 4: Analyzing Top and Bottom Wallets ---")
    
    # Get 5 highest and 5 lowest scoring wallets for analysis
    high_scorers = scores_df.sort_values('credit_score', ascending=False).head(5)
    low_scorers = scores_df.sort_values('credit_score', ascending=True).head(5)
    
    # Generate explanations for high and low scorers
    wallet_analyses = []
    
    print("\nAnalyzing top 5 wallets:")
    for _, row in high_scorers.iterrows():
        explanation = scorer.explain_score(row['wallet_address'], wallet_df, scores_df)
        wallet_analyses.append(explanation)
        print(f"Wallet {row['wallet_address']}: Score {row['credit_score']}")
    
    print("\nAnalyzing bottom 5 wallets:")
    for _, row in low_scorers.iterrows():
        explanation = scorer.explain_score(row['wallet_address'], wallet_df, scores_df)
        wallet_analyses.append(explanation)
        print(f"Wallet {row['wallet_address']}: Score {row['credit_score']}")
    
    # Generate and save wallet analysis document
    analysis_doc = generate_wallet_analysis_document(wallet_analyses)
    with open('output/wallet_analysis.md', 'w') as f:
        f.write(analysis_doc)
    print(f"Saved wallet analysis document to output/wallet_analysis.md")
    
    # Step 5: Generate methodology document
    print("\n--- Step 5: Generating Methodology Document ---")
    methodology_doc = """# Compound V2 Credit Scoring Methodology

## Overview
This document outlines the methodology used to develop a credit scoring system for wallets interacting with the Compound V2 protocol. The scoring system assigns a value between 0 and 100 to each wallet, with higher scores indicating more reliable and responsible usage patterns.

## Data Processing
The raw transaction data from Compound V2 was processed to extract wallet-level behavioral patterns. Each transaction record contains information about the wallet address, transaction type (deposit, borrow, repay, withdraw, liquidation), amount, asset, and timestamp.

## Feature Engineering
We engineered the following features to capture wallet behavior:

### Transaction Activity
- Total number of transactions
- Transactions per day
- Number of unique assets used
- Activity diversity (types of actions performed)

### Financial Health
- Borrow to deposit ratio
- Repay to borrow ratio
- Withdraw to deposit ratio
- Net position (deposits + repays - borrows - withdraws)
- Healthy repayment behavior (consistent repayment of borrowed funds)

### Risk Factors
- Liquidation history
- Borrowing without repayment
- Same-day deposit-withdraw patterns (potential wash trading)
- High-risk behavioral flags

### Stability and Consistency
- Account age (days)
- Transaction size consistency
- Transaction timing patterns
- Longevity of platform engagement

### Bot Detection
- Indicators of automated trading behavior
- Extremely consistent transaction timing
- Uniform transaction sizes

## Scoring Model
The scoring model uses a weighted feature approach combined with rule-based adjustments:

1. **Base Score Calculation**: Each feature is weighted according to its importance in determining creditworthiness
2. **Risk Adjustments**: Penalties are applied for high-risk behaviors
3. **Bot Detection**: Significant penalties for wallets exhibiting bot-like behavior
4. **Liquidation Impact**: Graduated penalties based on liquidation history
5. **Score Normalization**: Raw scores are normalized to a 0-100 scale

## Feature Weights
The following weights reflect the importance of each feature category:
- Transaction behavior: 25%
- Financial health: 35%
- Risk factors: 25%
- Stability and consistency: 15%

## Score Interpretation
- **80-100**: Excellent - Responsible protocol usage with consistent repayment and long-term engagement
- **60-79**: Good - Generally responsible behavior with minor risk factors
- **40-59**: Moderate - Mixed behavior with some concerning patterns
- **20-39**: Poor - Multiple risk factors present
- **0-19**: Very Poor - High-risk behavior, liquidations, or bot-like activity

## Validation
The model was validated by examining wallet behavior across the score spectrum and confirming that scoring aligns with expected patterns of responsible protocol usage.
"""
    
    with open('output/methodology.md', 'w') as f:
        f.write(methodology_doc)
    print(f"Saved methodology document to output/methodology.md")
    
    # Print execution time
    execution_time = time.time() - start_time
    print(f"\nCredit scoring pipeline completed in {execution_time:.2f} seconds")
    print(f"All outputs saved to the 'output' directory")

if __name__ == "__main__":
    main()