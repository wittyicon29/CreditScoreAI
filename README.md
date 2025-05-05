# Compound V2 Credit Scoring System

## Overview
This project implements an AI-powered, decentralized credit scoring system for Compound V2 protocol wallets. The system analyzes historical transaction data to assign credit scores between 0 and 100 to each wallet, with higher scores indicating more reliable and responsible usage.

## Features
- **End-to-end data pipeline**: Processes raw transaction data, engineers features, and generates scores
- **Comprehensive feature engineering**: Extracts behavioral patterns including transaction activity, financial health, risk indicators, and stability metrics
- **Explainable scoring system**: Provides detailed reasoning for each wallet's score
- **Bot detection**: Identifies and penalizes bot-like behavior
- **Risk assessment**: Evaluates liquidation history and other risk factors

## Project Structure
```
compound-v2-scoring/
├── data_processing.py      # Data loading and preprocessing
├── feature_engineering.py  # Wallet-level feature extraction
├── scoring_model.py        # Credit scoring model
├── main.py                 # Main execution script
├── output/
│   ├── top_1000_wallets.csv    # Top 1000 wallets sorted by score
│   ├── wallet_analysis.md      # Analysis of top 5 and bottom 5 wallets
│   ├── methodology.md          # Scoring methodology documentation
│   └── score_distribution.png  # Distribution of scores
```

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/compound-v2-scoring.git
cd compound-v2-scoring

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn tqdm
```

## Usage
1. Download the Compound V2 dataset from the provided Google Drive link
2. Place the 3 largest JSON files in the project directory
3. Run the main script:
```bash
python main.py
```

## Methodology
Our credit scoring approach is based on four key pillars:

1. **Transaction Behavior**: We analyze patterns in transaction frequency, size, and diversity.
2. **Financial Health**: We assess borrowing/repayment ratios and overall balance sheet health.
3. **Risk Factors**: We identify liquidations, high leverage, and other warning signs.
4. **Account Stability**: We consider account age, consistency, and pattern regularity.

The model uses a weighted feature approach where:
- Positive behaviors (consistent repayment, long-term engagement) increase scores
- Negative behaviors (liquidations, excessive borrowing) decrease scores
- Bot-like behaviors result in significant penalties

## Score Interpretation
- **80-100**: Excellent - Responsible protocol usage with consistent repayment and long-term engagement
- **60-79**: Good - Generally responsible behavior with minor risk factors
- **40-59**: Moderate - Mixed behavior with some concerning patterns
- **20-39**: Poor - Multiple risk factors present
- **0-19**: Very Poor - High-risk behavior, liquidations, or bot-like activity

## License
MIT License
