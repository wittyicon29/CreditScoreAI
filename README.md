## Problem Understanding

The challenge requires developing a credit scoring system for Compound V2 protocol wallets based solely on transaction behavior. This is an unsupervised learning problem where we need to:
1. Define what constitutes "good" vs "bad" wallet behavior
2. Engineer features from raw transaction data
3. Create a scoring model that ranks wallets from 0-100

## My Approach

### 1. Data Processing

I designed a data pipeline that:
- Loads and combines transaction data from multiple JSON files (deposits, borrows, repays, etc.)
- Standardizes column names and data types
- Converts timestamps to datetime objects
- Handles extremely large values (common in blockchain data)

### 2. Feature Engineering

I engineered features in four key categories:

**Transaction Behavior:**
- Transaction counts and frequency
- Asset diversity
- Transaction size patterns
- Activity diversity (types of actions performed)

**Financial Health:**
- Borrow-to-deposit ratio
- Repayment consistency
- Net position calculation
- Withdrawal patterns

**Risk Indicators:**
- Liquidation history
- Borrowing without repayment
- Same-day deposit-withdrawals (potential wash trading)
- High leverage positions

**Stability Metrics:**
- Account age
- Transaction timing regularity
- Bot-like behavior detection
- Long-term engagement

### 3. Scoring Model

I implemented a weighted feature approach where:
- Each feature receives a weight based on its importance to creditworthiness
- Positive behaviors increase scores
- Negative behaviors (liquidations, high leverage) reduce scores
- Bot-like behaviors receive significant penalties
- Final scores are normalized to 0-100 range

### 4. Wallet Analysis

The model explains each wallet's score by:
- Identifying key strengths and weaknesses
- Providing specific behavioral patterns
- Comparing against overall population
- Grouping wallets into behavioral clusters

## Key Design Decisions

1. **Unsupervised Approach**: Since no labeled data exists, I defined good vs. bad behavior based on financial principles.

2. **Explainable Scoring**: Every score component is transparent and interpretable.

3. **Behavioral vs. Balance Focus**: The model emphasizes patterns of behavior rather than just account balances.

4. **Risk-Adjusted Metrics**: Higher penalties for behaviors that threaten protocol health.

5. **Bot Detection**: Special attention to identifying and appropriately scoring automated trading.

## Deliverables

1. **Methodology Document**: Detailed explanation of the scoring approach
2. **Code Implementation**: Complete, modular codebase with:
   - Data processing
   - Feature engineering
   - Scoring model
   - Analysis generation
3. **Top 1000 Wallets CSV**: Sorted by credit score
4. **Wallet Analysis**: In-depth examination of 5 high and 5 low scoring wallets

## Implementation Details

The code is structured in a modular, maintainable way:
- `data_processing.py`: Handles data loading and preparation
- `feature_engineering.py`: Extracts wallet-level features
- `scoring_model.py`: Implements the credit scoring logic
- `main.py`: Orchestrates the end-to-end pipeline
- `test_scoring.py`: Validates the model's functionality

## Credit Score Interpretation

- **80-100**: Excellent - Responsible protocol usage with consistent repayment
- **60-79**: Good - Generally responsible with minor risk factors
- **40-59**: Moderate - Mixed behavior with some concerning patterns
- **20-39**: Poor - Multiple risk factors present
- **0-19**: Very Poor - High-risk behavior or bot-like activity

The model emphasizes these key factors for high scores:
1. Consistent repayment of borrowed funds
2. Long-term protocol engagement
3. Responsible borrowing (maintaining healthy collateral ratios)
4. No liquidation events
5. Human-like (non-bot) transaction patterns
