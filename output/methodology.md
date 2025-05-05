# Compound V2 Credit Scoring Methodology

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
