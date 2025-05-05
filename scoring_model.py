import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import silhouette_score
warnings.filterwarnings('ignore')

class CompoundCreditScorer:
    """
    A credit scoring model for Compound V2 protocol wallets
    """
    
    def __init__(self):
        # Define the features to use for scoring
        self.scoring_features = [
            # Transaction behavior
            'total_transactions',
            'transactions_per_day',
            'unique_assets',
            'activity_diversity',
            
            # Financial health
            'borrow_to_deposit_ratio',
            'repay_to_borrow_ratio',
            'withdraw_to_deposit_ratio',
            'net_position',
            'healthy_repayment',
            
            # Risk factors
            'has_liquidations',
            'borrow_without_repay',
            'deposit_withdraw_same_day',
            'high_risk_factors',
            
            # Stability and consistency
            'account_age_days',
            'longevity_score',
            'tx_size_consistency',
            'time_diff_stability',
            
            # Bot detection
            'bot_likelihood'
        ]
        
        # Feature importance weights (subjective, based on domain knowledge)
        self.feature_weights = {
            # Transaction behavior - 25%
            'total_transactions': 0.05,
            'transactions_per_day': 0.05,
            'unique_assets': 0.05,
            'activity_diversity': 0.10,
            
            # Financial health - 35%
            'borrow_to_deposit_ratio': 0.05,
            'repay_to_borrow_ratio': 0.15,
            'withdraw_to_deposit_ratio': 0.05,
            'net_position': 0.05,
            'healthy_repayment': 0.05,
            
            # Risk factors - 25%
            'has_liquidations': -0.10,  # Negative impact
            'borrow_without_repay': -0.05,  # Negative impact
            'deposit_withdraw_same_day': -0.05,  # Negative impact
            'high_risk_factors': -0.05,  # Negative impact
            
            # Stability and consistency - 15%
            'account_age_days': 0.05,
            'longevity_score': 0.05,
            'tx_size_consistency': 0.025,
            'time_diff_stability': 0.025,
            
            # Bot detection - 0% (handled separately)
            'bot_likelihood': 0  # Will be handled separately
        }
        
        self.scaler = MinMaxScaler()
    
    def preprocess_data(self, df):
        """Preprocess the data for scoring"""
        # Make a copy of the input data
        working_df = df.copy()
        
        # Check if all required features are present
        missing_features = [f for f in self.scoring_features if f not in working_df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Scale the features
        feature_df = working_df[self.scoring_features].copy()
        
        # Replace infinities and NaNs
        feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill NaNs with zeros
        feature_df.fillna(0, inplace=True)
        
        # Scale the values to 0-1 range
        scaled_features = self.scaler.fit_transform(feature_df)
        feature_df_scaled = pd.DataFrame(scaled_features, columns=self.scoring_features, index=feature_df.index)
        
        return feature_df_scaled
    
    def calculate_raw_scores(self, scaled_df):
        """Calculate raw scores based on feature weights"""
        raw_scores = pd.Series(0, index=scaled_df.index)
        
        for feature, weight in self.feature_weights.items():
            if weight > 0:
                # Positive impact features (higher is better)
                raw_scores += scaled_df[feature] * weight
            elif weight < 0:
                # Negative impact features (lower is better)
                raw_scores += (1 - scaled_df[feature]) * abs(weight)
        
        return raw_scores
    
    def adjust_for_bot_behavior(self, raw_scores, scaled_df):
        """Penalize bot-like behavior"""
        # Significant penalty for bot-like behavior
        bot_penalty = scaled_df['bot_likelihood'] * 0.20
        adjusted_scores = raw_scores - bot_penalty
        return adjusted_scores
    
    def adjust_for_liquidations(self, scores, df):
        """Apply liquidation penalties"""
        # Add penalty for liquidations - more severe with more liquidations
        liquidation_counts = df['num_liquidations']
        
        # No penalty for 0 liquidations
        # Moderate penalty for 1 liquidation
        # Severe penalty for multiple liquidations
        liquidation_penalty = np.where(
            liquidation_counts == 0, 0,
            np.where(liquidation_counts == 1, 0.10, 0.25)
        )
        
        adjusted_scores = scores - liquidation_penalty
        return adjusted_scores
    
    def normalize_to_0_100(self, scores):
        """Normalize scores to 0-100 range"""
        min_score = scores.min()
        max_score = scores.max()
        
        # Protect against division by zero
        if max_score == min_score:
            return pd.Series(50, index=scores.index)  # Default to middle score
        
        normalized_scores = (scores - min_score) / (max_score - min_score) * 100
        return normalized_scores
    
    def cluster_analysis(self, df, scaled_features):
        """Perform cluster analysis to identify distinct wallet behavior groups"""
        # Select features for clustering
        cluster_features = [
            'total_transactions', 'borrow_to_deposit_ratio', 'repay_to_borrow_ratio',
            'has_liquidations', 'account_age_days', 'bot_likelihood'
        ]
        
        # Get data for clustering
        cluster_data = scaled_features[cluster_features].copy()
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(cluster_data)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(cluster_data)
        
        # Plot PCA explained variance
        plt.figure(figsize=(10, 5))
        explained_variance = pca.explained_variance_ratio_
        plt.bar(range(len(explained_variance)), explained_variance)
        plt.title('PCA Explained Variance Ratio')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.savefig('pca_explained_variance.png')
        plt.close()
        
        inertias = []
        k_range = range(1, 11)
    
        for k in k_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42)
            kmeans_temp.fit(cluster_data)
            inertias.append(kmeans_temp.inertia_)
    
        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertias, 'bo-')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.grid(True)
        plt.savefig('elbow_plot.png')
        plt.close()
        
        # Plot clustering results
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            pca_result[:, 0], 
            pca_result[:, 1],
            c=clusters,
            cmap='viridis',
            alpha=0.6
        )
        plt.colorbar(scatter)
        plt.title('Wallet Clusters (PCA Visualization)')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        
        # Add cluster centers
        centers_pca = pca.transform(kmeans.cluster_centers_)
        plt.scatter(
            centers_pca[:, 0],
            centers_pca[:, 1],
            c='red',
            marker='x',
            s=200,
            linewidth=3,
            label='Cluster Centers'
        )
        plt.legend()
        plt.savefig('cluster_visualization.png')
        plt.close()
        
        # Feature importance plot for each cluster
        plt.figure(figsize=(15, 8))
        cluster_means = []
        for i in range(5):
            cluster_means.append(
                cluster_data[clusters == i].mean()
            )
        
        cluster_means_df = pd.DataFrame(cluster_means, columns=cluster_features)
        sns.heatmap(
            cluster_means_df,
            cmap='RdYlBu',
            center=0,
            annot=True,
            fmt='.2f'
        )
        plt.title('Feature Importance by Cluster')
        plt.ylabel('Cluster')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('cluster_features_heatmap.png')
        plt.close()
        
        # Calculate and plot silhouette scores for different k
        silhouette_scores = []
        k_range = range(2, 11)
        
        for k in k_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42)
            clusters_temp = kmeans_temp.fit_predict(cluster_data)
            silhouette_scores.append(
                silhouette_score(cluster_data, clusters_temp)
            )
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, silhouette_scores, 'bo-')
        plt.title('Silhouette Score vs Number of Clusters')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.grid(True)
        plt.savefig('silhouette_scores.png')
        plt.close()
        
        print("Generated clustering visualizations:")
        print("- PCA explained variance plot (pca_explained_variance.png)")
        print("- Cluster visualization plot (cluster_visualization.png)")
        print("- Cluster features heatmap (cluster_features_heatmap.png)")
        print("- Silhouette scores plot (silhouette_scores.png)")
        print("- Elbow plot (elbow_plot.png)")
        
        return clusters
    
    def score_wallets(self, df):
        """Score wallets and return scores in the range 0-100"""
        # Preprocess data
        scaled_features = self.preprocess_data(df)
        
        # Calculate raw scores
        raw_scores = self.calculate_raw_scores(scaled_features)
        
        # Adjust for bot behavior
        adjusted_scores = self.adjust_for_bot_behavior(raw_scores, scaled_features)
        
        # Adjust for liquidations
        adjusted_scores = self.adjust_for_liquidations(adjusted_scores, df)
        
        # Ensure scores are not negative
        adjusted_scores = np.maximum(adjusted_scores, 0)
        
        # Normalize to 0-100 range
        final_scores = self.normalize_to_0_100(adjusted_scores)
        
        # Round to integers
        final_scores = final_scores.round().astype(int)
        
        # Convert to DataFrame with wallet addresses
        result_df = pd.DataFrame({
            'wallet_address': df['wallet_address'],
            'credit_score': final_scores
        })
        
        # Perform cluster analysis
        clusters = self.cluster_analysis(df, scaled_features)
        result_df['behavior_cluster'] = clusters
        
        return result_df
    
    def explain_score(self, wallet_address, wallet_df, scores_df):
        """Explain the factors that influenced a wallet's score"""
        # Get the wallet data
        wallet_data = wallet_df[wallet_df['wallet_address'] == wallet_address].iloc[0]
        wallet_score = scores_df[scores_df['wallet_address'] == wallet_address]['credit_score'].iloc[0]
        
        explanation = {
            'wallet_address': wallet_address,
            'credit_score': wallet_score,
            'factors': {}
        }
        
        # Transaction behavior
        explanation['factors']['transaction_behavior'] = {
            'transactions': int(wallet_data['total_transactions']),
            'transactions_per_day': round(wallet_data['transactions_per_day'], 2),
            'unique_assets': int(wallet_data['unique_assets']),
            'activity_diversity': round(wallet_data['activity_diversity'], 2)
        }
        
        # Financial health
        explanation['factors']['financial_health'] = {
            'borrow_to_deposit_ratio': round(wallet_data['borrow_to_deposit_ratio'], 2) 
                if wallet_data['borrow_to_deposit_ratio'] != np.inf else 'Inf',
            'repay_to_borrow_ratio': round(wallet_data['repay_to_borrow_ratio'], 2)
                if wallet_data['repay_to_borrow_ratio'] != np.inf else 'Inf',
            'healthy_repayment': bool(wallet_data['healthy_repayment'])
        }
        
        # Risk factors
        explanation['factors']['risk_factors'] = {
            'liquidations': int(wallet_data['num_liquidations']),
            'borrow_without_repay': bool(wallet_data['borrow_without_repay']),
            'high_risk_factors': int(wallet_data['high_risk_factors'])
        }
        
        # Stability and longevity
        explanation['factors']['stability'] = {
            'account_age_days': int(wallet_data['account_age_days']),
            'tx_size_consistency': round(wallet_data['tx_size_consistency'], 2),
            'bot_likelihood': bool(wallet_data['bot_likelihood'])
        }
        
        # Key strengths and weaknesses
        explanation['key_strengths'] = []
        explanation['key_weaknesses'] = []
        
        # Identify strengths
        if wallet_data['healthy_repayment'] == 1:
            explanation['key_strengths'].append("Consistently repays borrowed amounts")
        
        if wallet_data['account_age_days'] > 180:
            explanation['key_strengths'].append("Long account history shows stability")
        
        if wallet_data['activity_diversity'] > 0.75:
            explanation['key_strengths'].append("Diverse transaction activity")
        
        if wallet_data['longevity_score'] > 0.5:
            explanation['key_strengths'].append("Sustained platform engagement")
        
        # Identify weaknesses
        if wallet_data['num_liquidations'] > 0:
            explanation['key_weaknesses'].append("Has experienced liquidations")
        
        if wallet_data['borrow_without_repay'] == 1:
            explanation['key_weaknesses'].append("Has outstanding unpaid borrows")
        
        if wallet_data['bot_likelihood'] == 1:
            explanation['key_weaknesses'].append("Shows bot-like transaction patterns")
        
        if wallet_data['deposit_withdraw_same_day'] > 3:
            explanation['key_weaknesses'].append("Frequent same-day deposit-withdrawals")
        
        return explanation

def main():
    # Load wallet features
    wallet_df = pd.read_csv('wallet_features.csv')
    
    # Initialize the credit scorer
    scorer = CompoundCreditScorer()
    
    # Score wallets
    scores_df = scorer.score_wallets(wallet_df)
    
    # Get top 1000 wallets by score
    top_wallets = scores_df.sort_values('credit_score', ascending=False).head(1000)
    
    # Save top wallets to CSV
    top_wallets.to_csv('top_1000_wallets.csv', index=False)
    print(f"Saved top 1000 wallets to top_1000_wallets.csv")
    
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
    
    # Save wallet analyses to a JSON file
    import json
    with open('wallet_analyses.json', 'w') as f:
        json.dump(wallet_analyses, f, indent=2)
    
    print(f"Saved analyses for 10 wallets to wallet_analyses.json")
    
    # Print score distribution statistics
    print("\nScore distribution:")
    print(f"Mean score: {scores_df['credit_score'].mean():.2f}")
    print(f"Median score: {scores_df['credit_score'].median():.2f}")
    print(f"Min score: {scores_df['credit_score'].min()}")
    print(f"Max score: {scores_df['credit_score'].max()}")
    
    # Plot score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(scores_df['credit_score'], bins=20, kde=True)
    plt.title('Distribution of Credit Scores')
    plt.xlabel('Credit Score')
    plt.ylabel('Count')
    plt.savefig('score_distribution.png')
    print("Saved score distribution plot to score_distribution.png")
    
    # Analyze relationship between features and scores
    plt.figure(figsize=(12, 10))
    
    # Select key features to visualize
    key_features = [
        'total_transactions',
        'repay_to_borrow_ratio', 
        'account_age_days',
        'has_liquidations',
        'borrow_without_repay',
        'activity_diversity'
    ]
    
    # Create scatter plots for key features vs credit score
    for i, feature in enumerate(key_features):
        plt.subplot(3, 2, i+1)
        plt.scatter(
            wallet_df[feature], 
            scores_df['credit_score'], 
            alpha=0.3, 
            s=10
        )
        plt.title(f'{feature} vs Credit Score')
        plt.xlabel(feature)
        plt.ylabel('Credit Score')
    
    plt.tight_layout()
    plt.savefig('feature_score_relationships.png')
    print("Saved feature relationships plot to feature_score_relationships.png")

if __name__ == "__main__":
    main()