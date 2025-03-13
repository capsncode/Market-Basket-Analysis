import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
from datetime import datetime

class MarketBasketAnalyzer:
    def __init__(self, transactions_df):
        """
        Initialize the analyzer with a DataFrame containing transaction data.
        Required column: transaction_id, product_id
        Optional columns: customer_id, timestamp
        """
        self.transactions_df = transactions_df
        self.binary_matrix = None
        self.frequent_itemsets = None
        self.rules = None
        
        # Validate required columns
        required_columns = ['transaction_id', 'product_id']
        for col in required_columns:
            if col not in self.transactions_df.columns:
                raise ValueError(f"Missing required column: {col}")
        
    def preprocess_data(self):
        """Clean and preprocess the transaction data"""
        # Remove duplicates
        self.transactions_df = self.transactions_df.drop_duplicates()
        
        # Handle missing values
        self.transactions_df = self.transactions_df.dropna(subset=['transaction_id', 'product_id'])
        
        # Convert timestamp if needed
        if 'timestamp' in self.transactions_df.columns:
            self.transactions_df['timestamp'] = pd.to_datetime(self.transactions_df['timestamp'])
            
    def generate_insights(self):
        """Generate basic statistics and insights about the dataset"""
        insights = {
            'total_transactions': self.transactions_df['transaction_id'].nunique(),
            'unique_products': self.transactions_df['product_id'].nunique(),
            'avg_basket_size': self.transactions_df.groupby('transaction_id')['product_id'].count().mean()
        }
        
        # Add optional insights if columns exist
        if 'customer_id' in self.transactions_df.columns:
            insights['unique_customers'] = self.transactions_df['customer_id'].nunique()
        
        if 'timestamp' in self.transactions_df.columns:
            insights['date_range'] = (
                self.transactions_df['timestamp'].min(),
                self.transactions_df['timestamp'].max()
            )
            
        return insights
    
    def create_binary_matrix(self):
        """Convert transactions into a binary matrix format"""
        # Group products by transaction
        transactions_grouped = self.transactions_df.groupby('transaction_id')['product_id'].agg(list)
        
        # Create transaction encoder
        te = TransactionEncoder()
        te_ary = te.fit(transactions_grouped).transform(transactions_grouped)
        
        # Convert to DataFrame
        self.binary_matrix = pd.DataFrame(te_ary, columns=te.columns_)
        return self.binary_matrix
    
    def find_frequent_itemsets(self, min_support=0.01):
        """Apply Apriori algorithm to find frequent itemsets"""
        if self.binary_matrix is None:
            self.create_binary_matrix()
            
        self.frequent_itemsets = apriori(
            self.binary_matrix,
            min_support=min_support,
            use_colnames=True
        )
        return self.frequent_itemsets
    
    def generate_rules(self, min_confidence=0.5):
        """Generate association rules from frequent itemsets"""
        if self.frequent_itemsets is None:
            raise ValueError("Must find frequent itemsets before generating rules")
            
        self.rules = association_rules(
            self.frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence
        )
        return self.rules
    
    def plot_product_frequency(self, top_n=20):
        """Plot top N most frequent products"""
        plt.figure(figsize=(12, 6))
        product_freq = self.transactions_df['product_id'].value_counts().head(top_n)
        sns.barplot(x=product_freq.values, y=product_freq.index)
        plt.title(f'Top {top_n} Most Frequent Products')
        plt.xlabel('Frequency')
        plt.ylabel('Product ID')
        return plt
    
    def plot_association_heatmap(self, top_n=20):
        """Create a heatmap of product co-occurrences"""
        if self.binary_matrix is None:
            self.create_binary_matrix()
            
        # Calculate co-occurrence matrix
        cooc_matrix = self.binary_matrix.T.dot(self.binary_matrix)
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cooc_matrix.iloc[:top_n, :top_n],
            annot=True,
            cmap='YlOrRd',
            fmt='g'
        )
        plt.title('Product Co-occurrence Heatmap')
        return plt
    
    def create_network_graph(self, min_confidence=0.5, min_lift=1.0):
        """Create a network graph of product associations"""
        if self.rules is None:
            self.generate_rules(min_confidence)
            
        # Create network graph
        G = nx.Graph()
        
        # Filter rules
        filtered_rules = self.rules[
            (self.rules['confidence'] >= min_confidence) &
            (self.rules['lift'] >= min_lift)
        ]
        
        if filtered_rules.empty:
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "No associations meet the criteria", 
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=14)
            plt.gca().set_axis_off()
            return plt
            
        # Add edges for rules meeting criteria
        for _, rule in filtered_rules.iterrows():
            antecedents = list(rule['antecedents'])[0]
            consequents = list(rule['consequents'])[0]
            G.add_edge(
                antecedents,
                consequents,
                weight=rule['lift'],
                confidence=rule['confidence']
            )
            
        # Draw the graph
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G)
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color='lightblue',
            node_size=1000,
            font_size=8,
            width=[G[u][v]['weight'] for u, v in G.edges()]
        )
        return plt
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        insights = self.generate_insights()
        
        report = f"""
Market Basket Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. Dataset Overview
------------------
Total Transactions: {insights['total_transactions']:,}
"""
        # Add optional insights if available
        if 'unique_customers' in insights:
            report += f"Unique Customers: {insights['unique_customers']:,}\n"
            
        report += f"""Unique Products: {insights['unique_products']:,}
Average Basket Size: {insights['avg_basket_size']:.2f} items

2. Association Rules Summary
--------------------------
Total Rules Generated: {len(self.rules) if self.rules is not None else 0}
"""
        if self.rules is not None and not self.rules.empty:
            # Sort rules using pandas sort_values for safety
            top_rules = self.rules.sort_values('lift', ascending=False).head(5)
            
            # Function to format an itemset
            def format_itemset(items):
                return ', '.join(list(items))
            
            # Format top rules for report
            rules_text = "Top 5 Rules by Lift:\n"
            for i, (_, rule) in enumerate(top_rules.iterrows(), 1):
                rules_text += f"{i}. {format_itemset(rule['antecedents'])} → {format_itemset(rule['consequents'])}"
                rules_text += f" (Conf: {rule['confidence']:.2f}, Lift: {rule['lift']:.2f})\n"
            
            report += rules_text

        report += """
3. Key Recommendations
---------------------
"""
        # Add recommendations based on rules
        if self.rules is not None and not self.rules.empty:
            # Sort rules safely
            top_rules = self.rules.sort_values('lift', ascending=False).head(3)
            
            for i, (_, rule) in enumerate(top_rules.iterrows(), 1):
                antecedent = ', '.join(list(rule['antecedents']))
                consequent = ', '.join(list(rule['consequents']))
                report += f"{i}. Consider bundling {antecedent} with {consequent} "
                report += f"(Confidence: {rule['confidence']:.2%}, Lift: {rule['lift']:.2f})\n"
        
        return report
