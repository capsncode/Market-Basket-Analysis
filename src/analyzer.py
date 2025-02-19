import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from mlxtend.frequent_patterns import fpgrowth, association_rules

class MarketBasketAnalyzer:
    def __init__(self, df):
        """Initialize the Market Basket Analyzer with a dataset."""
        self.df = df.copy()  # Ensure original dataframe remains unchanged
        print("MarketBasketAnalyzer initialized!")

    def preprocess_data(self):
        """Preprocess data: drop missing values and filter necessary columns."""
        print("Preprocessing data...")
        self.df.dropna(inplace=True)  # Remove missing values

        required_columns = {'transaction_id', 'product_id'}
        if not required_columns.issubset(self.df.columns):
            raise KeyError(f"Dataset must contain 'transaction_id' and 'product_id' columns. Found: {self.df.columns}")

        return self.df

    def generate_insights(self):
        """Generate high-level insights about the dataset."""
        print("Generating insights...")

        total_transactions = self.df['transaction_id'].nunique()
        unique_customers = self.df["customer_id"].nunique() if "customer_id" in self.df.columns else 0
        unique_products = self.df["product_id"].nunique()

        return {
            "total_transactions": total_transactions,
            "unique_customers": unique_customers,
            "unique_products": unique_products,
        }

    def create_binary_matrix(self, max_products=1000):
        """Creates a binary matrix (sparse) for market basket analysis."""
        print("Creating binary matrix...")

        # Limit to the most frequent products to reduce memory usage
        top_products = self.df["product_id"].value_counts().nlargest(max_products).index
        filtered_df = self.df[self.df["product_id"].isin(top_products)]

        # Pivot to create a binary matrix (one-hot encoding)
        binary_matrix = filtered_df.pivot_table(index='transaction_id', 
                                                columns='product_id', 
                                                aggfunc=lambda x: 1, 
                                                fill_value=0)
        
        # Convert to a sparse matrix to save memory
        sparse_matrix = csr_matrix(binary_matrix.values)
        return pd.DataFrame(sparse_matrix.toarray(), columns=binary_matrix.columns, index=binary_matrix.index)

    def find_frequent_itemsets(self, min_support=0.01):
        """Find frequent itemsets using FP-Growth (faster and less memory-intensive)."""
        print(f"Finding frequent itemsets with min_support={min_support}...")

        binary_matrix = self.create_binary_matrix()
        frequent_itemsets = fpgrowth(binary_matrix, min_support=min_support, use_colnames=True)

        return frequent_itemsets

    def generate_rules(self, min_confidence=0.1):
        """Generate association rules from frequent itemsets."""
        print(f"Generating association rules with min_confidence={min_confidence}...")

        frequent_itemsets = self.find_frequent_itemsets()
        if frequent_itemsets.empty:
            print("No frequent itemsets found. Try reducing min_support.")
            return pd.DataFrame()

        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        return rules

    def plot_product_frequency(self, top_n=10):
        """Plot the frequency of the top N most purchased products."""
        print(f"Plotting top {top_n} most frequently purchased products...")

        product_counts = self.df["product_id"].value_counts().head(top_n)

        fig, ax = plt.subplots(figsize=(8, 5))
        product_counts.plot(kind="bar", color="skyblue", ax=ax)

        ax.set_title(f"Top {top_n} Most Frequently Purchased Products")
        ax.set_xlabel("Product ID")
        ax.set_ylabel("Number of Purchases")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        return fig

    def plot_association_heatmap(self):
        """Plot a heatmap of association rules using lift values."""
        print("Plotting association rule heatmap...")

        rules = self.generate_rules()
        if rules.empty:
            print("No rules available for heatmap.")
            return None

        def frozen_set_to_str(fset):
            return ', '.join(map(str, fset))

        rules["antecedents"] = rules["antecedents"].apply(frozen_set_to_str)
        rules["consequents"] = rules["consequents"].apply(frozen_set_to_str)

        pivot_table = rules.pivot(index="antecedents", columns="consequents", values="lift")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)

        ax.set_title("Association Rules Heatmap (Lift Values)")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()
        return fig
