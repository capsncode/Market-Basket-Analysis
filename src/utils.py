import pandas as pd
import numpy as np

def generate_sample_data(n_transactions=1000):
    """
    Gumawa ng sample transaction data para sa testing at demonstration
    
    Args:
        n_transactions: Ilang transactions ang gagawin
        
    Returns:
        DataFrame ng synthetic transaction data
    """
    products = ['bread', 'milk', 'eggs', 'cheese', 'butter', 'yogurt', 
                'coffee', 'tea', 'sugar', 'flour']
    
    data = []
    for i in range(n_transactions):
        # Random na bilang ng products per transaction (1-5)
        n_products = np.random.randint(1, 6)
        transaction_products = np.random.choice(products, n_products, replace=False)
        
        for product in transaction_products:
            data.append({
                'transaction_id': i,
                'product_id': product,
                'customer_id': np.random.randint(1, 201),
                'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
            })
    
    return pd.DataFrame(data)

def clean_transaction_data(df):
    """
    Linisin ang transaction data bago mag-analyze
    
    Args:
        df: DataFrame ng raw transaction data
        
    Returns:
        Cleaned DataFrame
    """
    # I-check kung kumpleto ang mga required columns
    required_cols = ['transaction_id', 'product_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Kulang ang mga columns na ito: {missing_cols}")
    
    # I-drop ang duplicate transactions
    df_clean = df.drop_duplicates()
    
    # I-handle ang mga null values
    if df_clean.isnull().any().any():
        print(f"May {df_clean.isnull().sum().sum()} null values ang na-drop")
        df_clean = df_clean.dropna()
    
    # I-ensure ang proper data types
    if 'transaction_id' in df_clean.columns:
        df_clean['transaction_id'] = df_clean['transaction_id'].astype(str)
    
    if 'timestamp' in df_clean.columns:
        df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], errors='coerce')
    
    return df_clean

def format_rule_for_display(rule):
    """
    I-format ang association rule para madaling basahin
    
    Args:
        rule: Series containing antecedents, consequents, etc.
        
    Returns:
        Formatted string
    """
    antecedents = ', '.join(list(rule['antecedents']))
    consequents = ', '.join(list(rule['consequents']))
    
    return f"{antecedents} → {consequents} (support: {rule['support']:.3f}, confidence: {rule['confidence']:.3f}, lift: {rule['lift']:.3f})"

if __name__ == "__main__":
    # Test ang sample data generator
    df = generate_sample_data(100)
    print(f"Na-generate na sample data: {len(df)} rows")
    df.to_csv('../data/sample_transactions.csv', index=False)
    print("Na-save na ang data sa '../data/sample_transactions.csv'")