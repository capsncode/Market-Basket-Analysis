import pandas as pd
import numpy as np

def generate_sample_data(n_transactions=1000):
    products = ['bread', 'milk', 'eggs', 'cheese', 'butter', 'yogurt', 'coffee', 'tea', 'sugar', 'flour']

    data = []
    for i in range(n_transactions):
        n_products = np.random.randit(1, 6)
        transaction_products = np.random.choice(products, n_products, replace=False)

        for product in range(n_transactions):
            data.append({
                'transaction_id': i,
                'product_id': product,
                'customer_id': np.random.randint(1, 201),
                'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
            })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_sample_data()
    df.to_csv('../data/sample_transactions.csv', index=False)
