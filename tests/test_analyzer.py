import pytest
import pandas as pd
import numpy as np
from src.analyzer import MarketBasketAnalyzer
from src.utils import generate_sample_data

def test_analyzer_initialization():
    """Test kung nag-initialize ng tama ang analyzer"""
    df = generate_sample_data(100)
    analyzer = MarketBasketAnalyzer(df)
    assert analyzer is not None
    assert analyzer.transactions_df is not None
    assert analyzer.binary_matrix is None  # Dapat None pa ito sa initialization

def test_preprocessing():
    """Test kung gumagana ng tama ang preprocessing"""
    data = {
        'transaction_id': [1, 1, 2, 2, 3],
        'product_id': ['bread', 'bread', 'milk', None, 'eggs'],
        'customer_id': [101, 101, 102, 102, 103],
        'timestamp': ['2024-02-18'] * 5
    }
    df = pd.DataFrame(data)
    
    analyzer = MarketBasketAnalyzer(df)
    analyzer.preprocess_data()
    
    # Check kung na-remove ang duplicates at null values
    assert len(analyzer.transactions_df) == 3
    assert not analyzer.transactions_df.isnull().any().any()
    
    # Check kung na-convert ang timestamp
    assert pd.api.types.is_datetime64_any_dtype(analyzer.transactions_df['timestamp'])

def test_binary_matrix():
    """Test kung tama ang pag-create ng binary matrix"""
    data = {
        'transaction_id': [1, 1, 2, 2, 3],
        'product_id': ['bread', 'milk', 'bread', 'eggs', 'milk'],
        'customer_id': [101, 101, 102, 102, 103],
    }
    df = pd.DataFrame(data)
    
    analyzer = MarketBasketAnalyzer(df)
    binary_matrix = analyzer.create_binary_matrix()
    
    # Check kung tama ang shape
    assert binary_matrix.shape == (3, 3)  # 3 transactions, 3 products
    
    # Check kung tama ang values
    assert binary_matrix.iloc[0, binary_matrix.columns.get_indexer(['bread'])].values[0] == True
    assert binary_matrix.iloc[0, binary_matrix.columns.get_indexer(['milk'])].values[0] == True
    assert binary_matrix.iloc[0, binary_matrix.columns.get_indexer(['eggs'])].values[0] == False

def test_frequent_itemsets():
    """Test kung gumagana ang frequent itemset mining"""
    df = generate_sample_data(200)
    analyzer = MarketBasketAnalyzer(df)
    
    analyzer.create_binary_matrix()
    frequent_itemsets = analyzer.find_frequent_itemsets(min_support=0.05)
    
    assert frequent_itemsets is not None
    assert 'support' in frequent_itemsets.columns
    assert len(frequent_itemsets) > 0

def test_rule_generation():
    """Test kung gumagana ang association rule generation"""
    df = generate_sample_data(200)
    analyzer = MarketBasketAnalyzer(df)
    
    analyzer.create_binary_matrix()
    analyzer.find_frequent_itemsets(min_support=0.05)
    rules = analyzer.generate_rules(min_confidence=0.1)
    
    assert rules is not None
    assert 'confidence' in rules.columns
    assert 'lift' in rules.columns
    
    # Check kung may positive lift values
    assert (rules['lift'] > 1).any()

def test_report_generation():
    """Test kung gumagana ang report generation"""
    df = generate_sample_data(100)
    analyzer = MarketBasketAnalyzer(df)
    
    analyzer.preprocess_data()
    analyzer.create_binary_matrix()
    analyzer.find_frequent_itemsets(min_support=0.1)
    analyzer.generate_rules(min_confidence=0.5)
    
    report = analyzer.generate_report()
    
    assert report is not None
    assert isinstance(report, str)
    assert "Dataset Overview" in report
    assert "Association Rules Summary" in report
