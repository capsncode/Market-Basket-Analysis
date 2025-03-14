import streamlit as st
import pandas as pd
from src.analyzer import MarketBasketAnalyzer

st.set_page_config(page_title="Market Basket Analysis", layout="wide")

st.title("Market Basket Analysis Dashboard")


uploaded_file = st.file_uploader("Upload transaction data CSV", type="csv")

use_sample_data = st.checkbox("Use sample data instead")

@st.cache_data
def get_sample_data():
    data = {
        'transaction_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5],
        'product_id': ['bread', 'milk', 'eggs', 'bread', 'milk', 'milk', 'eggs', 'bread', 'milk', 'sugar', 'bread', 'milk', 'eggs', 'coffee']
    }
    return pd.DataFrame(data)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.success(f"Successfully loaded {len(df)} records")
elif use_sample_data:
    df = get_sample_data()
    st.info("Using sample data. Replace with your own data for more meaningful insights.")
else:
    st.info("Please upload a CSV file with transaction data or use sample data.")
    st.write("""
    Your CSV should have at least these columns:
    - transaction_id: Unique identifier for each transaction
    - product_id: Product identifier or name
    
    Optional columns:
    - customer_id: Unique identifier for each customer
    - timestamp: When the transaction occurred
    """)
    st.stop()

with st.expander("View Raw Data"):
    st.dataframe(df.head(10))
    st.text(f"Column names: {', '.join(df.columns)}")

try:
    analyzer = MarketBasketAnalyzer(df)
    analyzer.preprocess_data()

    st.header("Dataset Overview")
    insights = analyzer.generate_insights()

    metrics = []
    metrics.append(("Total Transactions", f"{insights['total_transactions']:,}"))
    if 'unique_customers' in insights:
        metrics.append(("Unique Customers", f"{insights['unique_customers']:,}"))
    metrics.append(("Unique Products", f"{insights['unique_products']:,}"))
    metrics.append(("Avg. Basket Size", f"{insights['avg_basket_size']:.2f}"))

    cols = st.columns(len(metrics))
    for i, (label, value) in enumerate(metrics):
        cols[i].metric(label, value)

    st.header("Analysis Parameters")
    col1, col2 = st.columns(2)
    with col1:
        min_support = st.slider("Minimum Support", 0.01, 0.5, 0.05, 
                              help="Minimum frequency threshold for items to be considered frequent")
    with col2:
        min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.5,
                                 help="Minimum probability threshold for rules to be considered strong")
  
    with st.spinner("Finding frequent itemsets..."):
        itemsets = analyzer.find_frequent_itemsets(min_support=min_support)
    
    if itemsets.empty:
        st.warning(f"No frequent itemsets found with minimum support of {min_support}. Try lowering the support threshold.")
        st.stop()
        
    with st.spinner("Generating association rules..."):
        rules = analyzer.generate_rules(min_confidence=min_confidence)
    
    if rules.empty:
        st.warning(f"No association rules found with minimum confidence of {min_confidence}. Try lowering the confidence threshold.")
        st.stop()
    
    st.header("Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Product Frequency", "Co-occurrence Heatmap", "Association Network"])
    
    with tab1:
        st.subheader("Product Frequency")
        fig_freq = analyzer.plot_product_frequency()
        st.pyplot(fig_freq)
        
    with tab2:
        st.subheader("Co-occurrence Heatmap")
        fig_heatmap = analyzer.plot_association_heatmap()
        st.pyplot(fig_heatmap)
    
    with tab3:
        st.subheader("Association Network")
        min_lift = st.slider("Minimum Lift for Network Graph", 1.0, 5.0, 1.2,
                           help="Higher values show stronger associations only")
        fig_network = analyzer.create_network_graph(min_confidence=min_confidence, min_lift=min_lift)
        st.pyplot(fig_network)

    st.header("Top Association Rules")
    
    display_rules = rules.copy()
    
    def format_itemset(itemset):
        return ', '.join(list(itemset))
    
    display_rules['antecedents_str'] = display_rules['antecedents'].apply(format_itemset)
    display_rules['consequents_str'] = display_rules['consequents'].apply(format_itemset)

    top_rules = display_rules.sort_values('lift', ascending=False).head(10)
   
    top_rules['confidence_str'] = top_rules['confidence'].apply(lambda x: f"{x:.2%}")
    top_rules['lift_str'] = top_rules['lift'].apply(lambda x: f"{x:.2f}")
    top_rules['support_str'] = top_rules['support'].apply(lambda x: f"{x:.4f}")
    
    # Display the formatted rules
    st.dataframe(
        top_rules[['antecedents_str', 'consequents_str', 'support_str', 'confidence_str', 'lift_str']].rename(columns={
            'antecedents_str': 'Antecedents',
            'consequents_str': 'Consequents',
            'support_str': 'Support',
            'confidence_str': 'Confidence',
            'lift_str': 'Lift'
        }),
        use_container_width=True
    )
    
    # Export options
    st.header("Export Results")
    col1, col2 = st.columns(2)
    with col1:
        report = analyzer.generate_report()
        st.download_button(
            "Download Analysis Report",
            report,
            file_name="market_basket_analysis_report.txt",
            mime="text/plain"
        )
    with col2:
        # Ensure we export the original rules with numeric values intact
        rules_csv = rules.to_csv(index=False)
        st.download_button(
            "Download Rules as CSV",
            rules_csv,
            file_name="association_rules.csv",
            mime="text/csv"
        )

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please check that your data has the required columns: transaction_id, product_id")
