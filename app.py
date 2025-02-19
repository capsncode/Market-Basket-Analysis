import streamlit as st
import pandas as pd
import numpy as np
from src.analyzer import MarketBasketAnalyzer
from src.utils import generate_sample_data, format_rule_for_display

st.set_page_config(page_title="Market Basket Analysis", layout="wide")

st.title("Market Basket Analysis Dashboard")
st.write("I-upload ang iyong transaction data o gumamit ng sample data para sa analysis")

# Mga tabs para sa navigation
tab1, tab2, tab3 = st.tabs(["Data Upload", "Analysis", "Report"])

# Data Upload Tab
with tab1:
    st.header("Data Selection")
    
    data_option = st.radio(
        "Pumili ng data source:",
        ["Upload CSV", "Use Sample Data"]
    )
    
    if data_option == "Upload CSV":
        uploaded_file = st.file_uploader("I-upload ang transaction data CSV", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df.rename(columns={'Member_number': 'transaction_id', 'itemDescription': 'product_id'}, inplace=True)
            st.success(f"Na-upload na ang file! {len(df)} rows ang na-detect.")
        else:
            df = None
    else:
        st.write("Gagawa ng sample transaction data")
        num_transactions = st.slider("Ilang transactions?", 100, 5000, 1000)
        
        if st.button("Generate Sample Data"):
            df = generate_sample_data(num_transactions)
            df.rename(columns={'Member_number': 'transaction_id', 'itemDescription': 'product_id'}, inplace=True)
            st.success(f"Na-generate na ang {len(df)} rows ng sample data!")
        else:
            df = None
    
    if df is not None:
        st.subheader("Data Preview")
        st.dataframe(df.head(10))
        st.session_state['transaction_data'] = df

# Analysis Tab
with tab2:
    st.header("Market Basket Analysis")
    
    if 'transaction_data' not in st.session_state:
        st.warning("Wala pang data na na-upload o na-generate. Pumunta muna sa 'Data Upload' tab.")
    else:
        df = st.session_state['transaction_data']
        
        analyzer = MarketBasketAnalyzer(df)
        analyzer.preprocess_data()
        
        st.subheader("Dataset Overview")
        insights = analyzer.generate_insights()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Transactions", f"{insights.get('total_transactions', 0):,}")
        with col2:
            st.metric("Unique Customers", f"{insights.get('unique_customers', 0):,}")
        with col3:
            st.metric("Unique Products", f"{insights.get('unique_products', 0):,}")
        
        st.subheader("Analysis Parameters")
        col1, col2 = st.columns(2)
        with col1:
            min_support = st.slider("Minimum Support", 0.01, 0.5, 0.05)
        with col2:
            min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.5)
        
        if st.button("Run Analysis"):
            with st.spinner("Nag-rurun ang analysis..."):
                st.session_state['binary_matrix'] = analyzer.create_binary_matrix()
                st.session_state['frequent_itemsets'] = analyzer.find_frequent_itemsets(min_support=min_support)
                st.session_state['rules'] = analyzer.generate_rules(min_confidence=min_confidence)
                st.success(f"Tapos na ang analysis! {len(st.session_state['rules'])} rules ang nakita.")
        
        if 'rules' in st.session_state and not st.session_state['rules'].empty:
            st.subheader("Visualizations")
            viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Product Frequency", "Co-occurrence Heatmap", "Association Network"])
            
            with viz_tab1:
                fig_freq = analyzer.plot_product_frequency()
                st.pyplot(fig_freq)
                
            with viz_tab2:
                fig_heatmap = analyzer.plot_association_heatmap()
                if fig_heatmap:
                    st.pyplot(fig_heatmap)
            
            with viz_tab3:
                min_lift = st.slider("Minimum Lift for Network", 1.0, 5.0, 1.5)
                fig_network = analyzer.create_network_graph(min_confidence=min_confidence, min_lift=min_lift)
                st.pyplot(fig_network)
            
            st.subheader("Top Association Rules")
            st.dataframe(st.session_state['rules'].nlargest(10, 'lift'))
            st.session_state['analyzer'] = analyzer

# Report Tab
with tab3:
    st.header("Analysis Report")
    
    if 'analyzer' not in st.session_state:
        st.warning("Wala pang completed analysis. Pumunta muna sa 'Analysis' tab at i-run ang analysis.")
    else:
        analyzer = st.session_state['analyzer']
        report = analyzer.generate_report()
        
        st.text_area("Complete Report", report, height=400)
        
        if st.button("Download Report"):
            st.download_button("Download as Text", report, file_name="market_basket_analysis_report.txt", mime="text/plain")
