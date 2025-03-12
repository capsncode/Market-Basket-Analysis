# Market Basket Analysis Portfolio Project

A comprehensive market basket analysis tool that identifies product associations and generates insights from transaction data to inform cross-selling and marketing strategies.

## Overview

This project implements a complete market basket analysis pipeline that:
- Processes transaction data to identify product associations
- Uses Apriori algorithm for association rule mining
- Visualizes product relationships through multiple chart types
- Provides actionable business insights through an interactive dashboard

## Features

- **Data Preprocessing**
  - Handles missing values and duplicates
  - Supports flexible input formats
  - Converts raw transaction data to binary format

- **Association Rule Mining**
  - Configurable support and confidence thresholds
  - Rule filtering and ranking by multiple metrics
  - Comprehensive rule evaluation

- **Interactive Visualizations**
  - Product frequency analysis
  - Co-occurrence heatmaps of products
  - Network graphs showing relationship strength
  - Tab-based organization

- **Business Insights**
  - Identifies cross-selling opportunities
  - Highlights frequently purchased product combinations
  - Generates actionable recommendations
  - Exportable reports and data

## Installation

1. **Clone or download the repository**

2. **Create and activate a virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate (Windows)
   venv\Scripts\activate
   
   # Activate (macOS/Linux)
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn mlxtend networkx streamlit
   ```

## Project Structure

```
market-basket-analysis/
├── data/                   # Store your transaction data files
├── src/                    # Core application code
│   ├── analyzer.py         # Market basket analysis implementation
│   └── utils.py           # Helper functions
├── app.py                  # Streamlit web application
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Upload your transaction data**
   - CSV format with at least 'transaction_id' and 'product_id' columns
   - Optional: 'customer_id', 'timestamp' for additional insights
   - Or use the sample data provided for testing

3. **Configure analysis parameters**
   - Adjust minimum support threshold
   - Set minimum confidence level
   - Filter network connections by lift value

4. **Explore the visualizations**
   - Product frequency distribution
   - Co-occurrence heatmap
   - Association network graph

5. **Export results**
   - Download the analysis report as a text file
   - Export association rules as CSV

## Data Format

Your transaction data should include at minimum:
- `transaction_id`: Identifier for each transaction/basket
- `product_id`: Identifier or name of each product

Optional but recommended fields:
- `customer_id`: Identifier for each customer
- `timestamp`: When the transaction occurred

Sample format:
```
transaction_id,product_id
1,bread
1,milk
1,eggs
2,bread
2,milk
```

## Customization

- Modify visualization parameters in the Streamlit interface
- Adjust chart sizes and appearance in analyzer.py
- Add additional metrics or visualizations by extending the MarketBasketAnalyzer class

## Extending the Project

Potential enhancements:
- Time-based association analysis
- Customer segmentation integration
- Product category-level analysis
- Recommendation system development

## License

This project is available for educational and portfolio purposes.

## Contact

For questions or feedback about this portfolio project, please contact me through my portfolio website or GitHub profile.
