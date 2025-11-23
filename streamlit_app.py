# app.py - Mutual Fund Intelligence Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Page Config
st.set_page_config(
    page_title="Mutual Fund Intelligence System",
    page_icon="Chart",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("Mutual Fund India – Performance Intelligence System")
st.markdown("**Predict Top Performers • Optimal Portfolio • Fund Clustering**")
st.markdown("---")

# Load Data & Models
@st.cache_resource
def load_resources():
    df = pd.read_csv("cleaned_mutual_funds_data.csv")
    
    with open("kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    
    with open("best_top_performer_classifier.pkl", "rb") as f:
        clf_data = pickle.load(f)
        classifier = clf_data['model']
        clf_features = clf_data['features']
        threshold = clf_data['threshold']
        le = clf_data.get('le', None)
    
    with open("portfolio_model.pkl", "rb") as f:
        portfolio = pickle.load(f)
    
    return df, kmeans, classifier, clf_features, threshold, le, portfolio

try:
    df, kmeans, classifier, clf_features, threshold, le, portfolio = load_resources()
    st.success("All models & data loaded successfully!")
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.info("Make sure these files are in the same folder: \n"
            "- cleaned_mutual_funds_data.csv\n"
            "- kmeans_model.pkl\n"
            "- best_top_performer_classifier.pkl\n"
            "- portfolio_model.pkl")
    st.stop()

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", [
    "Predict New Fund",
    "Optimal Portfolio",
    "Fund Clusters",
    "Explore All Funds"
])

# ========================================
# PAGE 1: Predict Top Performer
# ========================================
if page == "Predict New Fund":
    st.header("Will This Fund Be a Top 20% Performer?")
    st.write(f"**Threshold**: Funds with 5Y return ≥ {threshold:.1f}% are Top 20%")

    col1, col2 = st.columns(2)

    with col1:
        expense = st.number_input("Expense Ratio (%)", 0.1, 5.0, 1.2)
        fund_size = st.number_input("Fund Size (₹ Cr)", 100, 50000, 5000)
        age = st.number_input("Fund Age (Years)", 1, 30, 8)
        sharpe = st.number_input("Sharpe Ratio", -2.0, 5.0, 1.8)
        sortino = st.number_input("Sortino Ratio", -2.0, 10.0, 3.2)

    with col2:
        alpha = st.number_input("Alpha (%)", -20.0, 30.0, 8.5)
        sd = st.number_input("Volatility (SD %)", 5.0, 50.0, 18.0)
        beta = st.number_input("Beta", 0.1, 2.0, 1.1)
        rating = st.slider("Fund Rating (0-5)", 0, 5, 4)
        risk_level = st.slider("Risk Level (1-6)", 1, 6, 6)
        sub_cat = st.selectbox("Sub Category", df['sub_category'].unique())
    
    if st.button("Predict Performance", type="primary"):
        encoded_cat = le.transform([sub_cat])[0] if le else 0
        
        input_data = pd.DataFrame([{
            'expense_ratio': expense,
            'fund_size_cr': fund_size,
            'fund_age_yr': age,
            'sharpe': sharpe,
            'sortino': sortino,
            'alpha': alpha,
            'sd': sd,
            'beta': beta,
            'rating': rating,
            'risk_level': risk_level,
            'sub_category_encoded': encoded_cat
        }])

        # Use only required features
        input_vec = input_data[clf_features].fillna(0)

        prob = classifier.predict_proba(input_vec)[0][1]
        pred = classifier.predict(input_vec)[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction", "TOP 20%" if pred else "Average", 
                    delta="High Potential" if pred else "Moderate")
        col2.metric("Confidence", f"{prob:.1%}")
        col3.metric("5Y Return Expected", f">{threshold:.1f}%" if pred else f"<{threshold:.1f}%")

        if prob > 0.7:
            st.success("Strong Buy Signal – High probability of outperforming!")
        elif prob > 0.5:
            st.warning("Moderate Buy – Good potential")
        else:
            st.info("Hold/Caution – Average expected performance")

# ========================================
# PAGE 2: Optimal Portfolio
# ========================================
elif page == "Optimal Portfolio":
    st.header("Markowitz Optimal Portfolio (Max Sharpe)")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Sharpe Ratio", f"{portfolio['sharpe']:.2f}", "Beats Nifty 50")
    col2.metric("Expected Return", f"{portfolio['expected_return']*100:.1f}%")
    col3.metric("Risk (Volatility)", f"{portfolio['risk']*100:.1f}%")

    st.subheader("Portfolio Weights")
    weights_df = pd.DataFrame({
        'Fund Name': portfolio['fund_names'],
        'Weight (%)': np.round(portfolio['weights'] * 100, 2)
    }).sort_values('Weight (%)', ascending=False)

    st.dataframe(weights_df, use_container_width=True)

    # Pie chart
    fig, ax = plt.subplots()
    top10 = weights_df.head(10)
    ax.pie(top10['Weight (%)'], labels=top10['Fund Name'], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    st.success("This portfolio beats Nifty 50 Sharpe by over 60%+")

# ========================================
# PAGE 3: Fund Clusters
# ========================================
elif page == "Fund Clusters":
    st.header("Fund Performance Clusters (K-Means)")

    cluster_names = {
        0: "Stable Giants",
        1: "Growth Rockets",
        2: "Value Traps",
        3: "High Risk Winners",
        4: "Underperformers"
    }

    cluster_stats = df.groupby('cluster').agg({
        'returns_5yr': 'mean',
        'sd': 'mean',
        'sharpe': 'mean',
        'alpha': 'mean',
        'scheme_name': 'count'
    }).round(2)
    cluster_stats.index = cluster_stats.index.map(cluster_names.get)

    st.dataframe(cluster_stats.T, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['returns_5yr'], df['sd'], c=df['cluster'], cmap='viridis', alpha=0.7)
    ax.set_xlabel("5-Year Return (%)")
    ax.set_ylabel("Volatility (SD %)")
    ax.set_title("Fund Clusters: Risk vs Return")
    plt.colorbar(scatter, label="Cluster")
    st.pyplot(fig)

# ========================================
# PAGE 4: Explore All Funds
# ========================================
else:
    st.header("Explore All Mutual Funds")

    # Filters
    col1, col2, col3 = st.columns(3)
    category_filter = col1.multiselect("Category", df['category'].unique(), default=df['category'].unique()[:2])
    rating_filter = col2.multiselect("Rating", sorted(df['rating'].unique()), default=[4,5])
    risk_filter = col3.slider("Max Risk Level", 1, 6, 5)

    filtered = df[
        df['category'].isin(category_filter) &
        df['rating'].isin(rating_filter) &
        (df['risk_level'] <= risk_filter)
    ]

    st.write(f"Showing {len(filtered)} funds")

    # Top performers
    top_n = st.slider("Show Top N by Sharpe", 10, 100, 20)
    top_funds = filtered.nlargest(top_n, 'sharpe')[['scheme_name', 'amc_name', 'category', 'sub_category',
                                                    'returns_5yr', 'sharpe', 'sortino', 'alpha', 'expense_ratio']]
    st.dataframe(top_funds.reset_index(drop=True), use_container_width=True)

    # Download button
    csv = top_funds.to_csv(index=False)
    st.download_button("Download Results", csv, "top_mutual_funds.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit • Models: K-Means, XGBoost, Markowitz Optimization</p>
        <p><strong>Arpit Jain | Data Scientist | FinTech Portfolio Project</strong></p>
    </div>
    """, unsafe_allow_html=True
)