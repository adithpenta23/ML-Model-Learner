import streamlit as st
import numpy as np
import pandas as pd
from algorithms import linear_regression, logistic_regression, decision_trees, kmeans, neural_networks

# Configure page
st.set_page_config(
    page_title="ML Algorithm Learning Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Sidebar navigation
    st.sidebar.title("🤖 ML Learning Hub")
    st.sidebar.markdown("Select an algorithm to explore:")
    
    algorithm = st.sidebar.selectbox(
        "Choose Algorithm",
        [
            "Linear Regression",
            "Logistic Regression", 
            "Decision Trees",
            "K-Means Clustering",
            "Neural Networks"
        ]
    )
    
    # Main content area
    if algorithm == "Linear Regression":
        linear_regression.show_page()
    elif algorithm == "Logistic Regression":
        logistic_regression.show_page()
    elif algorithm == "Decision Trees":
        decision_trees.show_page()
    elif algorithm == "K-Means Clustering":
        kmeans.show_page()
    elif algorithm == "Neural Networks":
        neural_networks.show_page()

if __name__ == "__main__":
    main()
