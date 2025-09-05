import streamlit as st
import numpy as np
import pandas as pd
from algorithms.linear_regression import show_page as linear_regression_page
from algorithms.logistic_regression import show_page as logistic_regression_page  
from algorithms.decision_trees import show_page as decision_trees_page
from algorithms.kmeans import show_page as kmeans_page
from algorithms.neural_networks import show_page as neural_networks_page

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
        linear_regression_page()
    elif algorithm == "Logistic Regression":
        logistic_regression_page()
    elif algorithm == "Decision Trees":
        decision_trees_page()
    elif algorithm == "K-Means Clustering":
        kmeans_page()
    elif algorithm == "Neural Networks":
        neural_networks_page()

if __name__ == "__main__":
    main()
