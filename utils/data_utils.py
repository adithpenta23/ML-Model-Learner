import pandas as pd
import numpy as np
import streamlit as st
from sklearn.datasets import load_iris, load_wine

def load_sample_data(dataset_name="iris"):
    """Load sample datasets for demonstration"""
    if dataset_name == "iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    elif dataset_name == "wine":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    else:
        # Generate random data if dataset not found
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = np.random.randint(0, 3, 100)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])
        df['target'] = y
        return df

