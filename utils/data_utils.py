import pandas as pd
import numpy as np
import streamlit as st
from sklearn.datasets import load_boston, load_iris, load_wine

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

def process_uploaded_data(uploaded_file):
    """Process uploaded CSV file and return DataFrame"""
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Basic validation
        if df.empty:
            st.error("The uploaded file is empty.")
            return None
        
        if len(df.columns) < 2:
            st.error("The dataset must have at least 2 columns (features and target).")
            return None
        
        # Display basic info about the dataset
        st.info(f"Dataset loaded successfully! Shape: {df.shape}")
        
        # Show first few rows
        with st.expander("Preview Data"):
            st.dataframe(df.head())
            
        # Show basic statistics
        with st.expander("Data Statistics"):
            st.dataframe(df.describe())
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            st.warning("Dataset contains missing values:")
            st.write(missing_values[missing_values > 0])
            
            # Option to handle missing values
            handle_missing = st.selectbox(
                "How to handle missing values?",
                ["Remove rows with missing values", "Fill with mean", "Fill with median"]
            )
            
            if handle_missing == "Remove rows with missing values":
                df = df.dropna()
            elif handle_missing == "Fill with mean":
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
            elif handle_missing == "Fill with median":
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Convert categorical variables to numeric if needed
        categorical_columns = df.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            st.info(f"Found categorical columns: {list(categorical_columns)}")
            
            for col in categorical_columns:
                try:
                    # Try to convert to numeric first
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                    
                    # If still categorical, encode it
                    if df[col].dtype == 'object':
                        df[col] = pd.Categorical(df[col]).codes
                        st.info(f"Encoded categorical column '{col}' to numeric values.")
                except:
                    pass
        
        return df
        
    except Exception as e:
        st.error(f"Error processing the file: {str(e)}")
        return None

def validate_data_for_algorithm(df, algorithm_type):
    """Validate that the data is suitable for the specified algorithm"""
    
    if algorithm_type in ["linear_regression", "logistic_regression", "neural_networks"]:
        # Check for sufficient samples
        if len(df) < 50:
            st.warning("Dataset is quite small. Consider using more samples for better results.")
        
        # Check for sufficient features
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) < 2:
            st.error("Algorithm requires at least 2 numeric columns (features + target).")
            return False
    
    elif algorithm_type == "decision_trees":
        # Decision trees can handle smaller datasets
        if len(df) < 20:
            st.warning("Dataset is very small for decision trees.")
    
    elif algorithm_type == "kmeans":
        # K-means specific validations
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) < 2:
            st.error("K-means requires at least 2 numeric features.")
            return False
        
        if len(df) < 10:
            st.error("Dataset is too small for clustering.")
            return False
    
    return True

def prepare_features_target(df, target_column, feature_columns):
    """Prepare feature matrix X and target vector y"""
    try:
        X = df[feature_columns].values
        y = df[target_column].values
        
        # Check for any remaining non-numeric values
        if not np.issubdtype(X.dtype, np.number):
            st.error("Feature columns contain non-numeric values.")
            return None, None
        
        if not np.issubdtype(y.dtype, np.number):
            # Try to encode target if it's categorical
            if y.dtype == 'object':
                y = pd.Categorical(y).codes
            else:
                st.error("Target column contains non-numeric values.")
                return None, None
        
        return X, y
        
    except Exception as e:
        st.error(f"Error preparing features and target: {str(e)}")
        return None, None
