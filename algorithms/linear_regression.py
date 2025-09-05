import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression
from utils.data_utils import load_sample_data, process_uploaded_data
from utils.plot_utils import create_loss_curve_plot

def show_page():
    st.title("📈 Linear Regression")
    
    st.markdown("""
    ## What is Linear Regression?
    
    Linear regression is a fundamental supervised learning algorithm that models the relationship between 
    a dependent variable (target) and one or more independent variables (features) using a linear equation.
    
    **Key Concepts:**
    - **Objective**: Find the best line that fits through the data points
    - **Cost Function**: Mean Squared Error (MSE)
    - **Optimization**: Gradient Descent or Normal Equation
    - **Output**: Continuous values (regression)
    """)
    
    # Sidebar controls
    st.sidebar.subheader("Parameters")
    
    # Data source selection
    data_source = st.sidebar.radio("Data Source", ["Sample Data", "Upload CSV"])
    
    if data_source == "Sample Data":
        # Sample data parameters
        n_samples = st.sidebar.slider("Number of Samples", 50, 500, 100)
        noise_level = st.sidebar.slider("Noise Level", 0.0, 50.0, 10.0)
        n_features = st.sidebar.slider("Number of Features", 1, 5, 1)
        
        # Generate sample data
        X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                              noise=noise_level, random_state=42)
        
        if n_features == 1:
            df = pd.DataFrame({'X': X.flatten(), 'y': y})
        else:
            feature_cols = [f'Feature_{i+1}' for i in range(n_features)]
            df = pd.DataFrame(X, columns=feature_cols)
            df['y'] = y
    else:
        # File upload
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            df = process_uploaded_data(uploaded_file)
            if df is not None:
                target_col = st.sidebar.selectbox("Select Target Column", df.columns)
                feature_cols = st.sidebar.multiselect("Select Feature Columns", 
                                                    [col for col in df.columns if col != target_col])
                if feature_cols:
                    X = df[feature_cols].values
                    y = df[target_col].values
                    n_features = len(feature_cols)
                else:
                    st.warning("Please select at least one feature column.")
                    return
            else:
                st.warning("Please upload a valid CSV file.")
                return
        else:
            st.warning("Please upload a CSV file.")
            return
    
    # Model parameters
    fit_intercept = st.sidebar.checkbox("Fit Intercept", value=True)
    
    # Create and train model
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mse)
    
    # Display results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Visualization")
        
        if n_features == 1:
            # 2D plot for single feature
            fig = go.Figure()
            
            # Scatter plot of actual data
            fig.add_trace(go.Scatter(
                x=X.flatten(),
                y=y,
                mode='markers',
                name='Actual Data',
                marker=dict(color='blue', opacity=0.6)
            ))
            
            # Regression line
            x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_range_pred = model.predict(x_range)
            
            fig.add_trace(go.Scatter(
                x=x_range.flatten(),
                y=y_range_pred,
                mode='lines',
                name='Regression Line',
                line=dict(color='red', width=3)
            ))
            
            fig.update_layout(
                title="Linear Regression Fit",
                xaxis_title="X",
                yaxis_title="y",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Feature importance for multiple features
            feature_names = feature_cols if data_source == "Upload CSV" else [f'Feature_{i+1}' for i in range(n_features)]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=feature_names,
                y=np.abs(model.coef_),
                name='Feature Importance (|Coefficient|)'
            ))
            
            fig.update_layout(
                title="Feature Importance (Absolute Coefficients)",
                xaxis_title="Features",
                yaxis_title="Absolute Coefficient Value",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Residual plot
        st.subheader("Residual Analysis")
        residuals = y - y_pred
        
        fig_residual = go.Figure()
        fig_residual.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color='green', opacity=0.6)
        ))
        
        # Add horizontal line at y=0
        fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig_residual.update_layout(
            title="Residual Plot",
            xaxis_title="Predicted Values",
            yaxis_title="Residuals",
            height=400
        )
        
        st.plotly_chart(fig_residual, use_container_width=True)
    
    with col2:
        st.subheader("Model Performance")
        
        st.metric("R² Score", f"{r2:.4f}")
        st.metric("RMSE", f"{rmse:.4f}")
        st.metric("MSE", f"{mse:.4f}")
        
        st.subheader("Model Parameters")
        if fit_intercept:
            st.write(f"**Intercept:** {model.intercept_:.4f}")
        
        st.write("**Coefficients:**")
        if n_features == 1:
            st.write(f"Slope: {model.coef_[0]:.4f}")
        else:
            feature_names = feature_cols if data_source == "Upload CSV" else [f'Feature_{i+1}' for i in range(n_features)]
            for i, coef in enumerate(model.coef_):
                st.write(f"{feature_names[i]}: {coef:.4f}")
    
    # Step-by-step explanation
    st.subheader("🔍 Step-by-Step Explanation")
    
    with st.expander("How Linear Regression Works"):
        st.markdown("""
        **Step 1: Model Formulation**
        - For single feature: y = β₀ + β₁x + ε
        - For multiple features: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
        
        **Step 2: Cost Function**
        - We minimize Mean Squared Error: MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
        
        **Step 3: Parameter Estimation**
        - Using Normal Equation: β = (XᵀX)⁻¹Xᵀy
        - Or Gradient Descent for large datasets
        
        **Step 4: Model Evaluation**
        - R² Score: Proportion of variance explained by the model
        - RMSE: Average prediction error in original units
        """)
    
    # Current model explanation
    st.info(f"""
    **Current Model Analysis:**
    
    🎯 **R² Score: {r2:.4f}** - This means your model explains {r2*100:.1f}% of the variance in the target variable.
    
    📏 **RMSE: {rmse:.4f}** - On average, predictions are off by {rmse:.2f} units.
    
    📊 **Residuals:** The residual plot shows the difference between actual and predicted values. 
    Ideally, residuals should be randomly scattered around zero with no clear pattern.
    """)
