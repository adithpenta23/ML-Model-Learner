import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from utils.data_utils import process_uploaded_data

def show_page():
    st.title("📊 Logistic Regression")
    
    st.markdown("""
    ## What is Logistic Regression?
    
    Logistic regression is a statistical method used for binary and multiclass classification problems. 
    Unlike linear regression, it predicts the probability of class membership using the logistic (sigmoid) function.
    
    **Key Concepts:**
    - **Objective**: Classify data points into discrete categories
    - **Sigmoid Function**: Maps any real number to a value between 0 and 1
    - **Cost Function**: Log-likelihood (Cross-entropy loss)
    - **Output**: Probabilities and class predictions
    """)
    
    # Sidebar controls
    st.sidebar.subheader("Parameters")
    
    # Data source selection
    data_source = st.sidebar.radio("Data Source", ["Sample Data", "Upload CSV"])
    
    if data_source == "Sample Data":
        # Sample data parameters
        n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 300)
        n_features = st.sidebar.slider("Number of Features", 2, 10, 2)
        n_classes = st.sidebar.slider("Number of Classes", 2, 4, 2)
        class_sep = st.sidebar.slider("Class Separation", 0.5, 3.0, 1.0)
        
        # Generate sample data
        X, y = make_classification(
            n_samples=n_samples, 
            n_features=n_features,
            n_classes=n_classes,
            n_redundant=0,
            n_informative=n_features,
            class_sep=class_sep,
            random_state=42
        )
        
        feature_names = [f'Feature_{i+1}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
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
                    feature_names = feature_cols
                    n_classes = len(np.unique(y))
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
    C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0, step=0.01)
    max_iter = st.sidebar.slider("Max Iterations", 100, 2000, 1000)
    solver = st.sidebar.selectbox("Solver", ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'])
    
    # Split data for training
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Create and train model
    model = LogisticRegression(C=C, max_iter=max_iter, solver=solver, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Display results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Visualization")
        
        if n_features == 2 and n_classes == 2:
            # Decision boundary visualization for 2D binary classification
            fig = go.Figure()
            
            # Create mesh for decision boundary
            h = 0.02
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))
            
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            Z = model.predict_proba(mesh_points)[:, 1]
            Z = Z.reshape(xx.shape)
            
            # Add contour plot for decision boundary
            fig.add_trace(go.Contour(
                x=np.arange(x_min, x_max, h),
                y=np.arange(y_min, y_max, h),
                z=Z,
                showscale=True,
                colorscale='RdYlBu',
                opacity=0.3,
                name='Decision Boundary'
            ))
            
            # Add scatter plot for data points
            colors = ['red' if label == 0 else 'blue' for label in y]
            fig.add_trace(go.Scatter(
                x=X[:, 0],
                y=X[:, 1],
                mode='markers',
                marker=dict(color=colors, size=8, opacity=0.8),
                name='Data Points'
            ))
            
            fig.update_layout(
                title="Logistic Regression Decision Boundary",
                xaxis_title=feature_names[0],
                yaxis_title=feature_names[1],
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Feature importance for multi-dimensional data
            if n_classes == 2:
                coefficients = model.coef_[0]
            else:
                coefficients = np.mean(np.abs(model.coef_), axis=0)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=feature_names,
                y=np.abs(coefficients),
                name='Feature Importance (|Coefficient|)'
            ))
            
            fig.update_layout(
                title="Feature Importance",
                xaxis_title="Features",
                yaxis_title="Absolute Coefficient Value",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=[f'Predicted {i}' for i in range(n_classes)],
            y=[f'Actual {i}' for i in range(n_classes)],
            colorscale='Blues',
            showscale=True
        ))
        
        fig_cm.update_layout(
            title="Confusion Matrix",
            height=400
        )
        
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        st.subheader("Model Performance")
        
        st.metric("Accuracy", f"{accuracy:.4f}")
        st.metric("Training Samples", len(X_train))
        st.metric("Test Samples", len(X_test))
        
        # Sigmoid function visualization
        st.subheader("Sigmoid Function")
        z = np.linspace(-6, 6, 100)
        sigmoid = 1 / (1 + np.exp(-z))
        
        fig_sigmoid = go.Figure()
        fig_sigmoid.add_trace(go.Scatter(
            x=z,
            y=sigmoid,
            mode='lines',
            name='σ(z) = 1/(1+e^(-z))',
            line=dict(color='purple', width=3)
        ))
        
        fig_sigmoid.add_hline(y=0.5, line_dash="dash", line_color="red")
        fig_sigmoid.add_vline(x=0, line_dash="dash", line_color="gray")
        
        fig_sigmoid.update_layout(
            title="Sigmoid Function",
            xaxis_title="z",
            yaxis_title="σ(z)",
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig_sigmoid, use_container_width=True)
        
        # Model parameters
        st.subheader("Model Parameters")
        st.write(f"**Regularization (C):** {C}")
        st.write(f"**Solver:** {solver}")
        st.write(f"**Max Iterations:** {max_iter}")
    
    # Classification report
    st.subheader("📊 Detailed Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(4))
    
    # Step-by-step explanation
    st.subheader("🔍 Step-by-Step Explanation")
    
    with st.expander("How Logistic Regression Works"):
        st.markdown("""
        **Step 1: Linear Combination**
        - Compute z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
        
        **Step 2: Sigmoid Transformation**
        - Apply sigmoid function: σ(z) = 1/(1 + e^(-z))
        - This maps z to a probability between 0 and 1
        
        **Step 3: Decision Making**
        - If σ(z) ≥ 0.5, predict class 1
        - If σ(z) < 0.5, predict class 0
        
        **Step 4: Training (Maximum Likelihood)**
        - Minimize log-likelihood (cross-entropy) loss
        - Use gradient descent or other optimization methods
        
        **Step 5: Regularization**
        - Parameter C controls regularization strength
        - Higher C = less regularization, lower C = more regularization
        """)
    
    # Current model explanation
    st.info(f"""
    **Current Model Analysis:**
    
    🎯 **Accuracy: {accuracy:.4f}** - Your model correctly classifies {accuracy*100:.1f}% of test samples.
    
    🔢 **Classes: {n_classes}** - The model is classifying data into {n_classes} different categories.
    
    ⚖️ **Regularization (C={C})**: {'Strong regularization (prevents overfitting)' if C < 1 else 'Weak regularization (allows complex patterns)' if C > 1 else 'Balanced regularization'}
    
    🎲 **Decision Boundary:** The model creates a {'linear' if n_classes == 2 else 'complex'} decision boundary to separate different classes.
    """)
