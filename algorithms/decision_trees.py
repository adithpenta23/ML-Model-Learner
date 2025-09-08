import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score


def show_page():
    st.title("Decision Trees")
    
    st.markdown("""
    ## What are Decision Trees?
    
    Decision trees are intuitive machine learning models that make decisions by asking a series of questions.
    They create a tree-like structure where each internal node represents a test on an attribute, 
    each branch represents the outcome of the test, and each leaf represents a class label or prediction.
    
    **Key Concepts:**
    - **Splitting Criteria**: Information Gain, Gini Impurity, MSE
    - **Tree Structure**: Root → Internal Nodes → Leaves
    - **Interpretability**: Easy to understand and visualize
    - **Versatility**: Works for both classification and regression
    """)
    
    # Sidebar controls
    st.sidebar.subheader("Parameters")
    
    # Problem type selection
    problem_type = st.sidebar.radio("Problem Type", ["Classification", "Regression"])
    
    # Data source selection
    data_source = st.sidebar.radio("Data Source", ["Sample Data"])
    
    if data_source == "Sample Data":
        # Sample data parameters
        n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 300)
        n_features = st.sidebar.slider("Number of Features", 2, 8, 4)
        
        if problem_type == "Classification":
            n_classes = st.sidebar.slider("Number of Classes", 2, 5, 3)
            class_sep = st.sidebar.slider("Class Separation", 0.5, 2.0, 1.0)
            
            # Generate classification data
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                n_redundant=0,
                n_informative=n_features,
                class_sep=class_sep,
                random_state=42
            )
        else:
            noise_level = st.sidebar.slider("Noise Level", 0.0, 50.0, 10.0)
            
            # Generate regression data
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                noise=noise_level,
                random_state=42
            )
        
        feature_names = [f'Feature_{i+1}' for i in range(n_features)]
        
    else:
        return 
    
    # Tree parameters
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)
    min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 20, 1)
    
    if problem_type == "Classification":
        criterion = st.sidebar.selectbox("Splitting Criterion", ['gini', 'entropy'])
    else:
        criterion = st.sidebar.selectbox("Splitting Criterion", ['squared_error', 'absolute_error'])
    
    # Split data
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Create and train model
    if problem_type == "Classification":
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        n_classes = len(np.unique(y))
        
    else:
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    
    # Display results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Tree Visualization")
        
        # Create tree plot
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(model, 
                 feature_names=feature_names,
                 filled=True,
                 rounded=True,
                 fontsize=8,
                 ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Feature importance
        st.subheader("Feature Importance")
        importances = model.feature_importances_
        
        fig_importance = go.Figure()
        fig_importance.add_trace(go.Bar(
            x=feature_names,
            y=importances,
            name='Feature Importance'
        ))
        
        fig_importance.update_layout(
            title="Feature Importance in Decision Tree",
            xaxis_title="Features",
            yaxis_title="Importance",
            height=400
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
        if problem_type == "Classification" and n_features == 2:
            # Decision boundary for 2D classification
            st.subheader("Decision Boundary")
            
            # Create mesh
            h = 0.02
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))
            
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            Z = model.predict(mesh_points)
            Z = Z.reshape(xx.shape)
            
            fig_boundary = go.Figure()
            
            # Add contour for decision boundary
            fig_boundary.add_trace(go.Contour(
                x=np.arange(x_min, x_max, h),
                y=np.arange(y_min, y_max, h),
                z=Z,
                colorscale='Viridis',
                opacity=0.3,
                showscale=False
            ))
            
            # Add data points
            fig_boundary.add_trace(go.Scatter(
                x=X[:, 0],
                y=X[:, 1],
                mode='markers',
                marker=dict(
                    color=y,
                    colorscale='Viridis',
                    size=8,
                    showscale=True
                ),
                name='Data Points'
            ))
            
            fig_boundary.update_layout(
                title="Decision Tree Decision Boundary",
                xaxis_title=feature_names[0],
                yaxis_title=feature_names[1],
                height=500
            )
            
            st.plotly_chart(fig_boundary, use_container_width=True)
    
    with col2:
        st.subheader("Model Performance")
        
        if problem_type == "Classification":
            st.metric("Accuracy", f"{accuracy:.4f}")
            st.metric("Number of Classes", n_classes)
        else:
            st.metric("R² Score", f"{r2:.4f}")
            st.metric("RMSE", f"{np.sqrt(mse):.4f}")
        
        st.metric("Tree Depth", model.get_depth())
        st.metric("Number of Leaves", model.get_n_leaves())
        st.metric("Training Samples", len(X_train))
        
        st.subheader("Tree Structure")
        st.write(f"**Max Depth:** {max_depth}")
        st.write(f"**Min Samples Split:** {min_samples_split}")
        st.write(f"**Min Samples Leaf:** {min_samples_leaf}")
        st.write(f"**Criterion:** {criterion}")
        
        # Splitting criteria explanation
        st.subheader("Splitting Criteria")
        if problem_type == "Classification":
            if criterion == "gini":
                st.write("**Gini Impurity:** Measures how often a randomly chosen element would be incorrectly classified")
            else:
                st.write("**Entropy:** Measures the amount of information needed to classify samples")
        else:
            if criterion == "squared_error":
                st.write("**MSE:** Mean Squared Error measures average squared difference from mean")
            else:
                st.write("**MAE:** Mean Absolute Error measures average absolute difference from median")
    
    # Step-by-step explanation
    st.subheader("How Decision Trees Work")
    
    with st.expander("Algorithm Explanation"):
        st.markdown("""
        **Step 1: Choose Best Split**
        - For each feature and threshold, calculate impurity reduction
        - Select the split that maximizes information gain (or minimizes impurity)
        
        **Step 2: Create Tree Nodes**
        - Split the dataset based on the chosen feature and threshold
        - Create left and right child nodes
        
        **Step 3: Recursive Splitting**
        - Repeat the process for each child node
        - Continue until stopping criteria are met (max depth, min samples, etc.)
        
        **Step 4: Leaf Node Predictions**
        - Classification: Majority class in the leaf
        - Regression: Mean value in the leaf
        
        **Step 5: Pruning (Optional)**
        - Remove branches that don't improve validation performance
        - Helps prevent overfitting
        """)
    
    # Current model explanation
    if problem_type == "Classification":
        st.info(f"""
        **Current Tree Analysis:**
        
        **Accuracy: {accuracy:.4f}** - Your tree correctly classifies {accuracy*100:.1f}% of test samples.
        
        **Tree Depth: {model.get_depth()}** - {'This is a shallow tree (good for interpretability)' if model.get_depth() <= 3 else 'This is a deeper tree (more complex patterns)' if model.get_depth() <= 7 else 'This is a very deep tree (risk of overfitting)'}
        
        **Leaves: {model.get_n_leaves()}** - Each leaf represents a final decision rule.
        
        **Top Feature:** {feature_names[np.argmax(importances)]} (Importance: {max(importances):.3f})
        """)
    else:
        st.info(f"""
        **Current Tree Analysis:**
        
        **R² Score: {r2:.4f}** - Your tree explains {r2*100:.1f}% of the variance in the target.
        
        **Tree Depth: {model.get_depth()}** - {'Shallow tree - simple patterns' if model.get_depth() <= 3 else 'Deeper tree - complex patterns' if model.get_depth() <= 7 else 'Very deep tree - potential overfitting'}
        
        **RMSE: {np.sqrt(mse):.4f}** - Average prediction error.

        **Most Important Feature:** {feature_names[np.argmax(importances)]} (Importance: {max(importances):.3f})
        """)
