import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import streamlit as st

def create_loss_curve_plot(loss_history, title="Loss Curve"):
    """Create a loss curve plot using Plotly"""
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(loss_history) + 1)),
        y=loss_history,
        mode='lines+markers',
        name='Loss',
        line=dict(color='blue', width=3),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Iteration",
        yaxis_title="Loss",
        height=400,
        showlegend=False
    )
    
    return fig

def create_confusion_matrix_plot(cm, class_names=None):
    """Create a confusion matrix plot using Plotly"""
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        showscale=True,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400
    )
    
    return fig

def create_feature_importance_plot(importances, feature_names, title="Feature Importance"):
    """Create a feature importance bar plot"""
    
    # Sort features by importance
    sorted_indices = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sorted_names,
        y=sorted_importances,
        marker_color='skyblue',
        name='Importance'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Features",
        yaxis_title="Importance",
        height=400,
        xaxis={'tickangle': -45}
    )
    
    return fig

def create_scatter_plot_2d(X, y, title="2D Scatter Plot", feature_names=None):
    """Create a 2D scatter plot with color-coded points"""
    
    if feature_names is None:
        feature_names = ['Feature 1', 'Feature 2']
    
    fig = go.Figure()
    
    unique_labels = np.unique(y)
    colors = px.colors.qualitative.Set1[:len(unique_labels)]
    
    for i, label in enumerate(unique_labels):
        mask = y == label
        fig.add_trace(go.Scatter(
            x=X[mask, 0],
            y=X[mask, 1],
            mode='markers',
            name=f'Class {label}',
            marker=dict(
                color=colors[i],
                size=8,
                opacity=0.7
            )
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=feature_names[0],
        yaxis_title=feature_names[1],
        height=500
    )
    
    return fig

def create_regression_plot(X, y, y_pred, title="Regression Plot"):
    """Create a regression plot showing actual vs predicted values"""
    
    fig = go.Figure()
    
    # Actual vs Predicted scatter
    fig.add_trace(go.Scatter(
        x=y,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(color='blue', opacity=0.6)
    ))
    
    # Perfect prediction line
    min_val = min(min(y), min(y_pred))
    max_val = max(max(y), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        height=400
    )
    
    return fig

def create_residual_plot(y_true, y_pred, title="Residual Plot"):
    """Create a residual plot"""
    
    residuals = y_true - y_pred
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        name='Residuals',
        marker=dict(color='green', opacity=0.6)
    ))
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Values",
        yaxis_title="Residuals",
        height=400
    )
    
    return fig

def create_decision_boundary_2d(X, y, model, title="Decision Boundary", feature_names=None):
    """Create a 2D decision boundary plot"""
    
    if feature_names is None:
        feature_names = ['Feature 1', 'Feature 2']
    
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    try:
        # Try to get prediction probabilities
        if hasattr(model, 'predict_proba'):
            Z = model.predict_proba(mesh_points)[:, 1]
        else:
            Z = model.predict(mesh_points)
    except:
        # Fallback to simple predictions
        Z = model.predict(mesh_points)
    
    Z = Z.reshape(xx.shape)
    
    fig = go.Figure()
    
    # Add contour for decision boundary
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, h),
        y=np.arange(y_min, y_max, h),
        z=Z,
        colorscale='RdYlBu',
        opacity=0.3,
        showscale=True,
        name='Decision Boundary'
    ))
    
    # Add data points
    unique_labels = np.unique(y)
    colors = px.colors.qualitative.Set1[:len(unique_labels)]
    
    for i, label in enumerate(unique_labels):
        mask = y == label
        fig.add_trace(go.Scatter(
            x=X[mask, 0],
            y=X[mask, 1],
            mode='markers',
            name=f'Class {label}',
            marker=dict(
                color=colors[i],
                size=8,
                opacity=0.8
            )
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=feature_names[0],
        yaxis_title=feature_names[1],
        height=500
    )
    
    return fig

def create_learning_curve_plot(train_sizes, train_scores, val_scores, title="Learning Curve"):
    """Create a learning curve plot"""
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig = go.Figure()
    
    # Training scores
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=train_mean,
        mode='lines+markers',
        name='Training Score',
        line=dict(color='blue'),
        error_y=dict(
            type='data',
            array=train_std,
            visible=True
        )
    ))
    
    # Validation scores
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=val_mean,
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='red'),
        error_y=dict(
            type='data',
            array=val_std,
            visible=True
        )
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Training Set Size",
        yaxis_title="Score",
        height=400
    )
    
    return fig

def create_cluster_plot_2d(X, labels, centers=None, title="Cluster Plot", feature_names=None):
    """Create a 2D cluster visualization"""
    
    if feature_names is None:
        feature_names = ['Feature 1', 'Feature 2']
    
    fig = go.Figure()
    
    unique_labels = np.unique(labels)
    colors = px.colors.qualitative.Set1[:len(unique_labels)]
    
    # Plot data points
    for i, label in enumerate(unique_labels):
        mask = labels == label
        fig.add_trace(go.Scatter(
            x=X[mask, 0],
            y=X[mask, 1],
            mode='markers',
            name=f'Cluster {label}',
            marker=dict(
                color=colors[i],
                size=8,
                opacity=0.7
            )
        ))
    
    # Plot cluster centers if provided
    if centers is not None:
        fig.add_trace(go.Scatter(
            x=centers[:, 0],
            y=centers[:, 1],
            mode='markers',
            name='Centroids',
            marker=dict(
                color='black',
                symbol='x',
                size=15,
                line=dict(width=2)
            )
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=feature_names[0],
        yaxis_title=feature_names[1],
        height=500
    )
    
    return fig
