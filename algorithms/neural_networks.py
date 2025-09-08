import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def show_page():
    st.title("Neural Networks")
    
    st.markdown("""
    ## What are Neural Networks?
    
    Neural networks are computing systems inspired by biological neural networks. They consist of 
    interconnected nodes (neurons) organized in layers that can learn complex patterns in data 
    through a process called backpropagation.
    
    **Key Concepts:**
    - **Layers**: Input → Hidden Layer(s) → Output
    - **Neurons**: Processing units that apply activation functions
    - **Weights & Biases**: Learnable parameters
    - **Backpropagation**: Algorithm for training the network
    """)
    
    # Sidebar controls
    st.sidebar.subheader("Parameters")
    
    # Problem type selection
    problem_type = st.sidebar.radio("Problem Type", ["Classification", "Regression"])
    
    # Data source selection
    data_source = st.sidebar.radio("Data Source", ["Sample Data"])
    
    if data_source == "Sample Data":
        # Sample data parameters
        n_samples = st.sidebar.slider("Number of Samples", 200, 2000, 500)
        n_features = st.sidebar.slider("Number of Features", 2, 20, 4)
        
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
            noise_level = st.sidebar.slider("Noise Level", 0.0, 50.0, 15.0)
            
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

    # Network architecture
    st.sidebar.subheader("Network Architecture")
    
    # Hidden layers configuration
    n_hidden_layers = st.sidebar.slider("Number of Hidden Layers", 1, 5, 2)
    
    hidden_layer_sizes = []
    for i in range(n_hidden_layers):
        size = st.sidebar.slider(f"Hidden Layer {i+1} Size", 5, 200, 100)
        hidden_layer_sizes.append(size)
    
    hidden_layer_sizes = tuple(hidden_layer_sizes)
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    learning_rate_init = st.sidebar.slider("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
    max_iter = st.sidebar.slider("Max Iterations", 100, 2000, 500)
    activation = st.sidebar.selectbox("Activation Function", ['relu', 'tanh', 'logistic'])
    solver = st.sidebar.selectbox("Solver", ['adam', 'lbfgs', 'sgd'])
    
    # Regularization
    alpha = st.sidebar.slider("L2 Regularization (Alpha)", 0.0001, 1.0, 0.0001, format="%.4f")
    
    # Data preprocessing
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Scale features (important for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    if problem_type == "Classification":
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
    else:
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    
    # Display results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Network Architecture Visualization")
        
        # Create network architecture diagram
        fig_network = go.Figure()
        
        # Calculate layer positions
        layer_sizes = [n_features] + list(hidden_layer_sizes) + [n_classes if problem_type == "Classification" else 1]
        layer_names = ['Input'] + [f'Hidden {i+1}' for i in range(len(hidden_layer_sizes))] + ['Output']
        max_neurons = max(layer_sizes)
        
        # Draw neurons and connections
        for layer_idx, (size, name) in enumerate(zip(layer_sizes, layer_names)):
            x = layer_idx * 2
            
            # Draw neurons in this layer
            for neuron_idx in range(size):
                y = (neuron_idx - size/2) * (10/max_neurons) + 5
                
                color = 'lightblue' if layer_idx == 0 else 'lightgreen' if layer_idx == len(layer_sizes)-1 else 'lightyellow'
                
                fig_network.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers',
                    marker=dict(size=15, color=color, line=dict(width=1, color='black')),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Add layer labels
            fig_network.add_annotation(
                x=x, y=-2,
                text=f"{name}<br>({size} neurons)",
                showarrow=False,
                font=dict(size=10)
            )
            
            # Draw connections to next layer
            if layer_idx < len(layer_sizes) - 1:
                next_size = layer_sizes[layer_idx + 1]
                for i in range(min(3, size)):  # Show only a few connections for clarity
                    for j in range(min(3, next_size)):
                        y1 = (i - size/2) * (10/max_neurons) + 5
                        y2 = (j - next_size/2) * (10/max_neurons) + 5
                        
                        fig_network.add_trace(go.Scatter(
                            x=[x, x+2], y=[y1, y2],
                            mode='lines',
                            line=dict(width=0.5, color='gray'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
        
        fig_network.update_layout(
            title="Neural Network Architecture",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=300,
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig_network, use_container_width=True)
        
        # Training loss curve
        st.subheader("Training Progress")
        
        if hasattr(model, 'loss_curve_'):
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=list(range(1, len(model.loss_curve_) + 1)),
                y=model.loss_curve_,
                mode='lines',
                name='Training Loss',
                line=dict(color='blue', width=3)
            ))
            
            fig_loss.update_layout(
                title="Training Loss Curve",
                xaxis_title="Iteration",
                yaxis_title="Loss",
                height=400
            )
            
            st.plotly_chart(fig_loss, use_container_width=True)
        
        # Feature importance (based on input layer weights)
        st.subheader("Feature Importance")
        
        # Calculate feature importance based on average absolute weights from input layer
        input_weights = np.abs(model.coefs_[0]).mean(axis=1)
        
        fig_importance = go.Figure()
        fig_importance.add_trace(go.Bar(
            x=feature_names,
            y=input_weights,
            name='Average Absolute Weight'
        ))
        
        fig_importance.update_layout(
            title="Feature Importance (Input Layer Weights)",
            xaxis_title="Features",
            yaxis_title="Average |Weight|",
            height=400
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
    
    with col2:
        st.subheader("Model Performance")
        
        if problem_type == "Classification":
            st.metric("Accuracy", f"{accuracy:.4f}")
            st.metric("Number of Classes", n_classes if data_source == "Sample Data" else len(np.unique(y)))
        else:
            st.metric("R² Score", f"{r2:.4f}")
            st.metric("RMSE", f"{np.sqrt(mse):.4f}")
        
        st.metric("Training Iterations", model.n_iter_)
        st.metric("Training Samples", len(X_train))
        
        st.subheader("Network Configuration")
        st.write(f"**Architecture:** {n_features} → {' → '.join(map(str, hidden_layer_sizes))} → {n_classes if problem_type == 'Classification' else 1}")
        st.write(f"**Total Parameters:** {sum(layer1 * layer2 + layer2 for layer1, layer2 in zip([n_features] + list(hidden_layer_sizes), list(hidden_layer_sizes) + [n_classes if problem_type == 'Classification' else 1]))}")
        st.write(f"**Activation:** {activation}")
        st.write(f"**Solver:** {solver}")
        st.write(f"**Learning Rate:** {learning_rate_init}")
        
        st.subheader("Activation Functions")
        
        # Show activation function
        x = np.linspace(-3, 3, 100)
        if activation == 'relu':
            y_act = np.maximum(0, x)
            formula = "f(x) = max(0, x)"
        elif activation == 'tanh':
            y_act = np.tanh(x)
            formula = "f(x) = tanh(x)"
        else:  # logistic
            y_act = 1 / (1 + np.exp(-x))
            formula = "f(x) = 1/(1+e^(-x))"
        
        fig_activation = go.Figure()
        fig_activation.add_trace(go.Scatter(
            x=x, y=y_act,
            mode='lines',
            name=activation.title(),
            line=dict(color='purple', width=3)
        ))
        
        fig_activation.update_layout(
            title=f"{activation.title()} Activation",
            xaxis_title="x",
            yaxis_title="f(x)",
            height=250,
            annotations=[dict(x=0, y=max(y_act)*0.8, text=formula, showarrow=False)]
        )
        
        st.plotly_chart(fig_activation, use_container_width=True)
    
    # Step-by-step explanation
    st.subheader("🔍 How Neural Networks Work")
    
    with st.expander("Forward Propagation"):
        st.markdown("""
        **Step 1: Input Layer**
        - Receive input features
        - Each neuron represents one feature
        
        **Step 2: Hidden Layers**
        - Each neuron computes: output = activation(Σ(weight × input) + bias)
        - Information flows from input to output
        
        **Step 3: Output Layer**
        - Final layer produces predictions
        - Classification: probabilities for each class
        - Regression: continuous value
        """)
    
    with st.expander("Backpropagation Training"):
        st.markdown(f"""
        **Step 1: Forward Pass**
        - Input flows through network to generate prediction
        
        **Step 2: Calculate Loss**
        - Compare prediction with actual target
        - {'Cross-entropy loss for classification' if problem_type == 'Classification' else 'Mean squared error for regression'}
        
        **Step 3: Backward Pass**
        - Calculate gradients of loss with respect to weights
        - Use chain rule to propagate error backward
        
        **Step 4: Update Weights**
        - Solver: {solver}
        - Learning rate: {learning_rate_init}
        - L2 regularization: {alpha}
        
        **Current model trained for {model.n_iter_} iterations**
        """)
    
    # Current model explanation
    if problem_type == "Classification":
        convergence_status = "converged" if model.n_iter_ < max_iter else "reached max iterations"
        st.info(f"""
        **Current Neural Network Analysis:**
        
        **Accuracy: {accuracy:.4f}** - Your network correctly classifies {accuracy*100:.1f}% of test samples.
        
        **Architecture: {n_features}-{'-'.join(map(str, hidden_layer_sizes))}-{n_classes}** - {len(hidden_layer_sizes)} hidden layer(s) with {sum(hidden_layer_sizes)} total hidden neurons.
        
        **Training: {convergence_status}** in {model.n_iter_} iterations
        """)
    else:
        convergence_status = "converged" if model.n_iter_ < max_iter else "reached max iterations"
        st.info(f"""
        **Current Neural Network Analysis:**
        
        **R² Score: {r2:.4f}** - Your network explains {r2*100:.1f}% of the variance in the target.
        
        **Architecture: {n_features}-{'-'.join(map(str, hidden_layer_sizes))}-1** - {len(hidden_layer_sizes)} hidden layer(s) for regression.
        
        **RMSE: {np.sqrt(mse):.4f}** - Average prediction error.
        
        **Training: {convergence_status}** in {model.n_iter_} iterations
        """)
