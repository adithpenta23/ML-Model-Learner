import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

def show_page():
    st.title("K-Means Clustering")
    
    st.markdown("""
    ## What is K-Means Clustering?
    
    K-Means is an unsupervised learning algorithm that groups data points into K clusters. 
    It works by finding cluster centers (centroids) that minimize the within-cluster sum of squared distances.
    
    **Key Concepts:**
    - **Unsupervised Learning**: No target labels needed
    - **Centroid-based**: Each cluster is represented by its center point
    - **Iterative Process**: Alternates between assigning points and updating centroids
    - **Distance-based**: Uses Euclidean distance by default
    """)
    
    # Sidebar controls
    st.sidebar.subheader("Parameters")
    
    # Data source selection
    data_source = st.sidebar.radio("Data Source", ["Sample Data"])
    
    if data_source == "Sample Data":
        # Sample data parameters
        n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 300)
        n_features = st.sidebar.slider("Number of Features", 2, 8, 2)
        true_clusters = st.sidebar.slider("True Number of Clusters", 2, 8, 3)
        cluster_std = st.sidebar.slider("Cluster Standard Deviation", 0.5, 3.0, 1.0)
        
        # Generate sample data
        X, y_true = make_blobs(
            n_samples=n_samples,
            centers=true_clusters,
            n_features=n_features,
            cluster_std=cluster_std,
            random_state=42
        )
        
        feature_names = [f'Feature_{i+1}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        
    else:
                return
    
    # K-Means parameters
    k = st.sidebar.slider("Number of Clusters (K)", 2, 10, 3)
    init_method = st.sidebar.selectbox("Initialization Method", ['k-means++', 'random'])
    max_iter = st.sidebar.slider("Max Iterations", 100, 1000, 300)
    n_init = st.sidebar.slider("Number of Initializations", 1, 20, 10)
    
    # Preprocessing
    scale_features = st.sidebar.checkbox("Standardize Features", value=True)
    
    if scale_features:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    # Create and fit K-Means model
    kmeans = KMeans(
        n_clusters=k,
        init=init_method,
        max_iter=max_iter,
        n_init=n_init,
        random_state=42
    )
    
    cluster_labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_
    
    # Calculate metrics
    inertia = kmeans.inertia_
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    
    if y_true is not None:
        ari_score = adjusted_rand_score(y_true, cluster_labels)
    
    # Display results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Clustering Visualization")
        
        if n_features == 2:
            # 2D scatter plot
            fig = go.Figure()
            
            # Plot data points colored by cluster
            colors = px.colors.qualitative.Set1[:k]
            for i in range(k):
                cluster_mask = cluster_labels == i
                fig.add_trace(go.Scatter(
                    x=X[cluster_mask, 0],
                    y=X[cluster_mask, 1],
                    mode='markers',
                    name=f'Cluster {i}',
                    marker=dict(
                        color=colors[i],
                        size=8,
                        opacity=0.7
                    )
                ))
            
            # Plot centroids
            if scale_features:
                # Transform centroids back to original scale
                centroids_orig = scaler.inverse_transform(centroids)
            else:
                centroids_orig = centroids
            
            fig.add_trace(go.Scatter(
                x=centroids_orig[:, 0],
                y=centroids_orig[:, 1],
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
                title="K-Means Clustering Results",
                xaxis_title=feature_names[0],
                yaxis_title=feature_names[1],
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show true clusters if available
            if y_true is not None:
                st.subheader("True vs Predicted Clusters")
                
                fig_comparison = go.Figure()
                
                # True clusters
                for i in range(len(np.unique(y_true))):
                    mask = y_true == i
                    fig_comparison.add_trace(go.Scatter(
                        x=X[mask, 0],
                        y=X[mask, 1],
                        mode='markers',
                        name=f'True Cluster {i}',
                        marker=dict(size=6, opacity=0.6),
                        visible='legendonly'
                    ))
                
                # Predicted clusters (visible by default)
                for i in range(k):
                    cluster_mask = cluster_labels == i
                    fig_comparison.add_trace(go.Scatter(
                        x=X[cluster_mask, 0],
                        y=X[cluster_mask, 1],
                        mode='markers',
                        name=f'Predicted Cluster {i}',
                        marker=dict(size=8, opacity=0.7)
                    ))
                
                fig_comparison.update_layout(
                    title="True vs Predicted Clusters (Toggle in Legend)",
                    xaxis_title=feature_names[0],
                    yaxis_title=feature_names[1],
                    height=400
                )
                
                st.plotly_chart(fig_comparison, use_container_width=True)
        
        else:
            # Feature importance based on cluster separation
            st.subheader("Feature Analysis")
            
            # Calculate variance between clusters for each feature
            feature_importance = []
            for i in range(n_features):
                cluster_means = []
                for cluster_id in range(k):
                    cluster_data = X_scaled[cluster_labels == cluster_id, i]
                    cluster_means.append(np.mean(cluster_data))
                feature_importance.append(np.var(cluster_means))
            
            fig_features = go.Figure()
            fig_features.add_trace(go.Bar(
                x=feature_names,
                y=feature_importance,
                name='Feature Separation'
            ))
            
            fig_features.update_layout(
                title="Feature Importance in Clustering",
                xaxis_title="Features",
                yaxis_title="Variance Between Cluster Means",
                height=400
            )
            
            st.plotly_chart(fig_features, use_container_width=True)
        
        # Elbow method analysis
        st.subheader("Elbow Method Analysis")
        
        k_range = range(1, min(11, len(X)//2))
        inertias = []
        silhouette_scores = []
        
        for k_test in k_range:
            kmeans_test = KMeans(n_clusters=k_test, random_state=42, n_init=10)
            kmeans_test.fit(X_scaled)
            inertias.append(kmeans_test.inertia_)
            
            if k_test > 1:
                sil_score = silhouette_score(X_scaled, kmeans_test.labels_)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
        
        fig_elbow = go.Figure()
        
        # Inertia curve
        fig_elbow.add_trace(go.Scatter(
            x=list(k_range),
            y=inertias,
            mode='lines+markers',
            name='Inertia',
            yaxis='y',
            line=dict(color='blue')
        ))
        
        # Silhouette score curve
        fig_elbow.add_trace(go.Scatter(
            x=list(k_range),
            y=silhouette_scores,
            mode='lines+markers',
            name='Silhouette Score',
            yaxis='y2',
            line=dict(color='red')
        ))
        
        # Highlight current K
        fig_elbow.add_vline(x=k, line_dash="dash", line_color="green", 
                           annotation_text=f"Current K={k}")
        
        fig_elbow.update_layout(
            title="Elbow Method & Silhouette Analysis",
            xaxis_title="Number of Clusters (K)",
            yaxis=dict(title="Inertia", side="left", color="blue"),
            yaxis2=dict(title="Silhouette Score", side="right", overlaying="y", color="red"),
            height=400
        )
        
        st.plotly_chart(fig_elbow, use_container_width=True)
    
    with col2:
        st.subheader("Clustering Metrics")
        
        st.metric("Inertia (WCSS)", f"{inertia:.2f}")
        st.metric("Silhouette Score", f"{silhouette_avg:.4f}")
        
        if y_true is not None:
            st.metric("Adjusted Rand Index", f"{ari_score:.4f}")
        
        st.metric("Number of Iterations", kmeans.n_iter_)
        
        st.subheader("Algorithm Parameters")
        st.write(f"**K (Clusters):** {k}")
        st.write(f"**Initialization:** {init_method}")
        st.write(f"**Max Iterations:** {max_iter}")
        st.write(f"**Feature Scaling:** {'Yes' if scale_features else 'No'}")
        
        st.subheader("Cluster Sizes")
        cluster_sizes = np.bincount(cluster_labels)
        for i, size in enumerate(cluster_sizes):
            st.write(f"**Cluster {i}:** {size} points")
        
        # Silhouette score interpretation
        st.subheader("Score Interpretation")
        if silhouette_avg > 0.7:
            st.success("Excellent clustering structure")
        elif silhouette_avg > 0.5:
            st.info("Good clustering structure")
        elif silhouette_avg > 0.25:
            st.warning("Weak clustering structure")
        else:
            st.error("Poor clustering structure")
    
   
    
    # Step-by-step explanation
    st.subheader("How K-Means Works")
    
    with st.expander("Algorithm Steps"):
        st.markdown(f"""
        **Step 1: Initialize Centroids**
        - Method: {init_method}
        - Randomly place {k} centroids in the feature space
        
        **Step 2: Assign Points to Clusters**
        - Calculate distance from each point to each centroid
        - Assign each point to the nearest centroid
        
        **Step 3: Update Centroids**
        - Move each centroid to the center of its assigned points
        - New centroid = mean of all points in the cluster
        
        **Step 4: Repeat Until Convergence**
        - Repeat steps 2-3 until centroids stop moving
        - Or until maximum iterations ({max_iter}) reached
        
        **Current Model Converged in {kmeans.n_iter_} iterations**
        """)
    
    # Current model explanation
    st.info(f"""
    **Current Clustering Analysis:**
    
    **Silhouette Score: {silhouette_avg:.4f}** - {'Excellent' if silhouette_avg > 0.7 else 'Good' if silhouette_avg > 0.5 else 'Fair' if silhouette_avg > 0.25 else 'Poor'} cluster separation.
    
    **Inertia: {inertia:.2f}** - Within-cluster sum of squared distances (lower is better).
    
    **Convergence: {kmeans.n_iter_} iterations** - {'Fast convergence' if kmeans.n_iter_ < max_iter//3 else 'Normal convergence' if kmeans.n_iter_ < max_iter//2 else 'Slow convergence'}
    
    {f" **ARI Score: {ari_score:.4f}** - Agreement with true clusters ({'Good' if ari_score > 0.7 else 'Fair' if ari_score > 0.3 else 'Poor'})." if y_true is not None else ""}
    """)
