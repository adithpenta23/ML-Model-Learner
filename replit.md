# ML Algorithm Learning Hub

## Overview

This project is an interactive educational web application built with Streamlit that teaches machine learning algorithms through hands-on exploration. The application provides visual demonstrations and interactive parameter tuning for five core ML algorithms: Linear Regression, Logistic Regression, Decision Trees, K-Means Clustering, and Neural Networks. Users can experiment with both synthetic datasets and their own uploaded CSV files, making it an ideal tool for learning and understanding fundamental machine learning concepts.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit-based single-page application with multi-page navigation
- **UI Pattern**: Sidebar navigation with algorithm selection and parameter controls
- **Layout**: Wide layout with expandable sections for data preview and statistics
- **Visualization**: Dual plotting system using both Matplotlib and Plotly for interactive charts

### Application Structure
- **Modular Design**: Each ML algorithm is implemented as a separate module in the `algorithms/` directory
- **Algorithm Modules**: Individual files for each algorithm (linear_regression.py, logistic_regression.py, decision_trees.py, kmeans.py, neural_networks.py)
- **Utility Modules**: Shared functionality split into `data_utils.py` and `plot_utils.py`
- **Main Controller**: `app.py` serves as the entry point and routing logic

### Data Handling Architecture
- **Dual Data Sources**: Support for both synthetic dataset generation and CSV file uploads
- **Data Generation**: Uses scikit-learn's dataset generators (make_regression, make_classification, make_blobs)
- **Data Processing**: Centralized data validation and preprocessing in utility functions
- **Interactive Parameters**: Real-time dataset generation based on user-controlled parameters

### Machine Learning Implementation
- **Library Choice**: Built on scikit-learn for all ML algorithms
- **Algorithm Coverage**: Supervised learning (regression and classification) plus unsupervised clustering
- **Feature Engineering**: Automatic standardization and preprocessing where needed
- **Model Evaluation**: Integrated metrics and visualization for model performance assessment

### Visualization Strategy
- **Dual Plotting System**: Matplotlib for static plots, Plotly for interactive visualizations
- **Educational Focus**: Emphasis on interpretable plots that explain algorithm behavior
- **Real-time Updates**: Dynamic plot generation based on parameter changes
- **Custom Plot Utils**: Reusable plotting functions for consistency across algorithms

## External Dependencies

### Core Framework
- **Streamlit**: Web application framework for rapid deployment and interactive widgets

### Machine Learning Stack
- **scikit-learn**: Primary ML library for algorithms, datasets, and metrics
- **NumPy**: Numerical computing foundation
- **Pandas**: Data manipulation and analysis

### Visualization Libraries
- **Matplotlib**: Static plotting and scientific visualization
- **Plotly**: Interactive plotting with Express and Graph Objects APIs
- **Seaborn**: Statistical data visualization (used for enhanced matplotlib plots)

### Data Processing
- **CSV File Handling**: Built-in pandas CSV reading capabilities
- **Dataset Generators**: scikit-learn synthetic data generation tools
- **StandardScaler**: Feature scaling and normalization from scikit-learn