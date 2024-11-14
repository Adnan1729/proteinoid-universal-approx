# metrics.py (src/utils/metrics.py)
from scipy.stats import zscore
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def sigmoid(x):
    """Apply sigmoid function."""
    return 1 / (1 + np.exp(-x))

def calculate_meta_metric(metrics_df):
    """Calculate meta-metric from complexity metrics."""
    weights = {
        'num_nodes': 0.05, 'num_edges': 0.05, 'avg_degree': 0.10,
        'clustering_coefficient': 0.1, 'density': 0.2,
        'degree_entropy': 0.2, 'num_components': 0.10,
        'avg_resistance': 0.2
    }
    
    normalized_scores = metrics_df.apply(zscore)
    weighted_scores = normalized_scores.dot(pd.Series(weights))
    return sigmoid(weighted_scores)

def calculate_classification_metrics(y_true, y_pred):
    """Calculate classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
