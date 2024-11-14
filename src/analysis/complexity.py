# complexity.py (src/analysis/complexity.py)
import networkx as nx
from scipy.stats import entropy
import numpy as np

def F1(t):
    """Function F1 for complexity analysis."""
    return 10 + (10 - 2*t) * np.cos(t * np.pi)

def F2(x):
    """Function F2 for complexity analysis."""
    frac_part = x - np.floor(x)
    n = np.ceil(np.log10(1 / frac_part)).astype(int)
    return np.floor((10**n) * frac_part)

def find_point_less_than(df, value):
    """Find point in dataset just less than given value."""
    return df[df['Time'] < value]['Time'].max()

def find_points_with_same_digit(df, x, d):
    """Find points with same first significant digit."""
    return df[np.isclose(F2(df['Time']), d)]['Time'].tolist()

def calculate_modified_resistance(G):
    """Calculate modified resistance for graph."""
    resistances = []
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        if len(subgraph) > 1:
            L = nx.laplacian_matrix(subgraph).toarray()
            L_pseudo_inv = np.linalg.pinv(L)
            resistance = np.mean([
                L_pseudo_inv[i,i] + L_pseudo_inv[j,j] - 2*L_pseudo_inv[i,j]
                for i in range(len(L)) for j in range(i+1, len(L))
            ])
            resistances.append(resistance)
    return np.mean(resistances) if resistances else 0

def calculate_complexity_metrics(G):
    """Calculate complexity metrics for graph."""
    metrics = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'avg_degree': np.mean([d for n, d in G.degree()]),
        'clustering_coefficient': nx.average_clustering(G),
        'density': nx.density(G),
        'num_components': nx.number_connected_components(G)
    }
    
    degree_counts = nx.degree_histogram(G)
    degree_dist = np.array(degree_counts) / sum(degree_counts)
    metrics['degree_entropy'] = entropy(degree_dist)
    metrics['avg_resistance'] = calculate_modified_resistance(G)
    
    return metrics
