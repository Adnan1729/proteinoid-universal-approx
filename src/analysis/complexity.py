# Function F1
def F1(t):
    return 10 + (10 - 2*t) * np.cos(t * np.pi)

# Function F2 (vectorized)
def F2(x):
    frac_part = x - np.floor(x)
    n = np.ceil(np.log10(1 / frac_part)).astype(int)
    return np.floor((10**n) * frac_part)

# Function to find the point in the dataset just less than x(t)
def find_point_less_than(df, value):
    return df[df['Time'] < value]['Time'].max()

# Function to find points with the same first significant digit
def find_points_with_same_digit(df, x, d):
    return df[np.isclose(F2(df['Time']), d)]['Time'].tolist()

def calculate_complexity_metrics(G):
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

def calculate_modified_resistance(G):
    resistances = []
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        if len(subgraph) > 1:
            L = nx.laplacian_matrix(subgraph).toarray()
            L_pseudo_inv = np.linalg.pinv(L)
            resistance = np.mean([L_pseudo_inv[i,i] + L_pseudo_inv[j,j] - 2*L_pseudo_inv[i,j] 
                                  for i in range(len(L)) for j in range(i+1, len(L))])
            resistances.append(resistance)
    
    return np.mean(resistances) if resistances else 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_meta_metric(metrics_df):
    weights = {
        'num_nodes': 0.05, 'num_edges': 0.05, 'avg_degree': 0.10,
        'clustering_coefficient': 0.1, 'density': 0.2,
        'degree_entropy': 0.2, 'num_components': 0.10,
        'avg_resistance': 0.2
    }
    
    normalized_scores = metrics_df.apply(zscore)
    weighted_scores = normalized_scores.dot(pd.Series(weights))
    
    # Apply sigmoid function to map scores to (0, 1) range
    return sigmoid(weighted_scores)
