# visualization.py (src/analysis/visualization.py)
import matplotlib.pyplot as plt
import seaborn as sns

def plot_spike_trains(dataframes):
    """Plot spike trains for all datasets."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 3), sharex=True, sharey=True)
    
    for i, (df, ax) in enumerate(zip(dataframes, axes)):
        ax.scatter(df['Time'], df['Spike'], color='black', marker='|', s=100)
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([0, 1])
        ax.set_title(f'Dataset {i+1}', color='black', fontsize=12)
        ax.set_xlabel('Time (s)', color='black', fontsize=10)
        if i == 0:
            ax.set_ylabel('Spike', color='black', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')
        ax.grid(False)
    
    plt.tight_layout()
    return fig

def plot_complexity_metrics(metrics_df):
    """Plot complexity metrics comparison."""
    metrics_to_plot = [
        'num_nodes', 'num_edges', 'avg_degree', 'clustering_coefficient',
        'density', 'degree_entropy', 'num_components', 'avg_resistance'
    ]
    
    fig, axes = plt.subplots(1, 8, figsize=(24, 4))
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        metrics_df.set_index('Dataset')[metric].plot(kind='bar', ax=ax, color='black', width=0.7)
        ax.set_title(metric, color='black', fontsize=10)
        ax.set_xlabel('Dataset', color='black', fontsize=8)
        ax.set_ylabel('Value', color='black', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='black', labelsize=8)
        ax.grid(False)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, ha='right')
    
    plt.tight_layout()
    return fig
