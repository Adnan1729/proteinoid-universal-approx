def plot_spike_trains(dataframes):
    fig, axes = plt.subplots(1, 5, figsize=(20, 3), sharex=True, sharey=True)
    #fig.suptitle('Spike Trains for All Datasets', fontsize=16, color='black')

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
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
  
# Function to create and plot the multinodal graph
def plot_multinodal_graph(dataframes):
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), subplot_kw={'projection': '3d'})
    #fig.suptitle('Multinodal Graph Transformation of Spike Trains', fontsize=16)
    
    for i, (df, ax) in enumerate(zip(dataframes, axes)):
        G = nx.Graph()
        
        # Calculate F1 values
        t_values = np.arange(1, 21)
        x_t_values = F1(t_values)
        
        # Find points in dataset and create nodes
        selected_points = [find_point_less_than(df, x) for x in x_t_values]
        for j, point in enumerate(selected_points):
            if pd.notna(point):
                G.add_node(point, pos=(point, j, 0))
                d = F2(point)
                connected_points = find_points_with_same_digit(df, point, d)
                for connected_point in connected_points:
                    if connected_point != point:
                        G.add_node(connected_point, pos=(connected_point, j, 1))
                        G.add_edge(point, connected_point)
        
        # Get node positions
        pos = nx.get_node_attributes(G, 'pos')
        
        # Plot edges
        for edge in G.edges():
            x = [pos[edge[0]][0], pos[edge[1]][0]]
            y = [pos[edge[0]][1], pos[edge[1]][1]]
            z = [pos[edge[0]][2], pos[edge[1]][2]]
            ax.plot(x, y, z, c='gray', alpha=0.5)
        
        # Plot nodes
        x = [pos[node][0] for node in G.nodes()]
        y = [pos[node][1] for node in G.nodes()]
        z = [pos[node][2] for node in G.nodes()]
        ax.scatter(x, y, z, c='red', s=20)
        
        ax.set_title(f'Dataset {i+1}')
        ax.set_xlabel('Time')
        ax.set_ylabel('F1 Index')
        ax.set_zlabel('Layer')
        ax.set_xlim(0, df['Time'].max())
        ax.set_ylim(0, 20)
        ax.set_zlim(0, 1)
    
    plt.tight_layout()
    plt.show()

# Plot multinodal graphs
plot_multinodal_graph(dataframes)

