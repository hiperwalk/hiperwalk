#TODO: probabilities expects numpy array or matrix
def PlotDistributionProbabilityOnGraph(AdjMatrix, probabilities):
    
    G = nx.from_numpy_matrix(AdjMatrix)
    
    #TODO: set figure size according to graphdimension
    fig_width = 10
    fig_height = 8
    plt.figure(figsize=(fig_width, fig_height))
    
    #calculates vertices positions.
    #needed to do beforehand in order to fix position for multiple steps
    #TODO: check position calculation method
    pos = nx.kamada_kawai_layout(G)
    min_prob = probabilities.min()
    max_prob = probabilities.max()
    
    #drawing graph
    node_size = 500 * len(str(AdjMatrix.shape[0])) #standard size times number of largest label's characters
    nx.draw(G, node_size=node_size, node_color=probabilities, vmin=min_prob, vmax=max_prob,
            cmap='YlOrRd_r', linewidths=1, edgecolors='black',
            font_color='black', with_labels=True, pos=pos)

    #setting colorbar
    sm = plt.cm.ScalarMappable(cmap='YlOrRd_r', norm=plt.Normalize(vmin=min_prob, vmax=max_prob))
    cbar = plt.colorbar(sm, ticks=np.linspace(min_prob, max_prob, num=5),
                       fraction=0.047*fig_height/fig_width, aspect=40)
    cbar.ax.tick_params(labelsize=14, length=7)
