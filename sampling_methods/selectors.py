from skmatter.sample_selection import FPS
from sklearn.cluster import KMeans
import numpy as np

def get_selector(selector_type, descriptor_matrix, n_to_select, **kwargs):
    """
    Router to the appropriate selector function. 
        - "random_sampling"
        - "farthest_point_sampling"
        - "k_means_clustering"
    """
    if selector_type == "farthest_point_sampling":
        return farthest_point_sampling(descriptor_matrix, n_to_select, **kwargs)
    elif selector_type == "random_sampling":
        return random_sampling(descriptor_matrix, n_to_select)
    elif selector_type == "k_means_clustering":
        return k_means_clustering(descriptor_matrix, n_to_select, **kwargs)
    else:
        raise ValueError(
            f"Unknown selector type: {selector_type}. "
            f"Supported types: 'farthest_point_sampling', 'random_sampling', 'k_means_clustering'"
        )

def random_sampling(descriptor_matrix, n_to_select):
    """
    Randomly select n_to_select samples.
    """
    options = descriptor_matrix.shape[0]
    if n_to_select > options:
        raise ValueError(
            f"n_to_select ({n_to_select}) cannot be greater than the number of options ({options})."
        )
    return np.random.choice(options, size=n_to_select, replace=False)

def farthest_point_sampling(descriptor_matrix, n_to_select, initialize=0):
    """   
    Starting from the initial sample, choose the next sample to be as far away from it as possible. 
    Repeat, choosing samples farthest away from all already-sampled samples,
    until n_to_select samples are selected.
   
    initialize: Index of the first sample, or 'random' to pick a random value. Default: 0
    """
    
    # skmatter FPS implementation does not handle 1D descriptor matrices, so we need to implement a custom version for that case
    if descriptor_matrix.shape[1] < 2:
        # 1D case: flatten descriptor_matrix to 1D for proper distance calculations
        options = descriptor_matrix.shape[0]
        descriptors_1d = descriptor_matrix.flatten()
        selected_indices = []
        
        # 1. Initialize by picking the first or random index
        if initialize == 'random':
            first_idx = np.random.randint(0, options)
        else:
            first_idx = initialize
        
        selected_indices.append(first_idx)
        
        # 2. Iteratively add points
        for _ in range(n_to_select - 1):
            # Calculate distances from all points to the currently selected points
            distances = np.full(options, np.inf)
            
            for idx in selected_indices:
                # Distance from all points to this selected point
                dists = np.abs(descriptors_1d - descriptors_1d[idx])
                # Keep track of minimum distance for each point
                distances = np.minimum(distances, dists)
            
            # Mark already selected points so they won't be selected again
            distances[selected_indices] = -np.inf
            
            # Pick the point with the maximum minimum distance
            next_idx = np.argmax(distances)
            selected_indices.append(next_idx)
        return np.array(selected_indices)
    
    else:
        selector = FPS(
            n_to_select=n_to_select,
            initialize=initialize,
        )
        selector.fit(descriptor_matrix)
        return selector.selected_idx_

def k_means_clustering(descriptor_matrix, n_to_select, n_clusters="n_to_select", random_state=42):
    """
    Use k-means clustering to select n_to_select samples.
    If n_clusters is specified, it will be used to determine the number of clusters.
    Else by default, n_clusters will be set to n_to_select, ie one sample per cluster.
    """

    if n_clusters == "n_to_select":
        n_clusters = n_to_select
    samples_per_cluster = n_to_select // n_clusters

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(descriptor_matrix)
    cluster_centers = kmeans.cluster_centers_
    
    # Find the closest sample(s) to each cluster center
    selected_indices = []
    for center in cluster_centers:
        distances = np.linalg.norm(descriptor_matrix - center, axis=1)
        for _ in range(samples_per_cluster):
            closest_idx = np.argmin(distances)
            selected_indices.append(closest_idx)
            distances[closest_idx] = np.inf  # Exclude this index from future selections
    
    return np.array(selected_indices)