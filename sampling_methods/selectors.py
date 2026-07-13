from skmatter.sample_selection import FPS
from sklearn.cluster import KMeans, Birch, DBSCAN
import numpy as np

def get_selector(selector_type, descriptor_matrix, n_to_select, **kwargs):
    """
    Router to the appropriate selector function. 
        - "random_sampling"
        - "farthest_point_sampling"
        - "k_means_clustering"
        - "birch"
        - "dbscan"
    """
    if selector_type == "farthest_point_sampling":
        return farthest_point_sampling(descriptor_matrix, n_to_select, **kwargs)
    elif selector_type == "random_sampling":
        return random_sampling(descriptor_matrix, n_to_select)
    elif selector_type == "k_means_clustering":
        return k_means_clustering(descriptor_matrix, n_to_select, **kwargs)
    elif selector_type == "birch":
        return birch(descriptor_matrix, n_to_select, **kwargs)
    elif selector_type == "dbscan":
        return dbscan(descriptor_matrix, n_to_select, **kwargs)
    else:
        raise ValueError(
            f"Unknown selector type: {selector_type}. "
            f"Supported types: 'farthest_point_sampling', 'random_sampling', 'k_means_clustering', 'birch'"
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
    If n_clusters is not specified, it will be set to n_to_select, ie one sample per cluster.
    If n_clusters is specified, it samples equally from clusters, taking the samples closest to cluster centers.

    random_state: Random seed for reproducibility. Default: 42
    """
    
    if n_clusters == "n_to_select":
        n_clusters = n_to_select
    samples_per_cluster = n_to_select // n_clusters
    remainder = n_to_select % n_clusters

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(descriptor_matrix)

    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    selected_indices = []
    for label, center in enumerate(cluster_centers):
        cluster_members_idx = np.where(labels == label)[0]
        cluster_members = descriptor_matrix[cluster_members_idx]

        distances = np.linalg.norm(cluster_members - center, axis=1)
        
        updated_samples_per_cluster = samples_per_cluster
        if label < remainder:
            updated_samples_per_cluster += 1 # Handles remainder

        for _ in range(updated_samples_per_cluster): # Add the closest sample(s) to each cluster center
            closest_local_idx = np.argmin(distances) 
            closest_global_idx = cluster_members_idx[closest_local_idx] # Convert from local index (in cluster_members) to global index (in descriptor_matrix)
            
            selected_indices.append(closest_global_idx)
            distances[closest_local_idx] = np.inf  # Exclude this index from future selections
    
    return np.array(selected_indices)

def birch(descriptor_matrix, n_to_select, n_clusters="n_to_select", threshold=0.001, branching_factor=50):
    """
    Use BIRCH algorithm to select n_to_select samples.
    If n_clusters is not specified, it will be set to n_to_select, ie one sample per cluster.
    If n_clusters is specified, it samples equally from clusters, taking the samples closest to cluster centers.
    
    threshold: Radius which controls the formation of clusters. Default: 0.001
    branching_factor: Maximum number of subclusters in each node. Default: 50
    """

    if n_clusters == "n_to_select":
        n_clusters = n_to_select
    samples_per_cluster = n_to_select // n_clusters
    remainder = n_to_select % n_clusters

    birch = Birch(n_clusters=n_clusters, threshold=threshold, branching_factor=branching_factor)
    birch.fit(descriptor_matrix)
    labels = birch.labels_

    selected_indices = []
    for i, label in enumerate(np.unique(labels)):
        cluster_members_idx = np.where(labels == label)[0]
        cluster_members = descriptor_matrix[cluster_members_idx]
        
        center = cluster_members.mean(axis=0) # Find cluster center

        distances = np.linalg.norm(cluster_members - center, axis=1)
        
        updated_samples_per_cluster = samples_per_cluster
        if i < remainder:
            updated_samples_per_cluster += 1 # Handles remainder

        for _ in range(updated_samples_per_cluster): # Add the closest sample(s) to each cluster center
            closest_local_idx = np.argmin(distances) 
            closest_global_idx = cluster_members_idx[closest_local_idx] # Convert from local index (in cluster_members) to global index (in descriptor_matrix)
            
            selected_indices.append(closest_global_idx)
            distances[closest_local_idx] = np.inf  # Exclude this index from future selections

    return np.array(selected_indices)    

def dbscan(descriptor_matrix, n_to_select, eps=0.7, min_samples=5):
    '''
    Use DBSCAN to select n_to_select samples.
    It samples equally from clusters, taking the samples closest to cluster centers.

    eps: DBSCAN epsilon parameter. Default: 0.7
    min_samples: DBSCAN min_samples parameter. Default: 5
    '''

    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(descriptor_matrix)
    labels = db.labels_
    print("labels:", labels)

    n_clusters = len(np.unique_counts(labels)[0])
    samples_per_cluster = n_to_select // n_clusters
    print("n clusters:", n_clusters)

    selected_indices = []
    for label in np.unique(labels):
        members = np.where(labels == label)[0]
        center = descriptor_matrix[members].mean(axis=0) # Find cluster center
        print("members:", members)
        print("desc matrix members:", descriptor_matrix[members])

        distances = np.linalg.norm(descriptor_matrix - center, axis=1)
        for _ in range(samples_per_cluster): # Add the closest sample(s) to each cluster center
            closest_idx = np.argmin(distances) 
            selected_indices.append(closest_idx)
            distances[closest_idx] = np.inf  # Exclude this index from future selections

    return np.array(selected_indices) 