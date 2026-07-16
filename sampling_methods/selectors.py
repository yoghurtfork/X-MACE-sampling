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
        - "dbscan_weighted"
        - "uniform_grid"
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
    elif selector_type == "dbscan_weighted":
        return dbscan_weighted(descriptor_matrix, n_to_select, **kwargs)
    elif selector_type == "uniform_grid":
        return dbscan_weighted(descriptor_matrix, n_to_select, **kwargs)
    else:
        raise ValueError(
            f"Unknown selector type: {selector_type}. "
            f"Supported types: 'farthest_point_sampling', 'random_sampling', 'k_means_clustering', 'birch', 'dbscan', 'dbscan_weighted', 'uniform_grid'"
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
    print("n clusters:", n_clusters)

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
    
    if len(np.unique(selected_indices)) < len(selected_indices): print("warning: repeated indices")
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
    print("n clusters:", n_clusters)

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

    if len(np.unique(selected_indices)) < len(selected_indices): print("warning: repeated indices")
    return np.array(selected_indices)    

def dbscan(descriptor_matrix, n_to_select, eps=0.7, min_samples=5):
    '''
    Use DBSCAN to select n_to_select samples.
    It samples equally from clusters, taking the samples closest to cluster centers.
    Noise (label = -1) is ignored.

    eps: DBSCAN epsilon parameter. Default: 0.7
    min_samples: DBSCAN min_samples parameter. Default: 5
    '''

    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(descriptor_matrix)
    
    labels = db.labels_
    unique_labels = [l for l in np.unique(labels) if l != -1]

    n_clusters = len(unique_labels)
    samples_per_cluster = n_to_select // n_clusters
    remainder = n_to_select % n_clusters
    print("n clusters:", n_clusters)
    print("labels:", labels)

    selected_indices = []
    for i, label in enumerate(unique_labels):
        cluster_members_idx = np.where(labels == label)[0]
        cluster_members = descriptor_matrix[cluster_members_idx]
        
        center = cluster_members.mean(axis=0) # Find cluster center

        distances = np.linalg.norm(cluster_members - center, axis=1)

        updated_samples_per_cluster = samples_per_cluster
        if i < remainder:
            updated_samples_per_cluster += 1 # Handles remainder
        
        for _ in range(updated_samples_per_cluster): # Add the closest sample(s) to each cluster center
            closest_local_idx = np.argmin(distances)
            closest_global_idx = cluster_members_idx[closest_local_idx]

            selected_indices.append(closest_global_idx)
            distances[closest_local_idx] = np.inf # Exclude this index from future selections

    if len(np.unique(selected_indices)) < len(selected_indices): print("warning: repeated indices")
    return np.array(selected_indices) 

def dbscan_weighted(descriptor_matrix, n_to_select, eps=0.7, min_samples=5):
    '''
    Use weighted DBSCAN to select n_to_select samples.
    Samples are allocated to clusters in proportion to cluster size.
    Within each cluster, the samples closest to cluster centers are taken.
    Noise (label = -1) is ignored.

    eps: DBSCAN epsilon parameter. Default: 0.7
    min_samples: DBSCAN min_samples parameter. Default: 5
    '''

    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(descriptor_matrix)
    
    labels = db.labels_
    unique_labels = [l for l in np.unique(labels) if l != -1]

    cluster_sizes = np.array([
        np.sum(labels == label)
        for label in unique_labels
    ])

    # Ideal (fractional) allocation
    ideal = cluster_sizes / cluster_sizes.sum() * n_to_select

    # Integer allocation
    allocation = np.floor(ideal).astype(int)

    # Distribute remaining samples to the clusters with the largest fractional remainders
    remainder = n_to_select - allocation.sum()
    fractional = ideal - allocation
    order = np.argsort(fractional)[::-1]

    for i in order[:remainder]:
        allocation[i] += 1

    print("n clusters:", len(unique_labels))
    print("n clusters with samples:", len([i for i in allocation if i != 0]))
    print("labels:", labels)
    print("allocation:", allocation)

    selected_indices = []
    for label, n_allocated in zip(unique_labels, allocation):
        cluster_members_idx = np.where(labels == label)[0]
        cluster_members = descriptor_matrix[cluster_members_idx]
        
        center = cluster_members.mean(axis=0) # Find cluster center

        distances = np.linalg.norm(cluster_members - center, axis=1)
        
        for _ in range(n_allocated): # Add the closest sample(s) to each cluster center
            closest_local_idx = np.argmin(distances)
            closest_global_idx = cluster_members_idx[closest_local_idx]

            selected_indices.append(closest_global_idx)
            distances[closest_local_idx] = np.inf # Exclude this index from future selections

    if len(np.unique(selected_indices)) < len(selected_indices): print("warning: repeated indices")
    return np.array(selected_indices) 

def uniform_grid_sampling(desc_matrix, n_to_select, stagger=False):
    '''
    Uniformly samples n_to_select points over the 2D space of x (eg bond lengths) and y (eg dihedrals).
    
    Splits the sample space into a grid with the same number of intervals along each axes.
    Takes the samples closest to the center of each grid cell.

    stagger: every even row is offset to the left by 1/4 cell and
    every odd row is offset to the right by 1/4 cell. Default: False
    '''

    x = np.asarray([row[0] for row in desc_matrix])
    y = np.asarray([row[1] for row in desc_matrix])

    # Number of grid cells along each axis
    n_bins = int(round(np.sqrt(n_to_select)))

    x_edges = np.linspace(np.min(x), np.max(x), n_bins + 1)
    y_edges = np.linspace(np.min(y), np.max(y), n_bins + 1)
    x_width = (np.max(x) - np.min(x)) / n_bins

    selected_indices = []

    for i in range(n_bins):
        for j in range(n_bins):

            # Include right edge on the last bin
            if i == n_bins - 1:
                x_mask = (
                    (x >= x_edges[i]) &
                    (x <= x_edges[i + 1])
                )
            else:
                x_mask = (
                    (x >= x_edges[i]) &
                    (x < x_edges[i + 1])
                )

            if j == n_bins - 1:
                y_mask = (
                    (y >= y_edges[j]) &
                    (y <= y_edges[j + 1])
                )
            else:
                y_mask = (
                    (y >= y_edges[j]) &
                    (y < y_edges[j + 1])
                )

            candidates = np.where(x_mask & y_mask)[0] # returns points that lie within the grid cell

            if len(candidates) == 0: # if no points lie within the grid cell, continue
                continue

            # Grid cell centre. If stagger, displace x_center
            x_center = 0.5 * (x_edges[i] + x_edges[i + 1])
            if stagger:
                if j % 2 == 0:
                    x_center -= 0.25*x_width
                else:
                    x_center += 0.25*x_width
            
            y_center = 0.5 * (y_edges[j] + y_edges[j + 1])

            # distance of points from grid cell centre
            d = np.sqrt(
                ((x[candidates] - x_center) / 1.0) ** 2 +
                ((y[candidates] - y_center) / 180.0) ** 2
            )

            selected_indices.append(candidates[np.argmin(d)])

    return np.array(selected_indices)