from skmatter.sample_selection import FPS
import numpy as np

def get_selector(selector_type, descriptor_matrix, n_to_select, **kwargs):
    """
    Router to the appropriate selector function. 
        - "furthest_point_sampling"
    """
    if selector_type == "furthest_point_sampling":
        return furthest_point_sampling(descriptor_matrix, n_to_select, **kwargs)
    else:
        raise ValueError(
            f"Unknown selector type: {selector_type}. "
            f"Supported types: 'furthest_point_sampling'"
        )


def furthest_point_sampling(descriptor_matrix, n_to_select, initialize=0):
    """   
    Starting from the first sample (or a random one), FPS chooses the next sample 
    to be as far away from it as possible. Repeat until the specified number of 
    samples is selected.
   
    initialize: Index of the first selection, or 'random' to pick a random value. Default: 0
    """
    n_samples = descriptor_matrix.shape[0]
    
    # skmatter FPS implementation does not handle 1D descriptor matrices, so we need to implement a custom version for that case
    if descriptor_matrix.shape[1] < 2:
        # 1D case: flatten descriptor_matrix to 1D for proper distance calculations
        descriptors_1d = descriptor_matrix.flatten()
        selected_indices = []
        
        # 1. Initialize by picking the first or random index
        if initialize == 'random':
            first_idx = np.random.randint(0, n_samples)
        else:
            first_idx = initialize
        
        selected_indices.append(first_idx)
        
        # 2. Iteratively add points
        for _ in range(n_to_select - 1):
            # Calculate distances from all points to the currently selected points
            distances = np.full(n_samples, np.inf)
            
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
