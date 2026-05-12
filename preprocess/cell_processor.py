"""
This class deals with processing the spatiotemporal cells, including finding 
top-k cells and their distances
"""

from scipy import spatial 
import numpy as np

class CellProcessor():
    """
    A class that contains several utility functions for the spatiotemporal cell 
    processing. 
    """
    def get_hot_cells(self, all_cells, hot_cells_threshold):
        """
        Gets all the hot cells only from all_cells. Also, modify the cell IDs 
        so that the IDs now start at 0 
        
        Args:
            all_cells: (list of lists) A nested list where the innermost 
                        element is the individual cell stored as a dict that 
                        contains the important information. 
            hot_cells: (integer) The minimum number of trajectory point mapped 
                        to a cell for the cell to be classified as a hot cell 
                        
        Returns:
            A dict containing all the hot cells 
        """
        hot_cells = {}
        
        # Iterate through all cells   
        for x in all_cells:
            for y in x:
                for cell in y:
                    # Count the number of hits for the cell. Add to dict if 
                    # it passes the threshold
                    if cell['hit_count'] >= hot_cells_threshold:
                        cell_id = cell['cell_id']
                        s_centroid = cell['s_centroid']
                        t_centroid = cell['t_centroid']
                        hot_cells[cell_id] = s_centroid + [t_centroid]
        return hot_cells 


    def split_hot_cells_dict(self, hot_cells):
        """
        Given a dict of hot cells, split it to the key and values. Then, for 
        the keys, create a dict that copies hot_cells' dict as the keys and 
        for the values, have an integer starting from 0 and increments by 1 
        for every key. This allows us to transform the hot_cells string keys to
        an integer key that starts from 0.
        
        Inputs:
            hot_cells: (dict) The dictionary containing the hot cells 
        """
        # Split the dict to the keys and values 
        keys = list(hot_cells.keys())
        values = list(hot_cells.values())
        
        # Create the a lookup dict to convert these old keys to a new keys 
        new_id_lookup_dict = {}
        for i in range (len(keys)):
            new_id_lookup_dict[keys[i]] = i
        return [new_id_lookup_dict, values]
        
        
    def get_top_k_cells(self, all_centroids, k):
        """
        Given a list of all cell centroids, return the index of the top-k 
        closest centroids for each cell, as well as the top-k distance. 
        
        Args:
            all_centroids: (list of lists) List of all centroids. Each centroid 
                            is a list of three elements representing the 
                            scaled lat, lng, and timestamp centroid. 
            k: (integer) How many closest centroids to find 
            
        Returns:
           A list containing two lists. The first is the list of all top-k 
           cells for every cell. The second is the distance to the top-k 
           nearest cells, for every cell. 
        """
        # Uses a KD-Tree for fast top-k querying of the centroids 
        kdtree = spatial.KDTree(all_centroids)
        all_topk_id = []
        all_topk_dist = []
        for centroid in all_centroids:
            # Since the kdtree contains all the centroids, when we query a 
            # centroid, that centroid itself is going to pop up as the closest  
            # centroid with a distance of 0. So we actually query top-k+1 
            # and remove the first element
            [topk_dist, topk_id] = kdtree.query(centroid, k+1)
            all_topk_id.append(topk_id[1:])
            all_topk_dist.append(topk_dist[1:])
        all_topk_id = np.array(all_topk_id)
        all_topk_dist = np.array(all_topk_dist)
        all_topk_dist = self.__convert_dist_to_weight(all_topk_dist)
        return [all_topk_id, all_topk_dist]
        
        
    def __convert_dist_to_weight(self, D, dist_decay_speed=0.8):
        """
        Converts an array of distances between cell centroids to a weight. This 
        conversion makes it so that cells that are closer to each other are 
        given larger weights.
        
        Args:
            D: (numpy array) Matrix of distances between each cell to its 
                top-k neighbors 
                   
        Returns:
            A numpy array of weights. 
        """
        D = np.exp(-D * dist_decay_speed)
        s = np.sum(D, axis=1)
        s = np.array([np.array([x]) for x in s])
        D = np.divide(D, s)
        return D