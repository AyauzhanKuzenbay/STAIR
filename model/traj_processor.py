"""Handles the processing of trajectory data"""

import numpy as np 

class TrajProcessor():
    """Contains functions that handle trajectory processing"""
    
    def all_traj_to_topk(self, all_traj, topk_weights):
        """
        Given a numpy array of cell IDs, search for the top-k nearest neighbor
        weights for each cell.
        
        Args:
            all_traj: (numpy array) A numpy array of shape (num_traj,traj_len,1) 
                       where each entry in one trajectory is a cell ID. 
            topk_weights: (numpy array) A numpy array of shape 
                          (num_cells, traj_len, k) where each entry is the 
                           weight of a cell to its  k-nearest neighbors. 
        """
        all_traj_k = np.array([np.array([topk_weights[x[0]] for x in y]) \
                               for y in all_traj])
        return all_traj_k
        