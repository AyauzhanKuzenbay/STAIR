"""Class that handles the reading of files"""

import numpy as np 
import os 
from pathlib import Path 

class FileReader():
    """Class that handles the reading of files"""
    
    def read_data(self, path):
        """
        Reads dataset 
        
        Args:
            path: (string) Path of the dataset. Can either be a .npy file or 
                   a directory of .npy files 
                   
        Returns:
            A numpy array containing the data of the .npy file(s) 
        """
        if os.path.isdir(path):
            return self.read_data_dir(path)
        elif os.path.isfile(path):
            return self.read_npy(path)
        
    
    def read_npy(self, path):
        """
        Reads .npy files 
        
        Args:
            path: (string) The path to the .npy files 
            
        Returns: 
            A numpy array containing the data of the .npy files 
        """
        return np.load(path, allow_pickle = True)
        
        
    def read_data_dir(self, path):
        """
        Given a directory containing .npy files. Read all the files inside 
        and combine them to one dataset array. 
        
        Args:
            path: (string) The path to the directory 
            
        Returns:
            A numpy array containing the concatenation of all .npy files'  data
        """
        all_npy = os.listdir(path)
        all_npy.sort() 
        all_np_arr = self.read_npy(Path(path) / all_npy[0])
        for a_npy in all_npy[1:]:
            fullpath = Path(path) / a_npy
            new_arr = self.read_npy(fullpath)
            all_np_arr = np.concatenate((all_np_arr, new_arr))
        return all_np_arr