"""This module handles the creation of the spatiotemporal cells"""

import math 
import numpy as np 

class CellGenerator(): 
    """This class handles the generation of spatiotemporal cells"""
    
    def __init__(self, bounding_box_coords, spatial_grid_lat,
                 spatial_grid_lng, temporal_grid_length):
        """
        Initializes the arguments important for the creation of the 
        spatiotemporal 3D cells 
        
        Args:
            bounding_box_coords: (list of floats) Four coordinates representing 
                                  the min lat, min lng, max lat and max lng. 
                                  These represent the bounding box for the 
                                  entire area of interest 
            spatial_grid_lat: (integer) The lat (y-axis) length (in meters) of 
                               every spatial cell 
            spatial_grid_lng: (integer) The lng (x-axis) length (in meters) of 
                               every spatial cell 
            temporal_grid_length: (integer) The length (in minutes) of each 
                                  temporal segment 
        """
        self.bounding_box_coords = bounding_box_coords
        [self.min_lat, self.min_lng, 
         self.max_lat, self.max_lng] = self.bounding_box_coords
        self.spatial_grid_lat = spatial_grid_lat
        self.spatial_grid_lng = spatial_grid_lng
        self.temporal_grid_length = temporal_grid_length
        
        # Constants for later calculations 
        self.__MINUTES_IN_A_DAY = 1440 
        self.__R_EARTH = 6378.137
        self.__M = (1 / ((2 * math.pi / 360) * self.__R_EARTH)) / 1000 
        

    def generate_spatiotemporal_cells(self):
        """
        Generates the spatiotemporal cells
        
        Returns: 
            All the spatiotemporal grid cells stored in a numpy array. Each cell is 
            represented by a dict that stores the information for that cell. 
        """
        # Generate all possible lat and lng and timestamp
        # For simplicity, we use the min_lat in the lng calculation. 
        # Lng calculations require the latitude as well for it to be accurate.
        # For the best accuracy, we should use the current latitude, but that 
        # will increase the complexity to squared. If we just use a static 
        # latitude, the complexity is reduced with only minor inaccuracies 
        cur_lat = self.min_lat 
        cur_lng = self.min_lng 
        cur_time = 0 
        all_lat = []
        all_lng = []
        all_timestamp = []
        while cur_lat < self.max_lat:
            next_lat = self.__add_lat([cur_lat, self.min_lng], 
                                      self.spatial_grid_lat)
            all_lat.append((cur_lat, next_lat))
            cur_lat = next_lat 
        while cur_lng < self.max_lng:
            next_lng = self.__add_lng([self.min_lat, cur_lng], 
                                      self.spatial_grid_lng)
            all_lng.append((cur_lng, next_lng))
            cur_lng = next_lng 
        while cur_time < self.__MINUTES_IN_A_DAY:
            next_time = cur_time + self.temporal_grid_length
            all_timestamp.append((cur_time, next_time-1))
            cur_time = next_time 
        
        # Create the 3D grids 
        all_grids = []
        for i in range(len(all_lat)):
            grid_lat = []
            for j in range(len(all_lng)):
                grid_lng = []
                for k in range(len(all_timestamp)):
                    cell_id = str(i) + "_" + str(j) + "_" + str(k)
                    s_centroid = self.__get_scaled_s(all_lat[i], all_lng[j])
                    t_centroid = self.__get_scaled_t(all_timestamp[k])
                    hit_count = 0 
                    cell_dict = {"cell_id" : cell_id, 
                                 "lat_range" : all_lat[i],
                                 "lng_range" : all_lng[j],
                                 "s_centroid" : s_centroid,
                                 "timestamp_range" : all_timestamp[k],
                                 "t_centroid" : t_centroid,
                                 "hit_count" : hit_count}
                    grid_lng.append(cell_dict) 
                grid_lat.append(grid_lng)
            all_grids.append(grid_lat)
        all_grids = np.array(all_grids)
        return all_grids
            
        
    def __get_scaled_s(self, lat_pair, lng_pair):
        """
        Given a pair of latitudes and a pair of longitudes, get the centroid of 
        that area and scale the centroid coordinates to range 0 to 1.
        
        Args:
            lat_pair: (tuple of floats) A pair of latitudes 
            lng_pair: (tuple of floats) A pair of longitudes 
            
        Returns: 
            A list of two items: scaled_lat, and scaled lng. These two represent
            the centroid of the four arg points. The coordinates of the 
            centroids are scaled to the range between 0 to 1. 
        """
        lat = sum(lat_pair) / len(lat_pair)
        lng = sum(lng_pair) / len(lng_pair)
        scaled_lat = ((lat - self.min_lat) / (self.max_lat - self.min_lat))
        scaled_lng = ((lng - self.min_lng) / (self.max_lng - self.min_lng))
        return [scaled_lat, scaled_lng]
    
    
    def __get_scaled_t(self, time_pair):
        """
        Given a pair of timestamps (in minutes per day), get the middle point 
        and scale it to range 0 to 1. 
        
        Args:
            time_pair: (tuple of integers) A pair of minutes in a day 
            
        Returns:
            A float representing the midpoint of the input pair of minutes. 
            This is scaled to the range 0 to 1. 
        """
        center = (time_pair[1] + time_pair[0]) / 2
        scaled_center = center / self.__MINUTES_IN_A_DAY
        return scaled_center 
    
    
    def __add_lat(self, coordinates, meters):
        """
        Adds some meters to the given latitude 
        
        Args:
            coordinates : (tuple of floats) Latitude and longitude 
            meters      : (integer) How many meters you want to add 
            
        Returns:
            A float representing the new latitude after having the provided 
            amount of meters added to it. Addition means increasing the 
            latitude, i.e. moving the point north. 
        """
        [lat, lng] = coordinates 
        lat_add = meters * self.__M 
        new_lat = lat + lat_add 
        return new_lat  
        
        
    def __add_lng(self, coordinates, meters):
        """
        Adds some meters to the given latitude 
        
        Args:
            coordinates : (tuple of floats) Latitude and longitude 
            meters      : (integer) How many meters you want to add 
            
        Returns:
            A float representing the new longitude after having the provided 
            amount of meters added to it. Addition means increasing the 
            longitude, i.e. moving the point east. 
        """
        [lat, lng] = coordinates 
        lng_add = (meters * self.__M) / math.cos(lat * (math.pi / 180))
        new_lng = lng + lng_add 
        return new_lng 