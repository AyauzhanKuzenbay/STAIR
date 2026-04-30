"""Reads the trajectories from the input trajectory file - OPTIMIZED VERSION"""
from datetime import datetime
from os import listdir
from os.path import isfile, join
from shapely.geometry import Point
from shapely.geometry import Polygon 
import ast 
import numpy as np 

import pathlib 

class FileReader():
    """
    This class handles the processing of the trajectories. It reads the 
    trajectories from the .csv files and outputs the data in the format 
    ready to be used in the training model
    
    OPTIMIZED: Only parses trajectories that are needed, skips invalid ones early
    """
    def __init__(self):
        """Simply initialize important constants"""
        self.__MINUTES_IN_DAY = 1440 
        self.__SECONDS_IN_DAY = 86400
        self.__PORTO_SECOND_INCREMENT = 15 
        self.__SECONDS_IN_MINUTE = 60 
        self.__DEFAULT_TRAIN_SIZE_FRACTION = 0.7
        self.__DEFAULT_VALIDATION_SIZE_FRACTION = 0.2
        self.__DEFAULT_TEST_SIZE_FRACTION = 0.1
        
            
    def read_trajectory_from_file(self, in_path, dataset_mode, 
                                  min_trajectory_length, max_trajectory_length,
                                  bbox_coords, traj_nums): 
        """
        Reads the input_file while doing some pruning which are: removing 
        trajectories that are too short, removing trajectories that are too, 
        long, and removing trajectory points outside the valid area. The way 
        the data is read depends on the data_mode 
        
        Args:
            in_path: (String) The path to the input  
            data_mode: (String) The mode used to read the data. Different 
                       dataset require different ways of reading and 
                       consequently, different data_mode 
            min_trajectory_length: (Integer) The shortest allowable trajectory 
                                   length 
            max_trajectory_length: (Integer) The longest allowable trajectory 
                                   length 
            bbox_coords: (list of floats) Min lat, min lng, max lat and max lng
                         that represents the valid area. Points outside this 
                         area are to be removed 
            traj_nums: (list of integers) For the Porto data, this is a list 
                        containing the number of lines in the .csv trajectory 
                        to be assigned to the training, and validation data 
                        accordingly. For the Didi data, this is a list 
                        containing the number of trajectories to be assigned 
                        to the training and validation. 
            
        Returns:    
            A list of trajectories. Each trajectory is a list consisting of 
            latitude, longitude and timestamp in the form of minutes-in-day
        """
        [min_lat, min_lng, max_lat, max_lng] = bbox_coords
        # FIXED: Shapely expects (x, y) = (longitude, latitude), not (lat, lng)!
        self.bbox = Polygon([(min_lng, min_lat), (max_lng, min_lat), 
                             (max_lng, max_lat), (min_lng, max_lat),
                             (min_lng, min_lat)])
        
        # Read the .csv file line-by-line and process it according to the 
        # data_mode 
        in_file = open(in_path, 'r')
        if dataset_mode == 'porto':
            return(self.__read_porto(in_file, min_trajectory_length, 
                                     max_trajectory_length, traj_nums))
        elif dataset_mode == 'didi':
            return(self.__read_didi(in_file, min_trajectory_length, 
                                     max_trajectory_length, traj_nums))
        else:
            raise ValueError("'" + dataset_mode + "' not supported.")
        in_file.close()
    

    def read_npy(self, input_directory, file_name):
        """
        A general purpose function to read .npy files. If all you need to do is 
        to read an .npy file from somewhere and don't need to do any form of 
        preprocessing, use this. 
        
        Args:
            input_directory: (string) The directory where the file is located 
            file_name: (string) The file name 
            
        Returns:
            A numpy array containing the contents of the .npy file 
        """
        fullpath = pathlib.Path(input_directory) / (file_name + ".npy")
        return np.load(fullpath, allow_pickle = True)


    def __read_porto(self, in_file, min_trajectory_length, 
                     max_trajectory_length, traj_nums):
        """
        Reads the porto trajectory file line-by-line. Also keep track of the 
        actual number of lines read 
        
        Args:
            in_file: (file) The input porto trajectory file 
            min_trajectory_length: (Integer) The shortest allowable trajectory 
                                   length 
            max_trajectory_length: (Integer) The longest allowable trajectory 
                                   length 
            traj_nums: (list of integers) A list containing the number of 
                        trajectories for each training and validation data
        Returns:    
            A list of trajectories, and the actual number of lines read. Each 
            trajectory is a list consisting of latitude, longitude and timestamp 
            in the form of minutes-in-day
        """
        # Throws away the .csv header and then read line-by-line 
        in_file.readline()
        
        # Get the lines into the training, and validation lists
        [num_train, num_validation] = traj_nums
        all_train = []
        all_validation = []
        
        # Need to keep track of the actual number of lines read 
        num_lines = 0
        
        for line in in_file:
            num_lines += 1
            trajectory = ast.literal_eval(line.split('","')[-1].replace('"',''))
            # Only process the trajectory further if it's not too long or too 
            # short 
            if (len(trajectory) <= max_trajectory_length and 
                len(trajectory) >= min_trajectory_length):
                
                # Convert raw timestamp (seconds from epoch) to datetime 
                # and then convert to seconds-in-day
                start_dtime = datetime.fromtimestamp(int(line.split('","')[5]))
                start_second = (start_dtime.hour * 3600 + 
                                start_dtime.minute * 60 + start_dtime.second)
                                
                # Process the trajectory by checking coordinates and adding 
                # timestamp 
                new_traj = self.__check_point_and_add_timestamp(trajectory, 
                                                                start_second)
                # The new trajectory may be shorter because points outside of 
                # the area are removed. If it is now shorter, we ignore it 
                if (len(new_traj) >= min_trajectory_length):
                    # Add to either the training, or validation list 
                    if num_lines <= num_train:
                        all_train.append(new_traj)
                        print("READING TRAINING DATA %d" % (num_lines))
                    elif num_lines < num_validation + num_train:
                        all_validation.append(new_traj)
                        print("READING VALIDATION DATA %d" % (num_lines))
                    else:
                        break 
        return [all_train, all_validation]
        

    def __read_didi(self, in_file, min_trajectory_length, 
                     max_trajectory_length, traj_nums):
        """
        OPTIMIZED VERSION: Reads the didi trajectory file line-by-line.
        Only parses trajectories that match length requirements BEFORE expensive parsing.
        Stops as soon as we have enough valid trajectories.
        
        Args:
            in_file: (file) The input didi trajectory file 
            min_trajectory_length: (Integer) The shortest allowable trajectory 
                                   length 
            max_trajectory_length: (Integer) The longest allowable trajectory 
                                   length 
            traj_nums: (list of integers) A list containing the number of 
                        trajectories for each training and validation data
        Returns:    
            A list of trajectories. Each trajectory is a list consisting of 
            latitude, longitude and timestamp in the form of minutes-in-day
        """
        # Throws away the .csv header and then read line-by-line 
        in_file.readline()
        
        # Get the lines into the training and validation lists
        [num_train, num_validation] = traj_nums
        all_train = []
        all_validation = []
        
        # Track how many VALID trajectories we've collected (not just lines read)
        num_valid_train = 0
        num_valid_val = 0
        num_lines_read = 0
        num_skipped = 0
        
        print(f"Target: {num_train} training + {num_validation} validation trajectories")
        print("Reading and validating trajectories...")
        
        for line in in_file:
            num_lines_read += 1
            
            # Check if we have enough valid trajectories already
            if num_valid_train >= num_train and num_valid_val >= num_validation:
                print(f"\n✓ Collected enough valid trajectories!")
                print(f"  Lines read: {num_lines_read}")
                print(f"  Valid train: {num_valid_train}/{num_train}")
                print(f"  Valid val: {num_valid_val}/{num_validation}")
                print(f"  Skipped: {num_skipped}")
                break
            
            # OPTIMIZATION 1: Quick length check BEFORE parsing
            # Count approximate number of points by counting commas in trajectory string
            traj_str = line.split('","')[-1].replace('"','')
            
            # Very rough estimate: each point has ~3 commas (in [lng, lat, time])
            # This is just a heuristic to skip obviously invalid trajectories
            approx_num_points = traj_str.count(',') // 3
            
            # Skip if clearly too short or too long (quick rejection)
            if approx_num_points < min_trajectory_length * 0.8:  # Allow some margin
                num_skipped += 1
                if num_lines_read % 100 == 0:
                    print(f"Read {num_lines_read} lines | Train: {num_valid_train} | Val: {num_valid_val} | Skipped: {num_skipped}")
                continue
            
            if approx_num_points > max_trajectory_length * 1.2:  # Allow some margin
                num_skipped += 1
                if num_lines_read % 100 == 0:
                    print(f"Read {num_lines_read} lines | Train: {num_valid_train} | Val: {num_valid_val} | Skipped: {num_skipped}")
                continue
            
            # OPTIMIZATION 2: Only parse if length looks reasonable
            try:
                trajectory = ast.literal_eval(traj_str)
            except:
                # Skip malformed trajectories
                num_skipped += 1
                continue
            
            # Check exact length
            if len(trajectory) < min_trajectory_length or len(trajectory) > max_trajectory_length:
                num_skipped += 1
                if num_lines_read % 100 == 0:
                    print(f"Read {num_lines_read} lines | Train: {num_valid_train} | Val: {num_valid_val} | Skipped: {num_skipped}")
                continue
            
            # OPTIMIZATION 3: Check coordinates and keep valid points
            new_traj = self.__check_point(trajectory)
            
            # Reject if too many points removed (trajectory too short after filtering)
            if len(new_traj) < min_trajectory_length:
                num_skipped += 1
                if num_lines_read % 100 == 0:
                    print(f"Read {num_lines_read} lines | Train: {num_valid_train} | Val: {num_valid_val} | Skipped: {num_skipped}")
                continue
            
            # VALID TRAJECTORY - Add to appropriate list
            if num_valid_train < num_train:
                all_train.append(new_traj)
                num_valid_train += 1
                print(f"✓ TRAINING DATA {num_valid_train}/{num_train} (line {num_lines_read})")
            elif num_valid_val < num_validation:
                all_validation.append(new_traj)
                num_valid_val += 1
                print(f"✓ VALIDATION DATA {num_valid_val}/{num_validation} (line {num_lines_read})")
        
        # Final summary
        print(f"\n" + "="*70)
        print(f"Reading complete!")
        print(f"  Total lines read: {num_lines_read}")
        print(f"  Valid training trajectories: {num_valid_train}")
        print(f"  Valid validation trajectories: {num_valid_val}")
        print(f"  Skipped trajectories: {num_skipped}")
        print("="*70 + "\n")
        
        return [all_train, all_validation]
        

    def __check_point(self, trajectory):
        """
        Given a trajectory consisting of latitude, longitude and timestamp, 
        check if each point is inside the valid area. If it is not, remove it.
        
        Args:
            trajectory: (list) List of list of longitude, latitude and timestamp
                          
        Returns:
            A list of list of latitude, longitude and timestamp in the form 
            of minutes-in-day
        """
        new_trajectory = []
        for point in trajectory: 
            shapely_point = Point(point[0], point[1])
            if self.bbox.contains(shapely_point):
                new_trajectory.append([point[0], point[1], point[2]])
        return new_trajectory
            
        
        
    def __check_point_and_add_timestamp(self, trajectory, start_second):
        """
        Given a trajectory consisting of latitude and longitude points, check if 
        each point is inside the valid area. If it is not, remove it, if it is,
        add the minutes-in-day timestamp. We also flip the ordering between 
        lat and lng, because the raw Porto data has the longitude first. 
        
        Args:
            trajectory: (list) List of list of longitude and latitude points 
            start_second: (integer) The second-in-the-day where the trajectory 
                          starts
                          
        Returns:
            A list of list of latitude, longitude and timestamp in the form 
            of minutes-in-day
        """
        # We add the minutes in day information, but for the Porto dataset, 
        # each trajectory point is 15 seconds apart, so we need both the 
        # second and minute information 
        cur_second = start_second
        new_trajectory = []
        for point in trajectory:
            # After the 15 seconds addition, cur_second may pass the max. 
            # number of seconds in a day. We fix this. 
            if cur_second >= self.__SECONDS_IN_DAY:
                cur_second -= self.__SECONDS_IN_DAY
                    
            # Check if the point is inside the bbox. If it is, add time info and
            # append the point to new_trajectory 
            shapely_point = Point(point[1], point[0])
            if self.bbox.contains(shapely_point):
                cur_minute = int(cur_second / self.__SECONDS_IN_MINUTE)
                new_trajectory.append([point[1], point[0], cur_minute])
                
            # Add 15 seconds for the next trajectory point 
            cur_second += self.__PORTO_SECOND_INCREMENT
        return new_trajectory
