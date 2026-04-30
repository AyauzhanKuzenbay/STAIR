"""
This class handles the trajectory reading from file specifically for the test 
data. 
"""

import ast
import copy
import math
import csv
import numpy as np
import random
from datetime import datetime
from operator import itemgetter
from shapely.geometry import Point, Polygon


class TestFileProcessor():
    def __init__(self, input_file_path, line_start, bbox_coords, all_grids,
                 key_lookup_dict):
        """
        Initializes the file reader. line_start determine the number of line
        to start reading, because all the previous lines have been read in the 
        training+validation phase and therefore will be skipped for the test 
        data 

        Args: 
            input_file_path: (string) The path to the input .csv file 
            line_start: (int) The number of line to start reading the file from 
            bbox_coords: (list) List of four floats that determine the bounding 
                          box of the entire area 
            all_grids: (numpy array) Array of all grids and their corresponding
                        information such as its centroid and its coordinates. 
        """
        self.filepath = input_file_path
        self.file = open(input_file_path, 'r')

        for self.num_line, line in enumerate(self.file):
            if self.num_line == line_start:
                break
        print("初始行数")
        print(self.num_line)
        [self.min_lat, self.min_lng, self.max_lat, self.max_lng] = bbox_coords
        self.bbox = Polygon([(self.min_lng, self.min_lat),
                             (self.max_lng, self.min_lat),
                             (self.max_lng, self.max_lat),
                             (self.min_lng, self.max_lat),
                             (self.min_lng, self.min_lat)])
        self.all_grids = all_grids
        self.key_lookup_dict = key_lookup_dict
        self.__MINUTES_IN_DAY = 1440
        self.__R_EARTH = 6378137
        self.__SECONDS_IN_DAY = 86400
        self.__PORTO_SECOND_INCREMENT = 15
        self.__SECONDS_IN_MINUTE = 60

    def process_data(self, num_data, dataset_mode, min_traj_len, max_traj_len,
                     drop_rate, spatial_distortion, temporal_distortion,
                     start_ID):
        """
        Calling function that assigns the task of processing data to the 
        correct function depending on the data mode 

        Args:
            num_data: (integer) The number of data points to be returned 
            dataset_mode: (string) The dataset mode. Determines how to read 
                            the .csv file. 
            min_traj_len: (integer) The minimum trajectory length. Any 
                           trajectory shorter than this will be removed.
            max_traj_len: (integer) The maximum trajectory length. Any 
                           trajectory longer than this will be removed. 
            drop_rate: (float) The drop rate for the trajectories 
            spatial_distortion: (float) The spatial distortion rate
            temporal_distortion: (integer) The maximum temporal distortion 
            start_ID: (integer) Where the trajectory ID should start from 

        Returns:
            A numpy array containing the gridded and raw trajectories, 
            each assigned a trajectory ID 
        """
        # Two lists for both halves of each trajectory 
        list_traj_1 = []
        list_traj_2 = []
        list_traj_1_raw = []
        list_traj_2_raw = []
        originals_traj = []
        in_file = open(self.filepath, 'r')
        in_file.readline()
        num_valid = 0
        consecutive_empty_lines = 0  # Track consecutive empty lines
        max_consecutive_empty = 100  # Stop if we see 100 empty lines in a row
        
        # Continue reading the file--processing and adding trajectories to 
        # list_traj_1 and 2--until both lists are of sufficient length. 
        while len(list_traj_1) < num_data:
            # Reads line to get the trajectory
            print("数组长度")
            print(len(list_traj_1))
            print("最小轨迹长度")
            print(min_traj_len)
            line = self.file.readline()
            
            # Check for end of file
            if not line:
                print(f"\nReached end of file. Only collected {len(list_traj_1)} trajectories out of {num_data} requested.")
                print("Returning available data...")
                break
            
            self.num_line += 1
            if dataset_mode.lower() == 'porto':
                # Due to the trajectories being split to two later, we need to 
                # double min_traj_len here 
                new_traj = self.__process_csv_porto(line, min_traj_len * 2,
                                                    max_traj_len)
            elif dataset_mode.lower() == 'didi':
                # Due to the trajectories being split to two later, we need to 
                # double min_traj_len here 
                new_traj = self.__process_csv_didi(line, min_traj_len * 2,
                                                   max_traj_len)
            elif dataset_mode.lower() == 'ais':
                new_traj = self.__process_csv_ais(line, min_traj_len * 2, max_traj_len)

            else:
                assert False, "NOT IMPLEMENTED"

            if new_traj is not None:
                # Reset consecutive empty line counter when we find valid data
                consecutive_empty_lines = 0

                """
                # Applying the drop rate, spatial distortion, and temporal 
                # distortion. 
                # This is the old method that downsamples first before 
                # splitting. Also, the old downsampling method is used. 
                if drop_rate > 0:   
                    new_traj = self.__downsample_trajectory(new_traj, 
                                                            [drop_rate])[0]
                if spatial_distortion > 0 or temporal_distortion > 0:
                    new_traj = self.__distort_spatiotemporal_traj(new_traj,
                                                           spatial_distortion,
                                                           temporal_distortion)

                # Grid the trajectory, and then alternatively taking points from
                # the trajectory to form two sub-trajectories 
                new_traj = self.__grid_trajectory(copy.deepcopy(new_traj))
                new_traj = self.__remove_non_hot_cells(new_traj)
                traj_1 = new_traj[0::2]
                traj_2 = new_traj[1::2]
                """
                # Applying the drop rate, spatial distortion, and temporal 
                # distortion. new method based on the ICDE 2018 work.
                print("符合标准")
                traj_1 = new_traj[0::2]
                traj_2 = new_traj[1::2]
                ori_1 = traj_1
                ori_2 = traj_2
                if drop_rate > 0:
                    traj_1 = self.__downsample_trajectory_random(traj_1,
                                                                 [drop_rate])[0]
                    traj_2 = self.__downsample_trajectory_random(traj_2,
                                                                 [drop_rate])[0]
                if spatial_distortion > 0 or temporal_distortion > 0:
                    traj_1 = self.__distort_spatiotemporal_traj(traj_1,
                                                                spatial_distortion,
                                                                temporal_distortion)
                    traj_2 = self.__distort_spatiotemporal_traj(traj_2,
                                                                spatial_distortion,
                                                                temporal_distortion)

                # Grid the trajectory, and then alternatively taking points from
                # the trajectory to form two sub-trajectories 
                traj_1 = self.__grid_trajectory(copy.deepcopy(traj_1))
                traj_1 = self.__remove_non_hot_cells(traj_1)
                traj_2 = self.__grid_trajectory(copy.deepcopy(traj_2))
                traj_2 = self.__remove_non_hot_cells(traj_2)

                # BUG FIX: when a drop_rate is specified, too many trajectories 
                # will be pruned such that the .csv file will not provide 
                # enough trajectories. Due to this reason, we need to find 
                # the original trajectory length and compare that with the 
                # minimum trajectory length requirement. 
                if drop_rate > 0:
                    t1_len = int(len(traj_1) / (1 - drop_rate))
                    t2_len = int(len(traj_2) / (1 - drop_rate))
                else:
                    t1_len = len(traj_1)
                    t2_len = len(traj_2)

                # Only add the trajectory if it is sufficiently long 
                # No need to check max_len, that's been done in 
                # __process_csv_porto and no processing step is going to add 
                # points. 
                if t1_len >= min_traj_len and t2_len >= min_traj_len:
                    [traj_1, traj_forier_1, raw_traj_1] = self.__split_id_and_traj(traj_1)
                    [traj_2, traj_forier_2, raw_traj_2] = self.__split_id_and_traj(traj_2)
                    print("traj_forier", traj_forier_1)
                    list_traj_1.append(np.array([start_ID, traj_1, traj_forier_1], dtype=object))
                    list_traj_1_raw.append(np.array([start_ID, raw_traj_1], dtype=object))
                    list_traj_2.append(np.array([start_ID, traj_2, traj_forier_2], dtype=object))
                    list_traj_2_raw.append(np.array([start_ID, raw_traj_2], dtype=object))
                    start_ID += 1
                    if num_data == 200:
                        originals_traj.append([raw_traj_1, raw_traj_2, new_traj, [start_ID, traj_1]])
            else:
                # Increment counter for consecutive empty/invalid lines
                consecutive_empty_lines += 1
                if consecutive_empty_lines >= max_consecutive_empty:
                    print(f"\nEncountered {max_consecutive_empty} consecutive empty/invalid lines.")
                    print(f"Likely reached end of valid data. Only collected {len(list_traj_1)} trajectories out of {num_data} requested.")
                    print("Returning available data...")
                    break
                    
            print("Line no. %d. Trajectory read: %d out of %d" % \
                  (self.num_line, len(list_traj_1), num_data))
        if num_data == 200:
            ids = 1
            csv_file_path = 'original_trajs.csv'

            # 打开文件进行写操作
            with open(csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)

                # 逐行写入二维数组
                for row in originals_traj:
                    writer.writerow([ids, row])
                    ids += 1

            print(f"数据已成功保存到 {csv_file_path}")

        return [np.array(list_traj_1), np.array(list_traj_2),
                np.array(list_traj_1_raw), np.array(list_traj_2_raw)]

    # def __process_csv_ais(self, line, min_trajectory_length,
    #                         max_trajectory_length):
    #     reads = ast.literal_eval(line.strip(' ').strip('()').split('", "')[-1])
    #
    #     trajectory = eval(reads[1])
    #     if (len(trajectory) <= max_trajectory_length and
    #             len(trajectory) >= min_trajectory_length):
    #
    #         # Convert raw timestamp (seconds from epoch) to datetime
    #         # and then convert to seconds-in-day
    #         new_traj = self.__check_point(trajectory)
    #         # start_dtime = datetime.fromtimestamp(int(reads[1]))
    #
    #         # start_second = (start_dtime.hour * 3600 +
    #         #                 start_dtime.minute * 60 + start_dtime.second)
    #         # print("时间：")
    #         # print(start_second)
    #         # print("处理后的轨迹",trajectory)
    #         # new_traj = self.__check_point_and_add_timestamp_porto(trajectory,
    #         #                                                 0)
    #
    #         if len(new_traj) >= min_trajectory_length:
    #             return new_traj
    #             # If the code reaches this point, then new_traj is not of the right
    #     # length. We return None
    #     return None
    def __process_csv_ais(self, line, min_trajectory_length,
                          max_trajectory_length):
        reads = ast.literal_eval(line.strip(' ').strip('()').split('", "')[-1])

        trajectory = eval(reads[2])
        if (len(trajectory) <= max_trajectory_length and
                len(trajectory) >= min_trajectory_length):

            # Convert raw timestamp (seconds from epoch) to datetime
            # and then convert to seconds-in-day
            start_dtime = datetime.fromtimestamp(int(reads[1]))
            start_second = (start_dtime.hour * 3600 +
                            start_dtime.minute * 60 + start_dtime.second)

            # Process the trajectory by checking coordinates and adding
            # timestamp
            new_traj = self.__check_point_and_add_timestamp_porto(trajectory,
                                                                  start_second)
            if len(new_traj) >= min_trajectory_length:
                return new_traj
                # If the code reaches this point, then new_traj is not of the right
            # length. We return None
        return None

    def __process_csv_porto(self, line, min_trajectory_length,
                            max_trajectory_length):
        """
        Reads the porto trajectory file line-by-line. Also keep track of the 
        actual number of lines read 

        Args:
            line: (string) The line from the .csv file to process
            min_trajectory_length: (Integer) The shortest allowable trajectory
                                   length
            max_trajectory_length: (Integer) The longest allowable trajectory
                                   length
        Returns:
            Trajectory read from the provided line
        """
        trajectory = ast.literal_eval(line.split('","')[-1].replace('"', ''))
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
            new_traj = self.__check_point_and_add_timestamp_porto(trajectory,
                                                                  start_second)
            if len(new_traj) >= min_trajectory_length:
                return new_traj
                # If the code reaches this point, then new_traj is not of the right
        # length. We return None 
        return None

    def __process_csv_didi(self, line, min_trajectory_length,
                           max_trajectory_length):
        """
        Reads the didi trajectory file line-by-line. Also keep track of the 
        actual number of lines read 

        Args:
            line: (string) The line from the .csv file to process 
            min_trajectory_length: (Integer) The shortest allowable trajectory 
                                   length 
            max_trajectory_length: (Integer) The longest allowable trajectory 
                                   length 
        Returns:    
            Trajectory read from the provided line 
        """
        # Add error handling for empty or malformed lines
        try:
            # Check if line is empty or doesn't contain the expected format
            if not line or line.strip() == '':
                return None
            
            # Split and extract the trajectory part
            parts = line.split('","')
            if len(parts) == 0:
                return None
            
            # Get the last part and remove quotes
            trajectory_str = parts[-1].replace('"', '').strip()
            
            # Check if the trajectory string is empty after cleaning
            if not trajectory_str:
                return None
            
            # Parse the trajectory
            trajectory = ast.literal_eval(trajectory_str)
            
        except (SyntaxError, ValueError) as e:
            return None
        except Exception as e:
            return None
        
        # Only process the trajectory further if it's not too long or too 
        # short 
        if (len(trajectory) <= max_trajectory_length and
                len(trajectory) >= min_trajectory_length):
            # Process the trajectory by checking coordinates and adding 
            # timestamp 
            new_traj = self.__check_point(trajectory)
            if len(new_traj) >= min_trajectory_length:
                return new_traj
                # If the code reaches this point, then new_traj is not of the right
        # length. We return None 
        return None

    def __check_point_and_add_timestamp_porto(self, trajectory, start_second):
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

    def __check_point(self, trajectory):
        """
        Given a trajectory consisting of latitude and longitude points, check if 
        each point is inside the valid area. If it is not, remove it.

        Args:
            trajectory: (list) List of list of longitude, latitude and timestamp 
                         triplets 

        Returns:
            A list of list of latitude, longitude and timestamp in the form 
            of minutes-in-day
        """
        new_trajectory = []
        for point in trajectory:
            # DiDi format is [lat, lng, timestamp], so use point[0]=lat, point[1]=lng
            # Don't swap like Porto dataset!
            shapely_point = Point(point[0], point[1])
            if self.bbox.contains(shapely_point):
                new_trajectory.append([point[0], point[1], point[2]])
        return new_trajectory

    def __grid_trajectory(self, trajectory):
        """
        Assign each point in the trajectory to the spatiotemporal grid. 

        Args:
            trajectory: (list of lists) The trajectory to be gridded 

        Returns:
            The processed trajectory form in the format of:
            [cell_id, raw_data_array, gridded_data_array]
        """
        traj_data = []
        for traj_point in trajectory:
            [grid_data, cell_id] = self.__grid_traj_point(traj_point)
            traj_point_ = copy.deepcopy(traj_point)
            new_traj_point = [cell_id, traj_point_, grid_data]
            # print("new traj_point",new_traj_point)
            traj_data.append(new_traj_point)
        return traj_data

    def __grid_traj_point(self, traj_point):
        """
        Search all_cells to find the grid cell that contains traj_point. 

        Inputs:
            traj_point: (list of floats) The trajectory point consisting of 
                         lat, lng and timestamp 

        Returns:
            The trajectory point with the lat, lng and timestamp features 
            replaced with the lat, lng and timestamp IDs 
        """
        [lng, lat, time] = traj_point

        # We use binary search for this task 
        lat_floor = 0
        lat_ceil = int(self.all_grids.shape[0] - 1)
        lat_cur = int(lat_ceil / 2)
        lng_floor = 0
        lng_ceil = int(self.all_grids.shape[1] - 1)
        lng_cur = int(lng_ceil / 2)
        time_floor = 0
        time_ceil = int(self.all_grids.shape[2] - 1)
        time_cur = int(time_ceil / 2)

        # Do the binary search until we find the correct cell.
        # We start with the latitude. The logic for lat, lng and timestamp are 
        # all the same 
        while True:
            lat_range = self.all_grids[lat_cur][0][0]['lat_range']
            if lat < lat_range[0]:
                lat_ceil = lat_cur
                lat_new = math.floor((lat_floor + lat_ceil) / 2)
            elif lat > lat_range[1]:
                lat_floor = lat_cur
                lat_new = math.ceil((lat_floor + lat_ceil) / 2)
            else:
                break
            if lat_new == lat_cur:
                assert False, ("Latitude doesn't change in binary search. " +
                               "Possible infinite loop.")
            lat_cur = lat_new
        while True:
            lng_range = self.all_grids[0][lng_cur][0]['lng_range']
            if lng < lng_range[0]:
                lng_ceil = lng_cur
                lng_new = math.floor((lng_floor + lng_ceil) / 2)
            elif lng > lng_range[1]:
                lng_floor = lng_cur
                lng_new = math.ceil((lng_floor + lng_ceil) / 2)
            else:
                break
            if lng_new == lng_cur:
                assert False, ("Longitude doesn't change in binary search. " +
                               "Possible infinite loop.")
            lng_cur = lng_new
        while True:
            time_range = self.all_grids[0][0][time_cur]['timestamp_range']
            if time < time_range[0]:
                time_ceil = time_cur
                time_new = math.floor((time_floor + time_ceil) / 2)
            elif time > time_range[1]:
                time_floor = time_cur
                time_new = math.ceil((time_floor + time_ceil) / 2)
            else:
                break
            if time_new == time_cur:
                assert False, ("Timestamp doesn't change in binary search. " +
                               "Possible infinite loop.")
            time_cur = time_new

        # If the code reaches this point, we found the cell. 
        # Return the lat, lng, timestamp indices alongside the ID of the cell 
        cell_ID = self.all_grids[lat_cur][lng_cur][time_cur]['cell_id']
        return [[lat_cur, lng_cur, time_cur], cell_ID]

    def __downsample_trajectory_random(self, trajectory, point_drop_rates):
        """
        Downsamples a trajectory once for each point_drop_rate and return the 
        result. The downsampling must keep the first and last point in the 
        trajectory 

        This method performs a more random downsampling by assining each 
        point (except the first and last) a percentage chance to be removed, 
        as opposed to __downsample trajectory that uses the drop_rate to 
        find out the exact number of points to be removed. 

        Args:
            trajectory (list of lists): The trajectory to be downsampled 
            point_drop_rates (list of floats): The drop rates

        Returns:
            A list of trajectories where each item in the list represents one 
            downsampling of the input trajectories 
        """
        downsampled_trajs = []
        print("drop率")
        print(point_drop_rates)
        for point_drop_rate in point_drop_rates:
            traj_mid = [x for x in trajectory[1:-1] \
                        if random.random() > point_drop_rate]
            downsampled_trajs.append([trajectory[0]] + traj_mid +
                                     [trajectory[-1]])
        return downsampled_trajs

    def __downsample_trajectory(self, trajectory, point_drop_rates):
        """
        Downsamples a trajectory once for each point_drop_rate and return the 
        result. The downsampling must keep the first and last point in the 
        trajectory 

        Args:
            trajectory (list of lists): The trajectory to be downsampled 
            point_drop_rates (list of floats): The drop rates

        Returns:
            A list of trajectories where each item in the list represents one 
            downsampling of the input trajectories 
        """
        # For each point_drop_rates, determine the indices of the points to 
        # keep. Since the first and last points are to be kept, we minus 2 
        # from the intended number. 
        nums_point = [round((1 - dr) * len(trajectory)) - 2 \
                      for dr in point_drop_rates]

        # Do the downsampling by randomly picking points that we want to keep 
        downsampled_trajs = []
        for num_point in nums_point:
            # There are two unique cases to handle. 
            if num_point + 2 >= len(trajectory):
                # One unique case is when we want to keep all points, (i.e. 
                # when one num_point + 2 equals len of traj) in which we keep 
                # all points. 
                downsampled_trajs.append(copy.deepcopy(trajectory))
            elif num_point <= 0:
                # The other unique case is when num_point is 0 or negative, in 
                # which we only keep the first and last points.  
                trajectory_ = copy.deepcopy(trajectory)
                downsampled_trajs.append([trajectory_[i] for i in [0, -1]])
            else:
                # For the normal case, we randomly pick elements in order.
                trajectory_ = copy.deepcopy(trajectory)
                rand_indices = random.sample(range(1, len(trajectory_) - 1),
                                             num_point)
                rand_indices.sort()
                downsampled_traj = [trajectory_[i] for i in rand_indices]

                # Don't forget to add the first and last points 
                downsampled_traj.insert(0, trajectory_[0])
                downsampled_traj.append(trajectory_[-1])
                downsampled_trajs.append(downsampled_traj)
        return downsampled_trajs

    def __distort_spatiotemporal_traj(self, traj, s_dist_rate, t_dist):
        """
        Distorts a trajectory both spatially and temporally 

        Args:
            traj: (list) List of trajectory points, containing both the 
                   spatial and temporal information 
            s_dist_rate: (float) The rate of the spatial distortion 
            t_dist: (integer) The maximum temporal distortion  (in minutes) 
        """
        traj_ = copy.deepcopy(traj)
        # If s_dist_rate is 0, skip the spatial distortion 
        if s_dist_rate != 0:
            for traj_point in traj_:
                if random.random() < s_dist_rate:
                    self.__distort_spatial_fix(traj_point)
        if t_dist != 0:
            self.__distort_temporal_traj(traj_, t_dist)
        return traj_

    """
    def __distort_spatiotemporal_traj(self, traj, s_dist_rates, t_dist_rates, 
                                      bbox):
        Distorts a trajectory both spatially and temporally 

        Args:
            traj: (list) List of trajectory points, containing both the 
                   spatial and temporal information 
            s_dist_rates: (integer) The maximum spatial distortion 
                                    (in meters)
            t_dist: (integer) The maximum temporal distortion 
                                     (in minutes) 
            bbox: (shapely Polygon) A polygon that represents the valid area
        traj_ = copy.deepcopy(traj) 
        # Do not do the distortion if the spatial distortion is 0, same for 
        # temporal 
        if max_spatial_distortion != 0:
            for traj_point in traj_:
                self.__distort_spatial(traj_point, max_spatial_distortion, bbox)
        if max_temporal_distortion != 0:
            self.__distort_temporal_traj(traj_, max_temporal_distortion)
        return traj_ 


    def __distort_spatial(self, traj_point, max_spatial_distortion, bbox):
        Performs the spatial distortion. This is done by randomizing a bearing 
        and shifts the point a random distance in that bearing. The random 
        distance is limited by max_spatial_distortion. If after the distortion 
        the point is outside the valid area, revert it to the original position.
        Calculation is from: stackoverflow question #7222382

        Args:
            traj_point: (list)The triplet of latitude, longitude, timestamp that 
                        we want to distort spatially 
            max_spatial_distortion: (integer) The maximum distance that the 
                                    latitude and longitude points can be 
                                    distorted to. Unit is in meters. 
            bbox: (shapely Polygon) A polygon that represents the valid area
        # Convert lat and lng to radians 
        [old_lat, old_lng, _] = traj_point 
        lat1 = math.radians(old_lat)
        lng1 = math.radians(old_lng)

        # Calculate the new point 
        degree = random.randint(1,360)
        bearing = math.radians(degree)
        distort_dist = random.randint(0, max_spatial_distortion)
        d_r =  distort_dist / self.__R_EARTH
        lat2 = math.asin(math.sin(lat1) * math.cos(d_r) + 
                         math.cos(lat1) * math.sin(d_r) * math.cos(bearing))
        lng2 = lng1+math.atan2(math.sin(bearing)*math.sin(d_r)*math.cos(lat1), 
                               math.cos(d_r) - math.sin(lat1) * math.sin(lat2))
        lat2 = math.degrees(lat2)
        lng2 = math.degrees(lng2) 

        # There is a chance that the distorted points will be outside of the 
        # area. In this case, the easiest way to handle this is to use the 
        # original latitude and longitude 
        new_point = Point(lat2, lng2)
        if bbox.contains(new_point):
            traj_point[0] = lat2
            traj_point[1] = lng2 
    """

    def __distort_spatial_fix(self, traj_point):
        """
        Performs the spatial distortion. This is done by randomizing a bearing 
        and shifts the point a random distance in that bearing. The random 
        distance is limited by max_spatial_distortion. If after the distortion 
        the point is outside the valid area, revert it to the original position.
        Calculation is from: stackoverflow question #7222382

        This version doesn't take the maximum spatial distortion as the 
        argument; it is set at a fixed 30 meters. 

        Args:
            traj_point: (list)The triplet of latitude, longitude, timestamp that 
                        we want to distort spatially 
            bbox: (shapely Polygon) A polygon that represents the valid area
        """
        # Convert lat and lng to radians 
        [old_lat, old_lng, _] = traj_point
        lat1 = math.radians(old_lat)
        lng1 = math.radians(old_lng)

        # Calculate the new point 
        degree = random.randint(1, 360)
        bearing = math.radians(degree)
        distort_dist = random.randint(0, 30)
        d_r = distort_dist / self.__R_EARTH
        lat2 = math.asin(math.sin(lat1) * math.cos(d_r) +
                         math.cos(lat1) * math.sin(d_r) * math.cos(bearing))
        lng2 = lng1 + math.atan2(math.sin(bearing) * math.sin(d_r) * math.cos(lat1),
                                 math.cos(d_r) - math.sin(lat1) * math.sin(lat2))
        lat2 = math.degrees(lat2)
        lng2 = math.degrees(lng2)

        # There is a chance that the distorted points will be outside of the 
        # area. In this case, the easiest way to handle this is to use the 
        # original latitude and longitude 
        new_point = Point(lat2, lng2)
        if self.bbox.contains(new_point):
            traj_point[0] = lat2
            traj_point[1] = lng2

    def __distort_temporal_traj(self, trajectory, max_temporal_distortion):
        """
        Performs temporal distortion by adding or subtracting minutes from
        every point in the trajectory. 

        Inputs:
            trajectory: (list of lists) The trajectory to be distorted 
            max_temporal_distortion: (integer) The limit for the temporal 
                                      distortion. Since the distortion can be 
                                      positive or negative, this acts as both 
                                      upper and lower limit. 
        """
        # Get the amount of distortions 
        t_distort = random.randint(-max_temporal_distortion,
                                   max_temporal_distortion)

        # Distort every point in the trajectory using the same distortion 
        for i in range(len(trajectory)):
            traj_point = trajectory[i]
            traj_point[2] += t_distort

            # Distortions can cause the trajectory to go over the max. number 
            # of minutes in a day, or go to the negatives. We address both.
            if traj_point[2] >= self.__MINUTES_IN_DAY:
                traj_point[2] -= self.__MINUTES_IN_DAY
            if traj_point[2] < 0:
                traj_point[2] += self.__MINUTES_IN_DAY

    def __split_id_and_traj(self, trajectory):
        """
        Given a trajectory that contains the cell ID, raw trajectory data, and 
        gridded trajectory data, split the cell ID and raw trajectory data and 
        return them. For the cell ID, transform the trajectory, and all its 
        points, to a numpy array. For the raw trajectories, keep them in a list

        Args:
            trajectory: (list) List of trajectory points, where each trajectory 
                         point contains the grid ID, raw spatiotemporal data, 
                         and gridded spatiotemporal data.

        Returns:
            A trajectory with only the grid ID kept. The points and the 
            trajectory itself are in a numpy array. 
        """
        trajectory_ = copy.deepcopy(trajectory)
        traj_id = np.array([np.array([x[0]]) for x in trajectory_])
        traj_forier = np.array([np.array([x[2]]) for x in trajectory_])
        raw_traj = np.array([x[1] for x in trajectory_])
        return [traj_id, traj_forier, raw_traj]

    def __remove_non_hot_cells(self, trajectory):
        """
        Given a trajectory, remove any trajectory point not belonging to a 
        hot cells. For the remaining trajectory points, change the ID by 
        using the key_lookup_dict

        Args:
            trajectory: (list) List of trajectory points 

        Returns:
            The trajectory with the ID replaced. 
        """
        new_trajectory = []
        for point in trajectory:
            if point[0] in self.key_lookup_dict:
                point_ = copy.deepcopy(point)
                point_[0] = self.key_lookup_dict[point_[0]]
                new_trajectory.append(point_)
        return new_trajectory

    def close_file(self):
        """
        Simply close the file
        """
        self.file.close()
