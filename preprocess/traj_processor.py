"""
Given a list of trajectories, process them to get a dataset ready to be used 
for the deep neural network model training
"""

from shapely.geometry import Point
from shapely.geometry import Polygon 
import copy
import decimal
import math
import numpy as np 
import random   

class TrajProcessor():
    """
    Takes a list of trajectories and processes it to output a model-ready 
    dataset. 
    """
    def __init__(self):
        """
        Initializes several important constants 
        """
        self.__MINUTES_IN_DAY = 1440
        self.__R_EARTH = 6378137
        

    def first_loop(self, all_traj, point_drop_rates, spatial_distortions, 
                   temporal_distortions, all_cells, bbox_coords, span, stride):
        """
        The first loop through the whole dataset performs several tasks:
        1. Generate query trajectories from ground truth by downsampling it 
        2. Distort the query trajectories spatially and temporally 
        3. For both the query and ground truth trajectories, assign the 
           latitude, longitude and timestamp to spatiotemporal cells 
        4. Get the spatial and temporal pattern features 
        
        Args:
            all_traj: (list of lists) List of all trajectories 
            point_drop_rates: (list of floats) List of all drop rates 
            spatial_distortions: (list of floats) List of all spatial 
                                  distortion rates. 
            temporal_distortions: (list of integers) List of all temporal 
                                   distortion amounts. Unit is in minutes
            all_cells: (3D numpy array) All grid cells 
            bbox_coords: (list of double) List of four coordinates that 
                          determine the entire valid area, e.g. the city the 
                          dataset is based on. 
            span: (integer) The span is used to form the pattern features. This 
                   argument determines the window size for a pattern. 
            stride: (integer) The stride determines the distance of each 
                     pattern, which in turns determines the magnitude of point 
                     overlap between patterns. 
        Returns: 
            A list of ground truth and query trajectory pairs. There is only 
            one ground truth, but there may be several query trajectories for 
            one ground truth depending on the number of drop_rates provided.
            The 
        """
        [min_lat, min_lng, max_lat, max_lng] = bbox_coords
        bbox = Polygon([(min_lat, min_lng), (max_lat, min_lng), 
                        (max_lat, max_lng), (min_lat, max_lng),
                        (min_lat, min_lng)])
        
        # Iterate through all trajectories 
        all_pairs = []
        for i in range(len(all_traj)):
            print("Processing trajectory (1st loop) " + str(i+1) + " out of " +
                   str(len(all_traj)))
                   
            # Generate the downsampled trajectories
            all_cur_traj_q = []
            cur_traj_q = self.__downsample_trajectory_random(all_traj[i],
                                                             point_drop_rates)
            
            # Distort the downsampled trajectories 
            for traj_q in cur_traj_q:
                for s in spatial_distortions:
                    for t in temporal_distortions:
                        traj_q_new = self.__distort_spatiotemporal_traj(traj_q,
                                                                        s, t, 
                                                                        bbox)
                        # Also grid the trajectories 
                        traj_q_new = self.__grid_trajectory(traj_q_new, 
                                                            all_cells)
                        all_cur_traj_q.append(traj_q_new) 
                
            # Grid the ground truth trajectory 
            gt_traj_grid = self.__grid_trajectory(all_traj[i], all_cells)
            
            # Get the pattern features for the ground truth 
            all_ranges = self.__create_pattern_ranges(span, stride)
            gt_patt_features = self.__get_pattern_features(gt_traj_grid, 
                                                           all_ranges)
            gt_data = [gt_traj_grid, gt_patt_features]
            #print(gt_data)
            #print(all_cur_traj_q)
            all_pairs.append([gt_data, all_cur_traj_q])
        return all_pairs 


    def second_loop(self, all_traj_pairs, key_lookup_dict, 
                    min_gt_length , min_q_length):
        """
        The second loop is much simpler; it only performs three tasks for each 
        ground truth and query trajectories:
        
        1. Remove the cells if the do not belong to the hot cells list 
        2. After step 1, if the trajectory is now too short, remove it. Ground 
           truth and query trajectories have a different minimum length, with 
           the query minimum length being smaller. 
        3. Replace the cell ID of each trajectory point with an integer 
           specified in key_lookup_dict

        Args:
            all_traj_pairs: (list) List of all ground truth and query trajectory
                             pairs 
            key_lookup_dict: (dict) A dictionary to map the old cell IDs to the 
                              new, integer-based one. 
            min_gt_length: (integer) The minimum GT trajectory length. Any GT 
                                     trajectory shorter than this will be 
                                     removed. 
            min_q_length: (integer) The minimum Q trajectory length. Any q 
                                    trajectory shorter than this will be removed
                                    
        Returns:
            All the trajectory pairs after being processed.
        """
        i = 0
        new_all_traj_pairs = []
        while len(all_traj_pairs) > 0:
            i += 1
            print("Processing trajectory (2nd loop) " + str(i) + " out of " +
                   str(len(all_traj_pairs)))
            [gt, all_q] = all_traj_pairs.pop()

            new_gt = self.__remove_non_hot_cells(gt[0], key_lookup_dict)
            #print(new_gt)
            # Only process the query trajectories if the ground truth is not 
            # too short. 
            if len(new_gt) >= min_gt_length:
                new_q = []
                for q in all_q:
                    q_ = self.__remove_non_hot_cells(q, key_lookup_dict)
                    if len(q_) >= min_q_length: 
                        new_q.append(q_)
                if len(new_q) > 0:
                    yield [[new_gt, np.array(gt[1])], new_q]
        """
        new_all_traj_pairs = []
        for i in range(len(all_traj_pairs)):
            print("Processing trajectory (2nd loop) " + str(i+1) + " out of " +
                   str(len(all_traj_pairs)))
            [gt, all_q] = all_traj_pairs[i]
            new_gt = self.__remove_non_hot_cells(gt[0], key_lookup_dict)
            
            # Only process the query trajectories if the ground truth is not 
            # too short. 
            if len(new_gt) >= min_gt_length:
                new_q = []
                for q in all_q:
                    q_ = self.__remove_non_hot_cells(q, key_lookup_dict)
                    if len(q_) >= min_q_length: 
                        new_q.append(q_)
                if len(new_q) > 0:
                    new_all_traj_pairs.append([[new_gt, np.array(gt[1])], new_q])
        return new_all_traj_pairs
        """


    def split_and_process_dataset(self, all_traj_pairs, num_data):
        """
        Splits all the trajectories to the training, validation, and test 
        split. The training and validation set are processed differently 
        compared to the test set. This modified version selects data based on 
        the number of query trajectories instead of the ground truth. This means 
        that given a ground truth, not all of its matching queries will be part 
        of the same dataset. 
        
        Args:
            all_traj_pairs: (list) List of all ground truth, query trajectory 
                             pairs 
            num_data: (list) A list of either integers or decimals. If used, 
                       integer specifies the actual number of data for each 
                       split, while the decimal specifies the fraction of data 
                       (from the full dataset) to be used for each split. 
        """
        # A ground truth is associated with one or more queries. This messes 
        # up the calculation of the actual pairs. Here, we "flatten" them 
        # first by creating a pair for every ground truth and query 
        flattened_pairs = []
        id = 0 
        for one_pair in all_traj_pairs:
            [[gt, gt_patt], q] = copy.deepcopy(one_pair)
            for one_q in q:
                flattened_pairs.append([id, [gt, gt_patt, one_q]])
            id += 1
        
        # Get num of data for each split 
        # First is to handle the integer case. If the total number of data 
        # exceeds the total trajectory, default to a [70, 20, 10] split. 
        total_traj = len(flattened_pairs)
        if all(isinstance(x, int) for x in num_data):
            if sum(num_data) > total_traj:
                print("WARNING! Total number of data exceeds total number of " +
                      "trajectory! Defaulting to a [70, 20, 10] split")
                num_data = [decimal.Decimal('0.7'), decimal.Decimal('0.2'),
                            decimal.Decimal('0.1')]
            else:
                [num_train, num_val, num_test] = num_data
        if all(isinstance(x, decimal.Decimal) for x in num_data):    
            num_train = round(num_data[0] * total_traj)
            num_val = round(num_data[1] * total_traj)
            num_test = round(num_data[2] * total_traj)
        
        # Randomize and split the data 
        random.shuffle(flattened_pairs)
        train_pairs = flattened_pairs[:num_train]
        val_pairs = flattened_pairs[num_train : num_train + num_val]
        test_pairs = flattened_pairs[num_train + num_val: \
                                    num_train + num_val + num_test]
        
        # Process the training and validation data 
        train_processed = self.process_training_data(train_pairs)
        val_processed = self.process_training_data(val_pairs) 
        test_processed = self.__process_test_data(test_pairs)
        return [train_processed, val_processed, test_processed]
        
        
    def flatten_traj_pairs(self, all_traj_pairs):
        """
        A ground truth is associated with one or more queries. This messes 
        up the calculation of the actual pairs. Here, we "flatten" them 
        first by creating a pair for every ground truth and query 
        
        Args:
            all_traj_pairs: (list) List of all gt-q pairs 
            
        Returns 
            List of "flattened" trajectory pairs, in which a ground truth is 
            matched with one and only one query. 
        """
        flattened_pairs = []
        id = 0 
        for one_pair in all_traj_pairs:
            #print("Flattening trajectory pairs: %d" % (id))
            [[gt, gt_patt], q] = copy.deepcopy(one_pair)
            for one_q in q:
                yield [id, [gt, gt_patt, one_q]]
            id += 1
        print("trainpairs: ", id)
        
    def process_training_data(self, all_pairs):
        """
        Given an input dataset, process it for the training part. The output 
        consists of x (the input data) and y (the value the model should 
        predict). The contents of each are:
        
        x: 
        1. Ground truth trajectory. ONLY keep the ID. 
        2. Query trajectory. ONLY keep the ID
        3. Target pattern output (spatial)
        4. Target pattern output (temporal) 
        
        y: 
        1. Ground truth trajectory. ONLY keep the ID 
        2. Target pattern output (spatial). 
        3. Target pattern output (temporal). 
        
        Args:
            all_pairs: (list) List of all ground truth-query trajectory pairs.
                        It's a list of pairs where each pair consists of the 
                        ground truth trajectory, ground truth pattern features, 
                        and the query 
            
        Returns:
            A numpy array for the whole dataset that contains the data as 
            specified above. 
        """
        all_x = []
        all_y = []
        num_traj = 0 
        for one_pair in all_pairs:
            num_traj += 1
            #print("Processing train/val data: %d" % (num_traj))
            [_, [gt, gt_patt, q]] = one_pair 
            gt = self.__keep_id_only(gt)

            gt_patt_s = np.array([np.array(x[[0]]) for x in gt_patt])
            gt_patt_t = np.array([np.array(x[[1]]) for x in gt_patt])
            q = self.__keep_id_only(q)
            
            # Form the X and then y and then append  
            one_x = np.array([gt, q, gt_patt_s, gt_patt_t],dtype=object)

            one_y = np.array([gt, gt_patt_s, gt_patt_t],dtype=object)
            all_x.append(one_x)
            all_y.append(one_y)
        cnt = 0
        for x in all_x:
            cnt += 1
        yield [np.array(all_x), np.array(all_y)]
        

    def __remove_non_hot_cells(self, trajectory, key_lookup_dict):
        """
        Given a trajectory, remove any trajectory point not belonging to a 
        hot cells. For the remaining trajectory points, change the ID by 
        using the key_lookup_dict
        
        Args:
            trajectory: (list) List of trajectory points 
            key_lookup_dict: (dict) A dictionary with the old trajectory ID 
                              as key and the new one as the value. 
                              
        Returns:
            The trajectory with the ID replaced. 
        """
        new_trajectory = []
        for point in trajectory:
            if point[0] in key_lookup_dict:
                point_ = copy.deepcopy(point)
                point_[0] = key_lookup_dict[point_[0]]
                new_trajectory.append(point_)
        return new_trajectory
        

    def __keep_id_only(self, trajectory):
        """
        Given a trajectory that contains the cell ID, raw trajectory data, and 
        gridded trajectory data, keep only the cell ID and transform the 
        trajectory, and all its points, to a numpy array. 
        
        Args:
            trajectory: (list) List of trajectory points, where each trajectory 
                         point contains the grid ID, raw spatiotemporal data, 
                         and gridded spatiotemporal data.
                         
        Returns:
            A trajectory with only the grid ID kept. The points and the 
            trajectory itself are in a numpy array. 
        """
        trajectory_ = copy.deepcopy(trajectory)
        traj_array = np.array([np.array([x[0]]) for x in trajectory_])

        return traj_array
        
        
    def __grid_trajectory(self, trajectory, all_cells):
        """
        Assign each point in the trajectory to the spatiotemporal grid. 
        
        Args:
            trajectory: (list of lists) The trajectory to be gridded 
            all_cells: (3D numpy array) All grid cells 
            
        Returns:
            The processed trajectory form in the format of:
            [cell_id, raw_data_array, gridded_data_array]
        """
        traj_data = []
        for traj_point in trajectory:
            [grid_data, cell_id] = self.__grid_traj_point(traj_point, all_cells)
            traj_point_ = copy.deepcopy(traj_point)
            new_traj_point = [cell_id, traj_point_, grid_data]
            traj_data.append(new_traj_point)
        return traj_data 
            
            
    def __grid_traj_point(self, traj_point, all_cells):
        """
        Search all_cells to find the grid cell that contains traj_point. 
        
        Inputs:
            traj_point: (list of floats) The trajectory point consisting of 
                         lat, lng and timestamp
            all_cells: (3D numpy array) Array containing all grid cells, the
                        first dimension is for the latitude, the second is for
                        the longitude, and the third is for the timestamp. 
                      
        Returns:
            The trajectory point with the lat, lng and timestamp features 
            replaced with the lat, lng and timestamp IDs 
        """
        [lng, lat, time] = traj_point
        # We use binary search for this task 
        lat_floor = 0
        lat_ceil = int(all_cells.shape[0]-1)
        lat_cur = int(lat_ceil/2)
        lng_floor = 0
        lng_ceil = int(all_cells.shape[1]-1)
        lng_cur = int(lng_ceil/2)
        time_floor = 0
        time_ceil = int(all_cells.shape[2]-1)
        time_cur = int(time_ceil/2)
        
        # Do the binary search until we find the correct cell.
        # We start with the latitude. The logic for lat, lng and timestamp are 
        # all the same 
        while True:
            lat_range = all_cells[lat_cur][0][0]['lat_range']
            if lat < lat_range[0]:
                lat_ceil = lat_cur 
                lat_new = math.floor((lat_floor + lat_ceil)/2)
            elif lat > lat_range[1]:
                lat_floor = lat_cur 
                lat_new = math.ceil((lat_floor + lat_ceil)/2)
            else:
                break 
            if lat_new == lat_cur:
                assert False, ("Latitude doesn't change in binary search. " +
                               "Possible infinite loop.")
            lat_cur = lat_new
        while True:
            lng_range = all_cells[0][lng_cur][0]['lng_range']
            if lng < lng_range[0]:
                lng_ceil = lng_cur 
                lng_new = math.floor((lng_floor + lng_ceil)/2)
            elif lng > lng_range[1]:
                lng_floor = lng_cur 
                lng_new = math.ceil((lng_floor + lng_ceil)/2)
            else:
                break 
            if lng_new == lng_cur:
                assert False, ("Longitude doesn't change in binary search. " +
                               "Possible infinite loop.")
            lng_cur = lng_new
        while True:
            time_range = all_cells[0][0][time_cur]['timestamp_range']
            if time < time_range[0]:
                time_ceil = time_cur 
                time_new = math.floor((time_floor + time_ceil)/2)
            elif time > time_range[1]:
                time_floor = time_cur 
                time_new = math.ceil((time_floor + time_ceil)/2)
            else:
                break 
            if time_new == time_cur:
                assert False, ("Timestamp doesn't change in binary search. " +
                               "Possible infinite loop.")
            time_cur = time_new
            
        # If the code reaches this point, we found the cell. 
        # Update the cell count in all_cells and return the lat, lng, timestamp 
        # indices alongside the ID of the cell 
        hit_count = all_cells[lat_cur][lng_cur][time_cur]['hit_count']
        cell_ID = all_cells[lat_cur][lng_cur][time_cur]['cell_id']
        all_cells[lat_cur][lng_cur][time_cur]['hit_count'] = hit_count + 1
        
        return [[lat_cur, lng_cur, time_cur], cell_ID]
        
        
    def __get_pattern_features(self, trajectory, all_ranges):
        """
        Given a trajectory containing the raw and gridded points, get the 
        spatial and temporal pattern feature. The spatial feature is the 
        total distance travelled within each pattern and the temporal feature 
        is the total time elapsed
        
        Args:
            trajectory: (list of lists) Input trajectory consisting of the 
                         trajectory points, which has the cell ID, raw 
                         trajectory, and gridded trajectory 
            all_ranges: (list of ranges) List of all temporal ranges, each of 
                         which determines a pattern. 
        """
        all_ranges_ = copy.deepcopy(all_ranges) 
        trajectory_ = copy.deepcopy(trajectory) 
        
        # We get only the relevant ranges within all_ranges_
        start_time_id = trajectory_[0][1][2]
        end_time_id = trajectory_[-1][1][2]
        
        def __binary_search(all_ranges, val):
            i_cur = int(len(all_ranges) / 2)
            i_floor = 0
            i_ceil = len(all_ranges)-1
        
            while True:
                if val < all_ranges[i_cur][0]:
                    i_ceil = i_cur 
                    i_new = math.floor((i_floor + i_ceil)/2)
                elif val > all_ranges[i_cur][-1]:
                    i_floor = i_cur 
                    i_new = math.ceil((i_floor + i_ceil)/2)
                else:
                    return i_cur
                    
                if i_new == i_cur:
                    assert False, ("Index doesn't change in binary search. " +
                                   "Possible infinite loop.")
                i_cur = i_new
        
        # The case of day wraparound (i.e. the timestamp going from the end of 
        # one day to the start of next) is going to cause problems. So, we 
        # deal with it differently 
        # The normal case:
        if start_time_id <= end_time_id:
            # Do a binary search to find the range in all_ranges_ where the 
            # start, and then end time id is contained in 
            start_index = __binary_search(all_ranges_, start_time_id)
            end_index = __binary_search(all_ranges_, end_time_id)
        else:
            # The abnormal case, i.e. day wraparound. In this case, divide 
            # all_ranges_ in two and swaps the two sections such that the 
            # start_index is contained in the first range of this reordered list
            start_index = __binary_search(all_ranges_, start_time_id)
            all_ranges_ = all_ranges_[start_index:] + all_ranges_[:start_index]
            start_index = 0
            
            # Since all_ranges_ are now unordered, we cannot use binary search 
            # to find end_index. So we just find them manually 
            for i in range(len(all_ranges_)):
                a_range = all_ranges_[i]
                if end_time_id in a_range:
                    end_index = i 
                    break 
        assert start_index <= end_index, "ERROR! start index is larger than end"

        # Only get the relevant ranges 
        relevant_ranges = []
        for i in range(start_index, end_index + 1):
            relevant_ranges.append(all_ranges_[i])
        
        # We have the relevant ranges, now to assign trajectory points to them
        # Only collect the raw trajectories 
        all_patterns = [[] for x in relevant_ranges]
        for i in range(len(relevant_ranges)):
            for point in trajectory_:
                if point[1][2] in relevant_ranges[i]:
                    all_patterns[i].append(point[1])
        
        # Patterns assigned, now to find the features 
        all_pattern_features = []
        for i in range(len(all_patterns)):
            if len(all_patterns[i]) <= 1:
                all_pattern_features.append([0,0])
            else:
                # Get the cyclical features 
                [x.append(self.__get_time_cyclical(x[2])) 
                 for x in all_patterns[i]]
                s_dist = 0 
                t_dist = 0 
                for j in range(1, len(all_patterns[i])):
                    cur_s = np.array(all_patterns[i][j][:2])
                    cur_t = np.array(all_patterns[i][j][-1])
                    prev_s = np.array(all_patterns[i][j-1][:2])
                    prev_t = np.array(all_patterns[i][j-1][-1])
                    s_dist += np.linalg.norm(cur_s - prev_s)
                    t_dist += np.linalg.norm(cur_t - prev_t)
                all_pattern_features.append([s_dist, t_dist])
        return all_pattern_features
                    
        
    def __get_time_cyclical(self, timestamp):
        """
        GConvert a timestamp to the sin and cos cyclical features scaled to the 
        range between 0 and 1 
        
        Args:
            timestamp: (int) Timestamp in minutes 
            
        Outputs:
            A list of floats for the sin and cos feature of the timestamp. 
        """
        seconds_sin = (math.sin(2 * math.pi * timestamp / \
                                self.__MINUTES_IN_DAY) + 1) / 2
        seconds_cos = (math.cos(2 * math.pi * timestamp / \
                                self.__MINUTES_IN_DAY) + 1) / 2
        return [seconds_sin, seconds_cos]
        
        
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
        nums_point = [round((1-dr) * len(trajectory)) - 2 \
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
                downsampled_trajs.append([trajectory_[i] for i in [0,-1]])
            else:
                # For the normal case, we randomly pick elements in order.
                trajectory_ = copy.deepcopy(trajectory)
                rand_indices = random.sample(range(1, len(trajectory_)-1),
                                             num_point)
                rand_indices.sort()
                downsampled_traj = [trajectory_[i] for i in rand_indices]
                
                # Don't forget to add the first and last points 
                downsampled_traj.insert(0, trajectory_[0])
                downsampled_traj.append(trajectory_[-1])
                downsampled_trajs.append(downsampled_traj)
        return downsampled_trajs
        

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
        for point_drop_rate in point_drop_rates:
            traj_mid = [x for x in trajectory[1:-1] \
                        if random.random() > point_drop_rate]
            downsampled_trajs.append([trajectory[0]] + traj_mid + 
                                     [trajectory[-1]])
        return downsampled_trajs


    def __create_pattern_ranges(self, span, stride):
        """
        Creates all the temporal ranges that determine the assignment of 
        trajectory points to patterns. 
        
        Args:
            span: (integer) The span is an integer that determines the size of 
                   each pattern. The unit is in minutes.
            stride: (integer) The stride is an integer that determines the gap 
                     between the start of one pattern and the next. 
                     
        Returns: 
            A list of ranges. 
        """
        all_ranges = []
        cur_timestamp = 0
        while cur_timestamp + span <= self.__MINUTES_IN_DAY:
            all_ranges.append(range(cur_timestamp, cur_timestamp + span))
            cur_timestamp += stride 
        return all_ranges
        

    def __distort_spatiotemporal_traj(self, traj, s_dist_rate, t_dist, bbox):
        """
        Distorts a trajectory both spatially and temporally 
        
        Args:
            traj: (list) List of trajectory points, containing both the 
                   spatial and temporal information 
            s_dist_rate: (float) The rate of the spatial distortion 
            t_dist: (integer) The maximum temporal distortion  (in minutes) 
            bbox: (shapely Polygon) A polygon that represents the valid area
        """
        traj_ = copy.deepcopy(traj) 
        # If s_dist_rate is 0, skip the spatial distortion 
        if s_dist_rate != 0:
            for traj_point in traj_:
                if random.random() < s_dist_rate:
                    self.__distort_spatial_fix(traj_point, bbox)
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
        
    def __distort_spatial_fix(self, traj_point, bbox):
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
        degree = random.randint(1,360)
        bearing = math.radians(degree)
        distort_dist = random.randint(0, 30)
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
