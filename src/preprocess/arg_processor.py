"""This module processes the arguments given by the .ini file"""

from math import ceil 
import ast
import configparser
import decimal 
import os 

class ArgProcessor():
    """Class that handles the .ini arguments"""
    
    def __init__(self, ini_path):
        """
        Reads the arguments from the input .ini file and checks their validity
        
        Args:
            ini_path: The path to the input .ini file 
        """
        # Read the .ini file 
        config = configparser.ConfigParser()
        config.read(ini_path)
        self.__MINUTES_IN_A_DAY = 1440 
        
        ## MODE section 
        self.process_train_val = config['MODE']['ProcessTrainVal']
        self.process_test = config['MODE']['ProcessTest']
        # Check validity 
        # Transforms strings to booleans
        if self.process_train_val.lower() == "true":
            self.process_train_val = True 
        elif self.process_train_val.lower() == "false":
            self.process_train_val = False 
        if self.process_test.lower() == "true":
            self.process_test = True 
        elif self.process_test.lower() == "false":
            self.process_test = False 
        
        ## GENERAL section 
        self.input_file_path = config['GENERAL']['InputFilePath']
        self.output_directory = config['GENERAL']['OutputDirectory']
        self.dataset_mode = config['GENERAL']['DatasetMode'].lower() 
        # Check validity 
        # Check if the input_file is a valid file 
        if (not os.path.isfile(self.input_file_path)):
            raise IOError("'" + self.input_file_path + "' must be a file.'")
        # If the output_directory does not exist, create it 
        if not os.path.exists(self.output_directory):
            os.mkdir(self.output_directory)
        # Check data mode validity 
        self.dataset_modes = ['porto', 'didi']
        if self.dataset_mode not in self.dataset_modes:
            raise ValueError("'" + self.dataset_mode + "' is not one of the " + 
                             "valid dataset modes.")
        
        ## PRUNING section  
        self.min_trajectory_length = int(config['PRUNING']
                                               ['MinTrajectoryLength'])
        self.max_trajectory_length = int(config['PRUNING']
                                               ['MaxTrajectoryLength'])
        self.max_pattern_length = config['PRUNING']['MaxPatternLength']
        self.hot_cells_threshold = int(config['PRUNING']['HotCellsThreshold'])
        # Check validity 
        if self.min_trajectory_length > self.max_trajectory_length:
            raise ValueError('min_trajectory length is larger than ' +
                             'max_trajectory_length')
        if self.min_trajectory_length < 0:
            raise ValueError("MinTrajectoryLength must not be negative")
        if self.max_trajectory_length < 0:
            raise ValueError("MaxTrajectoryLength must not be negative")
        
        ## GRID section 
        self.topk_id_name = config['GRID']['TopKIDName']
        self.topk_weight_name = config['GRID']['TopKWeightName']
        self.topk_log_name = config['GRID']['TopKLogName']
        self.cell_dict_name = config['GRID']['CellDictName']
        self.all_cells_name = config['GRID']['AllCellsName']
        self.bounding_box_coords = ast.literal_eval(config['GRID']
                                                          ['BoundingBoxCoords'])
        self.spatial_grid_lat = int(config['GRID']['SpatialGridLat'])
        self.spatial_grid_lng = int(config['GRID']['SpatialGridLng'])
        self.temporal_grid_length = int(config['GRID']['TemporalGridLength'])
        self.k = int(config['GRID']['K'])
        # Check validity 
        if self.bounding_box_coords[0] > self.bounding_box_coords[2]:
            raise ValueError('The minimum latitude is larger than the maximum')
        if self.bounding_box_coords[1] > self.bounding_box_coords[3]:
            raise ValueError('The minimum longitude is larger than the maximum')
        if not self.__MINUTES_IN_A_DAY % self.temporal_grid_length == 0:
            raise ValueError('Invalid temporal_grid_length. For it to be ' + 
                             'valid, 1440 (minutes in a day must be ' + 
                             'divisible by it.')
        if self.temporal_grid_length <= 0:
            raise ValueError('TemporalGridLength must be greater than 0')
        if self.k <= 0:
            raise ValueError('K must be greater than 0')
        
        ## TRAINVAL section 
        self.train_x_name = config['TRAINVAL']['TrainXName']
        self.train_y_name = config['TRAINVAL']['TrainYName']
        self.train_log_name = config['TRAINVAL']['TrainLogName']
        self.val_x_name = config['TRAINVAL']['ValXName']
        self.val_y_name = config['TRAINVAL']['ValYName']
        self.val_log_name = config['TRAINVAL']['ValLogName']
        self.num_train = int(config['TRAINVAL']['NumTrain'])
        self.train_segment_size = int(config['TRAINVAL']['TrainSegmentSize'])
        self.val_segment_size = int(config['TRAINVAL']['ValSegmentSize'])
        self.num_val = int(config['TRAINVAL']['NumVal'])
        self.point_drop_rates = ast.literal_eval(config['TRAINVAL']
                                                       ['PointDropRates'])
        self.spatial_distortion_rates = ast.literal_eval(config['TRAINVAL']
                                                     ['SpatialDistortionRates'])
        self.temporal_distortions = ast.literal_eval(config['TRAINVAL']
                                                    ['TemporalDistortions'])
        ## Check validity 
        if self.num_train <= 0: 
            raise ValueError("NumTrain must be greater than 0")
        if self.num_val <= 0:
            raise ValueError("NumVal must be greater than 0")
        for x in self.point_drop_rates:
            if x < 0 or x > 1:
                raise ValueError("All values in PointDropRates must be " +
                                 "between 0 and 1 inclusive.")
        for x in self.spatial_distortion_rates:
            if x < 0 or x > 1:
                raise ValueError("All values in SpatialDistortionRates must " + 
                                 "be between 0 and 1")
        for x in self.temporal_distortions:
            if x < 0:
                raise ValueError("All values in TemporalDistortions must be " + 
                                 "0 or greater.")
        
        ## TEST section 
        self.test_q_name = config['TEST']['TestQName']
        self.test_db_name = config['TEST']['TestDBName']
        self.num_q = int(config['TEST']['NumQ'])
        self.nums_db = ast.literal_eval(config['TEST']['NumsDB'])
        self.drop_rate = float(config['TEST']['DropRate'])
        self.test_spatial_distortion = float(config['TEST']\
                                                 ['TestSpatialDistortion'])
        self.test_temporal_distortion = int(config['TEST']\
                                                  ['TestTemporalDistortion'])
        # Check validity 
        if self.num_q <= 0:
            raise ValueError("NumQ must be greater than 0.") 
        for x in self.nums_db:
            if x <= 0:
                raise ValueError("All values in NumsDB must be greater than 0.")
        if self.drop_rate < 0 or self.drop_rate > 1:
            raise ValueError("DropRate must be between 0 and 1")
        if self.test_spatial_distortion < 0 or self.test_spatial_distortion > 1:
            raise ValueError("TestSpatialDistortion must be between 0 and 1")
        if self.test_temporal_distortion < 0:
            raise ValueError("TestTemporalDistortion must not be negative")
       
        ## PATTERN section 
        self.span = int(config['PATTERN']['Span'])
        self.stride = int(config['PATTERN']['Stride'])
        # Check validity 
        if self.stride > self.span:
            raise ValueError('The stride is larger than the span. This is ' +
                             'disallowed as having a stride larger than the ' +
                             'span means that some trajectory points may be ' +
                             'skipped')
        if self.span <= 0:
            raise ValueError("Span must be greater than 0")
        if self.stride <= 0:
            raise ValueError("Stride must be greater than 0")
        if not self.__MINUTES_IN_A_DAY % self.span == 0:
            raise ValueError('Invalid Span. 1440 must be divisible by Span')
        if not self.__MINUTES_IN_A_DAY % self.stride == 0:
            raise ValueError('Invalid Stride. 1440 must be divisible by Stride')
        
        
        
        