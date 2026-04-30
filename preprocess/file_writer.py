"""Handles the writing of data to files"""

import h5py 
import math
import numpy as np 
import os 
import pathlib 
import shutil 

class FileWriter():
    """
    This class handles the printing of data to files. 
    """
    def write_train_data(self, data, file_name_x, file_name_y, 
                         file_name_log, output_directory, train_segment_size,
                         num_lines):
        """
        Writes the training data to files in the specified directory.
        
        Args:
            data_x: (numpy array) The input data to be used for training 
            data_y: (numpy array) The ground truth data to be used for training
            file_name_x: (string) Identifier to add to the file name for the 
                          x data 
            file_name_y: (string) Identifier to add to the file name for the y 
                          labels
            file_name_log: (string) The name of the output file 
            output_directory: (string) The output directory for the data files 
            train_segment_size: (int) The size of each segment of training data. 
                                 Specify a positive value to divide the 
                                 training data to segments of the specified  
                                 size. Each segment will be written to its own 
                                 file 
            num_lines: (int) The actual number of lines read from the input 
                        data file. Required for processing the test data 
        """
        # data_x and data_y are generators, but we need them in numpy arrays 
        data_x = []
        data_y = []

        for x in data:
            for item in x[0]:
                data_x.append(item)
            for item in x[1]:
                data_y.append(item)

        data_x = np.array(data_x)
        data_y = np.array(data_y)


        # Writes to .npy files
        # If train_segment_size is <= 0, print to one file 
        if train_segment_size <= 0:
            output_path_x = pathlib.Path(output_directory) / (file_name_x)
            output_path_y = pathlib.Path(output_directory) / (file_name_y)
            np.save(output_path_x, data_x)
            np.save(output_path_y, data_y)
        else:
            # If not, create a nested directory that will contain the training
            # data segments 
            out_dir_x = "./"+output_directory + "/" + file_name_x
            out_dir_y = "./"+output_directory + "/" + file_name_y
            print("out_dir_x")
            print(out_dir_x)
            print("out_dir_y")
            print(out_dir_y)
            print("trainseg", train_segment_size)
            if not os.path.exists(out_dir_x):
                os.mkdir(out_dir_x)
            if not os.path.exists(out_dir_y):
                os.mkdir(out_dir_y)
            
            # Get num. of leading zeroes for the training files naming 
            n_zero = len(str(math.ceil(len(data_x) / train_segment_size)))
            
            seg_sta = 0 
            seg_end = seg_sta + train_segment_size
            seg_num = 0 
            while seg_end <= len(data_x):
                seg_num += 1
                seg_name_x = file_name_x + "_" + str(seg_num).zfill(n_zero)
                seg_name_y = file_name_y + "_" + str(seg_num).zfill(n_zero)
                print("seg_name_x", seg_name_x)
                print("seg_name_y", seg_name_y)
                output_path_x = pathlib.Path(out_dir_x) / seg_name_x
                output_path_y = pathlib.Path(out_dir_y) / seg_name_y
                print("output_path_x",output_path_x)
                print("output_path_y",output_path_y)
                np.save(output_path_x, data_x[seg_sta:seg_end])
                np.save(output_path_y, data_y[seg_sta:seg_end])
                seg_sta += train_segment_size
                seg_end += train_segment_size
                
        # Write the log 
        output_path = pathlib.Path(output_directory) / (file_name_log)
        with open(output_path.with_suffix(".txt"), 'w') as f:
            f.write("Dataset contents: 'data_x', 'data_y'" )
            f.write("\ndata_x shape: " + str(data_x.shape))
            f.write("\ndata_y shape: " + str(data_y.shape))
            f.write("\nLines read: " + str(num_lines))
        
    
    def write_test_data(self, data_q, data_gt, file_name, output_directory):
        """
        Writes the test data to files in the specified directory 
        
        Args:
            data_q: (numpy array) Numpy array containing the query data 
            data_gt: (numpy array) Numpy array containing the ground truth data 
            file_name: (string) Identifier to add to the file name 
            output_directory: (string) The output directory for the data files 
        """
        # Writes to .npy files
        output_path = pathlib.Path(output_directory) / (file_name)
        output_path_gt = pathlib.Path(output_directory) / (file_name + "_gt")
        output_path_q = pathlib.Path(output_directory) / (file_name + "_q")
        np.save(output_path_gt, data_gt)
        np.save(output_path_q, data_q)
        
        with open(output_path.with_suffix(".txt"), 'w') as f:
            f.write("Dataset contents: 'data_gt', 'data_q'" )
            f.write("\ndata_gt shape: " + str(data_gt.shape))
            f.write("\ndata_q shape: " + str(data_q.shape))
    
            
    def write_test_data_split(self, data_q, data_qdb, data_db, data_qraw, 
                              data_qdbraw, data_dbraw, num_q, nums_db, 
                              test_q_name, test_db_name, output_directory,
                              lines_read):
        """
        Writes the test data for the case when the data selection mode is 
        "split". Basically, this prints only one query, but multiple databases 
        
        Args:
            data_q: (numpy array) Numpy array containing the query data 
            data_qdb: (numpy array) Numpy array containing the other half 
                       of each query's trajectory. This'll be added to 
                       each db data 
            data_db: (numpy array) Numpy array containing the database data 
            num_q: (int) Number of query trajectories 
            nums_db: (list) List of number of database trajectories 
            test_q_name: (string) Identifier to add to the query files names 
            test_db_name: (string) Identifier to add to the database file names
            output_directory: (string) The output directory for the data files 
            lines_read: (int) The number of lines read to produce the test 
                         data (includes the skipped lines from the trianing 
                         data) 
        """
        # Write q to an .npy file
        # Gridded trajectories 
        path_q = pathlib.Path(output_directory) / ("1_" + test_q_name)
        np.save(path_q, data_q)
        
        # Raw trajectories 
        path_qraw = pathlib.Path(output_directory) / ("1_raw_" + test_q_name)
        np.save(path_qraw, data_qraw)
        
        # Write all dbs to .npy files 
        db_shapes = []
        dbraw_lens = []
        for i in range(len(nums_db)):
            # Gridded trajectories 
            path_db = str(i+1) + "_" + test_db_name
            output_path_db = pathlib.Path(output_directory) / path_db
            data_qdb_db = np.concatenate((data_qdb, data_db[:nums_db[i]]))
            db_shapes.append(data_qdb_db.shape)
            np.save(output_path_db, data_qdb_db) 
            
            # Raw trajectories 
            path_dbraw = str(i+1) + "_raw_" + test_db_name
            output_path_dbraw = pathlib.Path(output_directory) / path_dbraw
            data_qdb_db_raw = np.concatenate((data_qdbraw, 
                                              data_dbraw[:nums_db[i]]))
            dbraw_lens.append(data_qdb_db_raw.shape)
            np.save(output_path_dbraw, data_qdb_db_raw)
        
        output_path = pathlib.Path(output_directory) / ("1_test_log")
        with open(output_path.with_suffix(".txt"), 'w') as f:
            f.write("Dataset contents: 'data_gt', 'data_q', 'data_gt_raw " +
                     "data_q_raw")
            f.write("\ndata_q shape: " + str(data_q.shape))
            f.write("\ndata_db shape: " + str([x for x in db_shapes]))
            f.write("\ndata_qraw len: " + str(data_qraw.shape))
            f.write("\ndata_dbraw len:" + str([x for x in dbraw_lens]))
            f.write("\nlines read:" + str(lines_read))
  
  
    def write_topk(self, topk_id, topk_weight, file_name_id, file_name_weight,
                   file_name_log, output_directory):
        """
        Writes the top-k nearest neighbors data to output files 
        
        Args:
            topk_id: (numpy array) The array containing the top-k closest 
                      cells' ID data 
            topk_weight: (numpy array) The array containig the top-k closest 
                          cells' weight data 
            file_name_id: (string) Output file name for the topk IDs
            file_name_weight: (string) Output file name for the topk weights
            file_name_log: (string) Name for the log file 
            output_directory: (string) Directory to write the file to
        """
        # Writes to .npy files
        output_path_log = pathlib.Path(output_directory) / (file_name_log)
        output_path_id = pathlib.Path(output_directory) / (file_name_id)
        output_path_weight = pathlib.Path(output_directory) / (file_name_weight)
        np.save(output_path_id, topk_id)
        np.save(output_path_weight, topk_weight)
        
        with open(output_path_log.with_suffix(".txt"), 'w') as f:
            f.write("Dataset contents: 'topk_id', 'topk_weight'")
            f.write("\ntopk_id shape: " + str(topk_id.shape))
            f.write("\ntopk_weight shape: " + str(topk_weight.shape))
            
    
    def write_cell_dict(self, cell_dict, file_name, output_directory):
        """
        Writes the cell dictionary to a file. This dictionary maps the raw 
        string-based IDs to the new int-based, which is required for the model 
        
        Args:
            cell_dict: (dictionary) Cell dictionary to be written to file 
            file_name: (string) File name header for the output file 
            output_directory: (string) Output directory for the file 
        """
        output_path_cell_dict = pathlib.Path(output_directory) / (file_name)
        np.save(output_path_cell_dict, cell_dict)
        
        
    def write_cells(self, all_cells, file_name, output_directory):
        """
        Writes all cells to a file. The cells are stored in a numpy array. This 
        numpy array allows the mapping from raw lat-lng coordinates to cell IDs 
        
        Args:
            all_cells: (numpy array) Numpy array containing all cells and their 
                        properties 
            file_name: (string) File name header for the output file 
            output_directory: (string) Output directory for the file 
        """
        output_path_cells = pathlib.Path(output_directory) / file_name
        np.save(output_path_cells, all_cells)
        
        
    def copy_ini_file(self, ini_path, output_directory):
        """
        Copies the input .ini file to the output directory 
        
        Args:
            ini_path: (string) The path to the .ini file 
            output_directory: (string) The output directory 
        """ 
        fname = ini_path.split("/")[-1]
        output_path = pathlib.Path(output_directory) / (fname)
        shutil.copyfile(ini_path, output_path)