"""This module handles the writing of the result log"""

import datetime 
import pathlib 
import shutil 

class LogWriter():
    def write_results(self, output_dir, training_x, training_y, validation_x, 
                      validation_y, test_gt, test_q, topk_id, topk_weights,
                      train_time, predict_time, ks, results):
        """
        Prints the result log 
        """
        output_path = output_dir + "/log.txt"
        with open(output_path, 'w+') as log_file: 
            log_file.write("Finish datetime: " + str(datetime.datetime.now()))
            log_file.write("\n")
            log_file.write("Training X Shape: " + str(training_x.shape))
            log_file.write("\n")
            log_file.write("Training y Shape: " + str(training_y.shape))
            log_file.write("\n")
            log_file.write("Validation X Shape: " + str(validation_x.shape))
            log_file.write("\n")
            log_file.write("Validation y Shape: " + str(validation_y.shape))
            log_file.write("\n")
            log_file.write("Test GT Shape: " + str(test_gt.shape))
            log_file.write("\n")
            log_file.write("Test Q Shape: " + str(test_q.shape))
            log_file.write("\n")
            log_file.write("Top-k ID Shape: " + str(topk_id.shape))
            log_file.write("\n")
            log_file.write("Top-k weights Shape: " + str(topk_weights.shape))
            log_file.write("\n")
            log_file.write("Total training time: " + str(train_time))
            log_file.write("\n")
            log_file.write("Total prediction time: " + str(predict_time))
            log_file.write("\n")
            log_file.write("All k: " + str(ks))
            log_file.write("\n")
            log_file.write("Top-k results: " + str(results[0]))
            if results[1] is not None:
                log_file.write("\n")
                log_file.write("Mean rank: " + str(results[1]))
        
        
    def write_train_results(self, output_dir, training_x, training_y, 
                            validation_x, validation_y, topk_id,topk_weights,
                            loss_weights, train_time):
        """
        Prints the training log 
        """
        output_path = output_dir + "/log_train.txt"
        with open(output_path, 'w+') as log_file: 
            log_file.write("Finish datetime: " + str(datetime.datetime.now()))
            log_file.write("\n")
            log_file.write("Training X Shape: " + str(training_x.shape))
            log_file.write("\n")
            log_file.write("Training y Shape: " + str(training_y.shape))
            log_file.write("\n")
            log_file.write("Validation X Shape: " + str(validation_x.shape))
            log_file.write("\n")
            log_file.write("Validation y Shape: " + str(validation_y.shape))
            log_file.write("\n")
            log_file.write("Top-k ID Shape: " + str(topk_id.shape))
            log_file.write("\n")
            log_file.write("Top-k weights Shape: " + str(topk_weights.shape))
            log_file.write("\n")
            log_file.write("Loss weights: " + str(loss_weights))
            log_file.write("\n")
            log_file.write("Total training time: " + str(train_time))
            
    
    def write_eval_results(self, output_dir, test_gt, test_q, predict_time, ks, 
                           results):
        """
        Prints the evaluation log 
        """
        output_path = output_dir + "/log_eval.txt"
        with open(output_path, 'w+') as log_file: 
            log_file.write("Finish datetime: " + str(datetime.datetime.now()))
            log_file.write("\n")
            log_file.write("Test GT Shape: " + str(test_gt.shape))
            log_file.write("\n")
            log_file.write("Test Q Shape: " + str(test_q.shape))
            log_file.write("\n")
            log_file.write("Total prediction time: " + str(predict_time))
            log_file.write("\n")
            log_file.write("All k: " + str(ks))
            log_file.write("\n")
            log_file.write("Top-k results: " + str(results[0]))
            if results[1] is not None:
                log_file.write("\n")
                log_file.write("Mean rank: " + str(results[1]))
            log_file.write("\n")
            log_file.write("Average evaluation time: " + str(results[2]))


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