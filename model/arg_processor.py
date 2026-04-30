"""This module handles the processing of the input arguments to the program"""

import ast 
import configparser
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
        
        # Reads the arguments 
        self.is_training = config['MODE']['IsTraining']
        self.is_evaluating = config['MODE']['IsEvaluating']
        
        self.training_x_path = config['DIRECTORY']['TrainingXPath']
        self.training_y_path= config['DIRECTORY']['TrainingYPath']
        self.validation_x_path = config['DIRECTORY']['ValidationXPath']
        self.validation_y_path = config['DIRECTORY']['ValidationYPath']
        self.test_gt_path = config['DIRECTORY']['TestGTPath']
        self.test_q_path = config['DIRECTORY']['TestQPath']
        self.topk_id_path = config['DIRECTORY']['TopKIDPath']
        self.topk_weights_path = config['DIRECTORY']['TopKWeightsPath']
        self.output_directory = config['DIRECTORY']['OutputDirectory']
        
        self.model_path = config['TRAINING']['ModelPath']
        self.batch_size = int(config['TRAINING']['BatchSize'])
        self.triplet_margin = float(config['TRAINING']['TripletMargin'])
        self.epochs = int(config['TRAINING']['Epochs'])
        self.patience = int(config['TRAINING']['Patience'])
        self.loss_weights = ast.literal_eval(config['TRAINING']['LossWeights'])
        
        self.gru_cell_size = int(config['MODEL']['GRUCellSize'])
        self.num_gru_layers = int(config['MODEL']['NumGruLayers'])
        self.gru_dropout_ratio = float(config['MODEL']['GRUDropoutRatio'])
        self.embedding_size = int(config['MODEL']['EmbeddingSize'])
        self.embedding_vocab_size = config['MODEL']['EmbeddingVocabSize']
        self.traj_repr_size = int(config['MODEL']['TrajReprSize'])
        self.bidirectional = bool(config['MODEL']['Bidirectional'])
        self.use_attention = bool(config['MODEL']['UseAttention'])
        
        self.ks = ast.literal_eval(config['PREDICTION']['KS'])
        self.predict_batch_size = int(config['PREDICTION']['PredictBatchSize'])
        self.use_mean_rank = bool(config['PREDICTION']['UseMeanRank'])
        self.ks.sort()
        
        self.gpu_used = ast.literal_eval(config['GPU']['GPUUsed'])
        self.gpu_memory = float(config['GPU']['GPUMemory'])
        
        # Boolean values 
        if self.is_training.lower() == 'true':
            self.is_training = True 
        elif self.is_training.lower() == 'false':
            self.is_training = False 
        else:
            raise ValueError("IsTraining must either be 'true or 'false'")
            
        if self.is_evaluating.lower() == 'true':
            self.is_evaluating = True 
        elif self.is_evaluating.lower() == 'false':
            self.is_evaluating = False 
        else:
            raise ValueError("IsEvaluating must either be 'true or 'false'")
        
        # Check if all the inputs files are valid files 
        # Only perform checks on files that are going to be used. So if 
        # only training is done, only training and validation files will be 
        # checked. 
        if self.is_training:
            if not (os.path.isfile(self.training_x_path) or 
                    os.path.isdir(self.training_x_path)):
                raise IOError("%s is not a valid file or directory" % 
                               self.training_x_path)
            if not (os.path.isfile(self.training_y_path) or 
                    os.path.isdir(self.training_y_path)):
                raise IOError("%s is not a valid file or directory" % 
                               self.training_y_path)
            if not (os.path.isfile(self.validation_x_path) or 
                    os.path.isdir(self.validation_x_path)):
                raise IOError("%s is not a valid file or directory" % 
                               self.validation_x_path)
            if not (os.path.isfile(self.validation_y_path) or 
                    os.path.isdir(self.validation_y_path)):
                raise IOError("%s is not a valid file or directory" % 
                               self.validation_y_path)
        if self.is_evaluating:
            if not (os.path.isfile(self.test_gt_path) or 
                    os.path.isdir(self.test_gt_path)):
                raise IOError("%s is not a valid file or directory" % 
                               self.test_gt_path)
            if not (os.path.isfile(self.test_q_path) or 
                    os.path.isdir(self.test_q_path)):
                raise IOError("%s is not a valid file or directory" % 
                               self.test_q_path)
        if not os.path.isfile(self.topk_id_path):
            raise IOError("'" + self.topk_id_path + "' is not a valid file")
        if not os.path.isfile(self.topk_weights_path):
            raise IOError("'" + self.topk_weights_path + "' is not a valid file")
        if not os.path.isdir(self.output_directory):
            print("Output directory does not exist. Creating...")
            os.makedirs(self.output_directory) 
            
        # Check numerical features 
        if self.batch_size <= 0:
            raise ValueError("BatchSize must be greater than 0")
        if self.gru_cell_size <= 0:
            raise ValueError("GRUCellSize must be greater than 0")
        if self.num_gru_layers <= 0:
            raise ValueError("NumGruLayers must be greater than 0")
        if self.gru_dropout_ratio <= 0 or self.gru_dropout_ratio >= 1:
            raise ValueError("GRUDropoutRatio must be between 0 and 1 exclusive")
        if self.traj_repr_size <= 0:
            raise ValueError("EmbeddingSize must be greater than 0")
        if self.predict_batch_size < 0:
            raise ValueError("PredictBatchSize must be greater than 0. If " +
                             "a batch size of 0 is provided, training is " + 
                             "done on the full data at once.")
        if self.embedding_vocab_size.isdigit():
            self.embedding_vocab_size = int(self.embedding_vocab_size)
        else:
            if self.embedding_vocab_size.lower() == "none":
                self.embedding_vocab_size = None 
            else: 
                raise ValueError("EmbeddingVocabSize must be greater " +
                                 "than 0, or the string 'None'")
        if len(self.ks) == 0:
            raise ValueError("KS must contain at least one integer.")  
        for k in self.ks:
            if not isinstance(k, int):
                raise ValueError("The values in KS must all be integers")
            if k < 1:
                raise ValueError("All values in KS must not be 0 or negative") 
        for x in self.gpu_used:
            if not isinstance(x, int):
                raise ValueError("The values in GPUUsed must all be integers")
        if self.gpu_memory <= 0:
            raise ValueError("GPUMemory must be a positive value")
        if self.patience < 0:
            raise ValueError("Patience must not be a negative integer")
        if isinstance(self.loss_weights, list):
            if sum(self.loss_weights) <= 0:
                raise ValueError("The sum of values in LossWeights must be " + 
                                 "greater than 0")
            for x in self.loss_weights:
                if not isinstance(x, (int, float)):
                    raise ValueError("All values in LossWeights must be an " +
                                     "integer or a float")
                elif x < 0:
                    raise ValueError("All values in LossWeights must be " +
                                     "greater than 0")
        elif self.loss_weights is not None:
            raise ValueError("LossWeights must either be a list of " +
                             "integers/floats or the string 'None'")