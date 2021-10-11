#############################
#          MODULES
#############################
import os, json, itertools
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import torch.nn as nn

from compile import create_experiment, create_heatmap
from utils import Opts, TypeDataset, TypeModel, TypeOptimizer, generate_model_string, generate_tuples_from_dict










######################################
#     EXAMPLE OF NN EXPERIMENTS
######################################

# Read user-given parameters
with open('config.json', 'r') as file_read :
    list_of_parameters = json.load(file_read)

dataset_type, type_of_model, dict_of_lists = list_of_parameters
model_string = generate_model_string(dataset_type, type_of_model)

# Create the folders for the data & results
data_path = os.path.split(os.getcwd())[0] + '/data'
results_path = os.path.split(os.getcwd())[0] + '/results'
best_params_path = os.path.split(os.getcwd())[0] + '/results/best-params'

if not(os.path.exists(data_path)) :
    os.mkdir(data_path)
if not(os.path.exists(results_path)) :
    os.mkdir(results_path)
if not(os.path.exists(best_params_path)) :
    os.mkdir(best_params_path)

# Define dictionaries that we will use later
tuples_list_dict = dict()
string_params = dict()

# Generate tuples from the above dictionaries
tuples_list_dict[TypeOptimizer.AAMMSU] = generate_tuples_from_dict(dict_of_lists[TypeOptimizer.AAMMSU], amsgrad = False, generate_pairs_option = True)
tuples_list_dict[TypeOptimizer.AMSGRAD] = generate_tuples_from_dict(dict_of_lists[TypeOptimizer.AMSGRAD], amsgrad = True, generate_pairs_option = False)

# Define the dictionary for the heatmaps
keys_AAMMSU = []
for key in dict_of_lists[TypeOptimizer.AAMMSU] :
    if len(dict_of_lists[TypeOptimizer.AAMMSU][key]) > 1 :
        keys_AAMMSU.append(key)

string_params[TypeOptimizer.AAMMSU] = []
res_AAMMSU = itertools.combinations(keys_AAMMSU, 2)
for pair in res_AAMMSU :
    string_params[TypeOptimizer.AAMMSU].append(list(pair))

string_params[TypeOptimizer.AMSGRAD] = [["lr", "batch_size"]]


# Construct experiment
print('START EXPERIMENT : ' + model_string.upper() + ' ... \n\n')
print('device: ', Opts.device)
print('n_gpu: ', Opts.ngpu)
print('\n******************************** \n\n')

if len(string_params[TypeOptimizer.AAMMSU]) > 0 and len(string_params[TypeOptimizer.AMSGRAD]) > 0 :
    create_experiment(tuple_list_dict = tuples_list_dict, epochs = Opts.epochs, n_runs = Opts.n_runs, dataset_type = dataset_type, model_type = type_of_model, model_string = model_string, 
    generate_plots_option = True, generate_heatmaps_option = True, string_params = string_params)    
else :
    create_experiment(tuple_list_dict = tuples_list_dict, epochs = Opts.epochs, n_runs = Opts.n_runs, dataset_type = dataset_type, model_type = type_of_model, model_string = model_string, 
    generate_plots_option = True, generate_heatmaps_option = False, string_params = string_params)  
    print('\n\n FINISHED EXPERIMENT : ' + model_string.upper() + ' ... \n')
print('________________________________________________________________________________________________________________________________________________________________________________ \n')
print('________________________________________________________________________________________________________________________________________________________________________________ \n')
print('________________________________________________________________________________________________________________________________________________________________________________ \n\n\n')

