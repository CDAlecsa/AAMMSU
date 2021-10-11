###########################
#          MODULES
###########################
import os, json, torch
import torch.nn as nn

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import seaborn as sns
sns.color_palette("magma", as_cmap = True)

from utils import Opts, TypeVGG, TypeResNet, TypeModel, TypeDataset, plot_values, TypeOptimizer, TypeStatistic, Scheduler_type
from nn_modules import VGG, CNN, LR, ResNet18
from functools import reduce

from AAMMSU_optim import AAMMSU
from datasets import generate_datasets










######################################
#      FUNCTION USED FOR TRAINING
######################################
def train_model(epoch, epochs, model, optimizer, loss_func, dataloaders, all_batch_lists, type_of_model = None) :

    # Prepare for training
    train_losses_on_batches, train_acc_on_batches = all_batch_lists
    model.train()
    
    # Variables
    total_steps = len(dataloaders['train'])
    running_train_loss = 0
    correct_train = 0
    total_train = 0


    # For the given epoch, loop over batches
    for i, train_data in enumerate(dataloaders['train']) :

        # Take the images & labels from the current batch
        images, labels = train_data[0].to(Opts.device), train_data[1].to(Opts.device)

        # The LogisticRegression model can be applied only on MNIST dataset (change the input/output dimension for other datasets)
        if type_of_model == TypeModel.LR :
            images = images.view(-1, 28 * 28)

        # Process the output of the current model
        train_output = model(images)            

        # Go through the backward phase (backpropagation)
        loss = loss_func(train_output, labels)
        optimizer.zero_grad()           
        loss.backward()    

        # Apply the optimizer
        optimizer.step()     

        # Compute the loss for the current batch
        current_train_loss = loss.item()
        running_train_loss += current_train_loss

        # Compute the predictions for the current batch
        _, train_predicted = train_output.max(1)
        current_total_train = labels.size(0)
        current_correct_train = train_predicted.eq(labels).sum().item() 

        # Compute the accuracy for the current batch
        total_train += current_total_train
        correct_train += current_correct_train
        train_accuracy_on_batch = 100.0 * correct_train / total_train

        # Append values corresponding to the current batch
        train_losses_on_batches.append(running_train_loss / (i + 1))
        train_acc_on_batches.append(train_accuracy_on_batch)

        if Opts.verbose :
            if (i + 1) % (int(total_steps / 6)) == 0 :
                print ('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}'.format(epoch, epochs, i + 1, total_steps, current_train_loss, train_accuracy_on_batch))


    # Loss & accuracy values after the current epoch
    train_loss = running_train_loss / len(dataloaders['train'])
    train_accuracy = 100.0 * correct_train / total_train

    # Option used for printing the results
    if Opts.verbose :
        print('---------------------------------------------------------------------------------- \n')
        print ('Epoch {}, Train Loss: {:.4f}, Train Accuracy: {:.2f}'.format(epoch, train_loss, train_accuracy))
        print('\n')
    
    # Return the values that we have obtained
    return train_losses_on_batches, train_acc_on_batches, train_loss, train_accuracy











####################################################
#      FUNCTION USED FOR VALIDATION & TESTING
####################################################
def evaluate_model(epoch, model, optimizer, loss_func, dataloaders, type_of_evaluation, type_of_model = None) :
    
    # Prepare for evaluation
    model.eval()

    # Variables
    total_steps = len(dataloaders[type_of_evaluation])
    running_evaluation_loss = 0
    correct_evaluation = 0
    total_evaluation = 0

    # Disable gradient computation
    with torch.no_grad() :

        # For the given epoch, loop over batches
        for i, evaluation_data in enumerate(dataloaders[type_of_evaluation]) :

            # Take the images & labels
            images, labels = evaluation_data[0].to(Opts.device), evaluation_data[1].to(Opts.device)

            if type_of_model == TypeModel.LR :
                images = images.view(-1, 28 * 28)

            # Process the output for the current model
            evaluation_output = model(images)         

            # Compute the loss for the current batch
            loss = loss_func(evaluation_output, labels)
            current_evaluation_loss = loss.item()
            running_evaluation_loss += current_evaluation_loss

            # Get the predictions
            _, evaluation_predicted = evaluation_output.max(1)
            current_total_evaluation = labels.size(0)
            current_correct_evaluation = evaluation_predicted.eq(labels).sum().item() 
            total_evaluation += current_total_evaluation
            correct_evaluation += current_correct_evaluation

    # Compute the loss & accuracy for the current epoch
    evaluation_loss = running_evaluation_loss / len(dataloaders[type_of_evaluation])
    evaluation_accuracy = 100.0 * correct_evaluation / total_evaluation

    # Option used for printing
    if Opts.verbose :
        print('---------------------------------------------------------------------------------- \n')
        print ('Epoch {}, Evaluation Loss: {:.4f}, Evaluation Accuracy: {:.2f}'.format(epoch + 1, evaluation_loss, evaluation_accuracy))
        print('================================================================================== \n\n\n')


    # Return the values that we have obtained above
    return evaluation_loss, evaluation_accuracy










####################################################
#      FUNCTION USED IN CREATING A MODEL
####################################################

def create_model(model_type) :

    # The LR case
    if model_type == TypeModel.LR :
        model = LR().to(Opts.device)

    # The CNN case
    if model_type == TypeModel.CNN :
        model = CNN().to(Opts.device)

    # The VGG case
    if model_type == TypeModel.VGG :
        current_vgg_str = TypeVGG.VGG_11
        model = VGG(current_vgg_str).to(Opts.device)
        
    # The ResNet case
    if model_type == TypeModel.ResNet :
        model = ResNet18().to(Opts.device)

    # CUDA option    
    if (Opts.device.type == 'cuda') and (Opts.ngpu > 1) :
        model = nn.DataParallel(model, list(range(Opts.ngpu)))

    # Return the model defined above
    return model










###################################################################
#         FUNCTION USED FOR TRAINING & EVALUATING A MODEL
###################################################################
def fit_eval_model(model, criterion, dataloaders, optimizer, epoch, epochs, batch_size, all_batch_lists, all_epoch_lists, scheduler = None, type_of_model = None) :

    # Train the model
    train_losses_on_batches, train_acc_on_batches = all_batch_lists
    train_acc_on_epochs, train_losses_on_epochs, validation_acc_on_epochs, validation_losses_on_epochs = all_epoch_lists
    
    train_losses_on_batches, train_acc_on_batches, train_loss, train_accuracy = train_model(epoch, epochs, model, optimizer, criterion, dataloaders, all_batch_lists, type_of_model)
    
    # Evaluate the model on the validation dataset
    validation_loss, validation_accuracy = evaluate_model(epoch, model, optimizer, criterion, dataloaders, 'validation', type_of_model)

    # Apply the scheduler at the end of the current epoch
    if Opts.scheduler_param and scheduler != None :
        scheduler.step()           

    # Append the results
    train_acc_on_epochs.append(train_accuracy)
    train_losses_on_epochs.append(train_loss)
    
    validation_acc_on_epochs.append(validation_accuracy)
    validation_losses_on_epochs.append(validation_loss)
    
    # Return the loss & accuracy values
    return train_losses_on_epochs, train_acc_on_epochs, train_losses_on_batches, train_acc_on_batches, validation_losses_on_epochs, validation_acc_on_epochs












#############################################################
#         FUNCTION USED FOR CREATING HEATMAPS
#############################################################
def create_heatmap(df, list_of_params, path_to_save, type_of_statistic) :

    # For the mean keep the highest mean of the accuracy, while for the std keep the lowest std value of the accuracy
    if type_of_statistic == TypeStatistic.mean_accuracy :
        df_copy = df.sort_values(list_of_params[-1], ascending = False).drop_duplicates([list_of_params[0], list_of_params[1]], keep = "first").copy()
    elif type_of_statistic == TypeStatistic.std_accuracy :
        df_copy = df.sort_values(list_of_params[-1], ascending = True).drop_duplicates([list_of_params[0], list_of_params[1]], keep = "first").copy()
    
    df_slice = df_copy.loc[:, ~df_copy.columns.isin(list_of_params + ['mean_train_accuracy', 'mean_validation_accuracy', 'mean_test_accuracy', 'mean_train_loss', 'mean_validation_loss', 'std_train_accuracy', 'std_validation_accuracy', 'std_test_accuracy', 'std_train_loss', 'std_validation_loss'] )]
    string_param_name = '---' + str(list(df_slice.keys())) + ' = ' + str(df_slice.values[0])

    results = df_copy.pivot(*list_of_params)
    ax = sns.heatmap(results, annot = True, linewidth = 1, linecolor = 'w')
    
    ax.figure.savefig(path_to_save + string_param_name + '.jpeg')
    ax.figure.savefig(path_to_save + string_param_name + '.eps')
    plt.clf()















######################################################################
#          FUNCTION USED FOR THE PLOTS INVOLVING MEAN & STD VALUES
######################################################################
def generate_plots(best_evaluation_params, n_runs, dataset_type, base_path_name, type_of_evaluation, type_of_model, model_string) :

    # Define the list of optimizers
    list_type_optimizers = [TypeOptimizer.AAMMSU, TypeOptimizer.AMSGRAD]

    # Empty dictionaries which will be used later
    train_acc_mean =  dict()
    validation_acc_mean =  dict()
    train_loss_mean =  dict()
    validation_loss_mean =  dict()

    train_acc_std =  dict()
    validation_acc_std =  dict()
    train_loss_std =  dict()
    validation_loss_std =  dict()


    # Loop over the list of optimizers in order to define the sub-dictionaries for the mean/std loss & accuracy values
    for type_of_optimizer in list_type_optimizers :

        train_acc_mean[type_of_optimizer] = []
        validation_acc_mean[type_of_optimizer] = []
        train_loss_mean[type_of_optimizer] = []
        validation_loss_mean[type_of_optimizer] = []

        train_acc_std[type_of_optimizer] = []
        validation_acc_std[type_of_optimizer] = []
        train_loss_std[type_of_optimizer] = []
        validation_loss_std[type_of_optimizer] = []



    # Loop over the list of optimizers
    for type_of_optimizer in list_type_optimizers :
        print('\nOPTIMIZER : ', type_of_optimizer, '\n')

        # Take the parameters corresponding to the given type of evaluation
        lr = best_evaluation_params[type_of_optimizer]['lr']
        batch_size = best_evaluation_params[type_of_optimizer]['batch_size']
        epochs = best_evaluation_params[type_of_optimizer]['epoch']
        beta_2 = best_evaluation_params[type_of_optimizer]['beta_2']
        eps = best_evaluation_params[type_of_optimizer]['eps']

        if type_of_optimizer == TypeOptimizer.AAMMSU :
            M = best_evaluation_params[type_of_optimizer]['M']
            mu = best_evaluation_params[type_of_optimizer]['mu']
            nu = best_evaluation_params[type_of_optimizer]['nu']
            tilde_gamma = best_evaluation_params[type_of_optimizer]['tilde_gamma']

        # Loop over the chosen number of runs
        for current_run in range(n_runs) :

            # Print the state of training
            print('Count Training : ', 100 * current_run / n_runs , '% \n')

            # Retrieve the PyTorch dataloaders
            dataloaders = generate_datasets(dataset_type, batch_size, Opts.split_ratio, True, False)
            
            # Define the model
            model = create_model(type_of_model)
        
            # Define the loss criterion
            criterion = nn.CrossEntropyLoss().to(Opts.device) 
            
            # Define the AAMMSU optimizer
            if type_of_optimizer == TypeOptimizer.AAMMSU :
                optimizer = AAMMSU(model.parameters(), lr = lr, M = M, mu = mu, nu = nu, tilde_gamma = tilde_gamma, beta_2 = beta_2, eps = eps)
            elif type_of_optimizer == TypeOptimizer.AMSGRAD :
                optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas = (0.9, beta_2), eps = eps, amsgrad = True)
            
            # Apply the learning rate scheduler
            if Opts.scheduler_param :
                if Opts.scheduler_type == Scheduler_type.step_lr :
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = Opts.scheduler_step, gamma = Opts.scheduler_coeff)
                elif Opts.scheduler_type == Scheduler_type.multi_step_lr :
                    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = Opts.scheduler_step, gamma = Opts.scheduler_coeff)
            else : 
                scheduler = None

            # Make empty lists for the results of the loss & accuracy
            train_losses_on_epochs, train_acc_on_epochs = [], []
            train_losses_on_batches, train_acc_on_batches = [], []
            validation_losses_on_epochs, validation_acc_on_epochs = [], []

            # Loop over epochs
            for epoch in range(1, epochs + 1) :

                # Make tuple variables
                all_batch_lists = (train_losses_on_batches, train_acc_on_batches)
                all_epoch_lists = (train_acc_on_epochs, train_losses_on_epochs, validation_acc_on_epochs, validation_losses_on_epochs)
                
                # Train & validate model
                train_losses_on_epochs, train_acc_on_epochs, train_losses_on_batches, train_acc_on_batches, validation_losses_on_epochs, validation_acc_on_epochs = fit_eval_model(model, criterion, dataloaders, optimizer, epoch, epochs, batch_size, all_batch_lists, all_epoch_lists, scheduler, type_of_model)
            
                # Create array for the results
                if epoch == 1 and current_run == 0 :
                    train_acc_results = np.zeros((n_runs, epochs))
                    validation_acc_results = np.zeros((n_runs, epochs))
                    train_loss_results = np.zeros((n_runs, epochs))
                    validation_loss_results = np.zeros((n_runs, epochs))


            # At the last epoch, we memorize the value of the current run
            train_acc_results[current_run, :] = train_acc_on_epochs
            validation_acc_results[current_run, :] = validation_acc_on_epochs
            
            train_loss_results[current_run, :] = train_losses_on_epochs
            validation_loss_results[current_run, :] = validation_losses_on_epochs

        # Take the mean values
        train_acc_mean[type_of_optimizer] = np.mean(train_acc_results, axis = 0)
        validation_acc_mean[type_of_optimizer] = np.mean(validation_acc_results, axis = 0)

        train_loss_mean[type_of_optimizer] = np.mean(train_loss_results, axis = 0)
        validation_loss_mean[type_of_optimizer] = np.mean(validation_loss_results, axis = 0)

        # Take the std values
        train_acc_std[type_of_optimizer] = np.std(train_acc_results, axis = 0)
        validation_acc_std[type_of_optimizer] =  np.std(validation_acc_results, axis = 0)

        train_loss_std[type_of_optimizer] = np.std(train_loss_results, axis = 0)
        validation_loss_std[type_of_optimizer] = np.std(validation_loss_results, axis = 0)

        # Make the paths for the plots
        save_plot_path = base_path_name + '/plots/'
        if not(os.path.exists(save_plot_path)) :
            os.mkdir(save_plot_path)

        save_file_name_acc = model_string + "++++" + "acc===best__params_" + type_of_evaluation + '_schd-' + str(Opts.scheduler_param) + '_type-' + str(Opts.scheduler_type) + '[' + str(Opts.scheduler_step) + '-' + str(Opts.scheduler_coeff) + "]"
        save_file_name_loss = model_string + "++++" + "loss===best__params_" + type_of_evaluation + '_schd-' + str(Opts.scheduler_param) + '_type-' + str(Opts.scheduler_type) + '[' + str(Opts.scheduler_step) + '-' + str(Opts.scheduler_coeff) + "]"



    # Define the strings for the legends
    legend_strings_AAMMSU_mean = ['train - mean - ' + TypeOptimizer.AAMMSU, 'validation - mean - ' + TypeOptimizer.AAMMSU]
    legend_strings_AMSGRAD_mean = ['train - mean - ' + TypeOptimizer.AMSGRAD, 'validation -mean - ' + TypeOptimizer.AMSGRAD]

    legend_strings_AAMMSU_band = ['train - error_band - ' + TypeOptimizer.AAMMSU, 'validation - error_band - ' + TypeOptimizer.AAMMSU]
    legend_strings_AMSGRAD_band = ['train - error_band - ' + TypeOptimizer.AMSGRAD, 'validation - error_band - ' + TypeOptimizer.AMSGRAD]

    legend_strings = legend_strings_AAMMSU_mean + legend_strings_AMSGRAD_mean + legend_strings_AAMMSU_band + legend_strings_AMSGRAD_band


    # Plot & save the images
    epoch_list_evaluation = [ best_evaluation_params[TypeOptimizer.AAMMSU]['epoch'], best_evaluation_params[TypeOptimizer.AMSGRAD]['epoch'] ]
    ranges_epoch_list = [range(1, epoch_item + 1) for epoch_item in epoch_list_evaluation]
    
    plot_values(ranges_epoch_list, 'Accuracy', legend_strings, [train_acc_mean, validation_acc_mean], [train_acc_std, validation_acc_std], save_plot_path + save_file_name_acc, n_runs) 
    plot_values(ranges_epoch_list, 'Loss', legend_strings, [train_loss_mean, validation_loss_mean], [train_loss_std, validation_loss_std], save_plot_path + save_file_name_loss, n_runs) 













#########################################################################
#         FUNCTION USED IN DESIGNING WITH CHOSEN COEFFICIENT TUPLES
#########################################################################

def create_experiment(tuple_list_dict, epochs, n_runs, dataset_type, model_type, model_string, generate_plots_option, generate_heatmaps_option, string_params) :

    # Set the count to 0
    count = 0

    # Dictionaries which will eventually contain the parameters for the best accuracy values
    best_validation_params, best_test_params = dict(), dict()
    
    best_validation_params[TypeOptimizer.AMSGRAD] = dict()
    best_validation_params[TypeOptimizer.AAMMSU] = dict()

    best_test_params[TypeOptimizer.AMSGRAD] = dict()
    best_test_params[TypeOptimizer.AAMMSU] = dict()

    # The list containing the string names of the AAMMSU & AMSGRAD optimizers
    list_type_optimizers = [TypeOptimizer.AAMMSU, TypeOptimizer.AMSGRAD]

    # Define the dictionary of DataFrames which will be used for the results we will obtain
    df = dict()
    df[TypeOptimizer.AMSGRAD] = dict()
    df[TypeOptimizer.AAMMSU] = dict()

    for epoch in epochs :
        df[TypeOptimizer.AMSGRAD][str(epoch)] = pd.DataFrame(columns = ['lr', 'lr_scheduler', 'eps', 'beta_2', 'batch_size', 'mean_train_accuracy', 'mean_validation_accuracy', 'mean_test_accuracy', 'mean_train_loss', 'mean_validation_loss', 
        'std_train_accuracy', 'std_validation_accuracy', 'std_test_accuracy', 'std_train_loss', 'std_validation_loss'])

        df[TypeOptimizer.AAMMSU][str(epoch)] = pd.DataFrame(columns = ['lr', 'lr_scheduler', 'M', 'mu', 'nu', 'tilde_gamma', 'eps', 'beta_2', 'batch_size', 'mean_train_accuracy', 'mean_validation_accuracy', 'mean_test_accuracy', 'mean_train_loss', 'mean_validation_loss', 
        'std_train_accuracy', 'std_validation_accuracy', 'std_test_accuracy', 'std_train_loss', 'std_validation_loss'])

    # Retrieve the length of the all the values, which will be used for the loop counting
    length = n_runs * ( len(tuple_list_dict[TypeOptimizer.AAMMSU]) + len(tuple_list_dict[TypeOptimizer.AMSGRAD]) )

    # Loop over the optimizers
    for type_of_optimizer in list_type_optimizers :

        index = 0
        best_validation_acc, best_test_acc = 0.0, 0.0

        # Loop over the parameters
        for elem_tuple in tuple_list_dict[type_of_optimizer] :

            print('\n\nTYPE OF OPTIMIZER: ', type_of_optimizer, '\n\n')
            print('\n\nPARAMETERS:\n ')
            if type_of_optimizer == TypeOptimizer.AAMMSU :
                lr, batch_size, M, nu, eps, beta_2, mu, tilde_gamma = elem_tuple
                print('lr = ', lr, '\t batch_size = ', batch_size, '\t M = ', M, '\t nu = ', nu, '\t eps = ', eps, '\t beta_2 = ', beta_2, '\t mu = ', mu, '\t tilde_gamma = ', tilde_gamma, '\n')
            elif type_of_optimizer == TypeOptimizer.AMSGRAD :
                lr, batch_size, eps, beta_2 = elem_tuple
                print('lr = ', lr, '\t batch_size = ', batch_size, '\t eps = ', eps, '\t beta_2 = ', beta_2, '\n')
                                        
            # Define empty dictionaries for the mean & std values
            train_acc_runs, validation_acc_runs, train_loss_runs, validation_loss_runs, test_acc_runs = dict(), dict(), dict(), dict(), dict()
            
            # Loop over the epochs in order to create the empty loss & accuracy values for each run
            for epoch in epochs :
                train_acc_runs[str(epoch)], validation_acc_runs[str(epoch)], train_loss_runs[str(epoch)], validation_loss_runs[str(epoch)], test_acc_runs[str(epoch)] = [], [], [], [], []

            # Loop over the chosen number of runs
            for current_run in range(n_runs) :

                # Print the state of training
                print('Count Training : ', 100 * count / length , '% \n')

                # Retrieve the PyTorch dataloaders
                dataloaders = generate_datasets(dataset_type, batch_size, Opts.split_ratio, True, False)
                
                # Define the model
                model = create_model(model_type)

                # Define the loss criterion
                criterion = nn.CrossEntropyLoss().to(Opts.device) 
                
                # Define the AAMMSU optimizer
                if type_of_optimizer == TypeOptimizer.AAMMSU :
                    optimizer = AAMMSU(model.parameters(), lr = lr, M = M, mu = mu, nu = nu, tilde_gamma = tilde_gamma, beta_2 = beta_2, eps = eps)
                elif type_of_optimizer == TypeOptimizer.AMSGRAD :
                    optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas = (0.9, beta_2), eps = eps, amsgrad = True)

                # Apply the learning rate scheduler
                if Opts.scheduler_param :
                    if Opts.scheduler_type == Scheduler_type.step_lr :
                        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = Opts.scheduler_step, gamma = Opts.scheduler_coeff)
                    elif Opts.scheduler_type == Scheduler_type.multi_step_lr :
                        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = Opts.scheduler_step, gamma = Opts.scheduler_coeff)
                else : 
                    scheduler = None

                # Make empty lists for the results of the loss & accuracy
                train_losses_on_epochs, train_acc_on_epochs = [], []
                train_losses_on_batches, train_acc_on_batches = [], []
                validation_losses_on_epochs, validation_acc_on_epochs = [], []

                # Loop over epochs
                for epoch in range(1, epochs[-1] + 1) :

                    # Make tuple variables
                    all_batch_lists = (train_losses_on_batches, train_acc_on_batches)
                    all_epoch_lists = (train_acc_on_epochs, train_losses_on_epochs, validation_acc_on_epochs, validation_losses_on_epochs)
                    
                    # Train & validate model
                    train_losses_on_epochs, train_acc_on_epochs, train_losses_on_batches, train_acc_on_batches, validation_losses_on_epochs, validation_acc_on_epochs = fit_eval_model(model, criterion, dataloaders, optimizer, epoch, epochs, batch_size, all_batch_lists, all_epoch_lists, scheduler, model_type)

                    # After the whole training, evaluate the model on the test dataset
                    if epoch in epochs :

                        # Retain the values for the current run
                        _, test_accuracy = evaluate_model(epoch, model, optimizer, criterion, dataloaders, 'test', model_type)
                        
                        test_acc_runs[str(epoch)].append(test_accuracy)

                        train_acc_runs[str(epoch)].append(train_acc_on_epochs[-1])
                        validation_acc_runs[str(epoch)].append(validation_acc_on_epochs[-1])
                        
                        train_loss_runs[str(epoch)].append(train_losses_on_epochs[-1])
                        validation_loss_runs[str(epoch)].append(validation_losses_on_epochs[-1])

                # Update the count for the next state of the current experiment
                count += 1
                



            # Set the parameters in the DataFrame of the results                                                                
            for epoch in epochs :

                df[type_of_optimizer][str(epoch)].loc[index, 'lr'] = lr
                df[type_of_optimizer][str(epoch)].loc[index, 'lr_scheduler'] = optimizer.param_groups[0]['lr']
                df[type_of_optimizer][str(epoch)].loc[index, 'eps'] = eps
                df[type_of_optimizer][str(epoch)].loc[index, 'batch_size'] = batch_size
                df[type_of_optimizer][str(epoch)].loc[index, 'beta_2'] = beta_2

                if type_of_optimizer == TypeOptimizer.AAMMSU :
                    df[type_of_optimizer][str(epoch)].loc[index, 'M'] = M
                    df[type_of_optimizer][str(epoch)].loc[index, 'mu'] = mu
                    df[type_of_optimizer][str(epoch)].loc[index, 'nu'] = nu
                    df[type_of_optimizer][str(epoch)].loc[index, 'tilde_gamma'] = tilde_gamma
                

                df[type_of_optimizer][str(epoch)].loc[index, 'mean_train_accuracy'] = np.mean(train_acc_runs[str(epoch)])
                df[type_of_optimizer][str(epoch)].loc[index, 'mean_validation_accuracy'] = np.mean(validation_acc_runs[str(epoch)])
                df[type_of_optimizer][str(epoch)].loc[index, 'mean_test_accuracy'] = np.mean(test_acc_runs[str(epoch)])

                df[type_of_optimizer][str(epoch)].loc[index, 'std_train_accuracy'] = np.std(train_acc_runs[str(epoch)])
                df[type_of_optimizer][str(epoch)].loc[index, 'std_validation_accuracy'] = np.std(validation_acc_runs[str(epoch)])
                df[type_of_optimizer][str(epoch)].loc[index, 'std_test_accuracy'] = np.std(test_acc_runs[str(epoch)])


                # Set the loss values in the DataFrame of the results
                df[type_of_optimizer][str(epoch)].loc[index, 'mean_train_loss'] = np.mean(train_loss_runs[str(epoch)])
                df[type_of_optimizer][str(epoch)].loc[index, 'mean_validation_loss'] = np.mean(validation_loss_runs[str(epoch)])
                
                df[type_of_optimizer][str(epoch)].loc[index, 'std_train_loss'] = np.std(train_loss_runs[str(epoch)])
                df[type_of_optimizer][str(epoch)].loc[index, 'std_validation_loss'] = np.std(validation_loss_runs[str(epoch)])

                # We update the best validation parameters
                if np.mean(validation_acc_runs[str(epoch)]) > best_validation_acc :
                    best_validation_acc = np.mean(validation_acc_runs[str(epoch)])

                    best_validation_params[type_of_optimizer]['lr'] = lr
                    best_validation_params[type_of_optimizer]['lr_scheduler'] = optimizer.param_groups[0]['lr']
                    best_validation_params[type_of_optimizer]['beta_2'] = beta_2
                    best_validation_params[type_of_optimizer]['eps'] = eps
                    best_validation_params[type_of_optimizer]['batch_size'] = batch_size
                    best_validation_params[type_of_optimizer]['epoch'] = epoch

                    if type_of_optimizer == TypeOptimizer.AAMMSU :
                        best_validation_params[type_of_optimizer]['M'] = M
                        best_validation_params[type_of_optimizer]['mu'] = mu
                        best_validation_params[type_of_optimizer]['nu'] = nu
                        best_validation_params[type_of_optimizer]['tilde_gamma'] = tilde_gamma


                # We update the best test parameters
                if np.mean(test_acc_runs[str(epoch)]) > best_test_acc :
                    best_test_acc = np.mean(test_acc_runs[str(epoch)])

                    best_test_params[type_of_optimizer]['lr'] = lr
                    best_test_params[type_of_optimizer]['lr_scheduler'] = optimizer.param_groups[0]['lr']
                    best_test_params[type_of_optimizer]['beta_2'] = beta_2
                    best_test_params[type_of_optimizer]['eps'] = eps
                    best_test_params[type_of_optimizer]['batch_size'] = batch_size
                    best_test_params[type_of_optimizer]['epoch'] = epoch

                    if type_of_optimizer == TypeOptimizer.AAMMSU :
                        best_test_params[type_of_optimizer]['M'] = M
                        best_test_params[type_of_optimizer]['mu'] = mu
                        best_test_params[type_of_optimizer]['nu'] = nu
                        best_test_params[type_of_optimizer]['tilde_gamma'] = tilde_gamma



            # Update the index of the DataFrame after we finish the simulations
            index += 1



        # Return the DataFrame consisting of the results
        for epoch in epochs :
            for col in df[type_of_optimizer][str(epoch)].columns :
                df[type_of_optimizer][str(epoch)][col] = df[type_of_optimizer][str(epoch)][col].astype(float)


        # Save results to csv files
        base_path_name = os.path.split(os.getcwd())[0] + '/results'
        base_model_name = '/' + model_string


        full_name = base_path_name + base_model_name
        if not(os.path.exists(full_name)) :
            os.mkdir(full_name)

        for epoch in epochs :
            if Opts.scheduler_param :
                base_file_name = '/' + type_of_optimizer + '_schd-' + str(Opts.scheduler_param) + '_type-' + str(Opts.scheduler_type) + '[' + str(Opts.scheduler_step) + '-' + str(Opts.scheduler_coeff) + ']__epochs-' + str(epoch) + '++n_runs-' + str(n_runs)
            else :
                base_file_name = '/' + type_of_optimizer + '_schd-' + str(Opts.scheduler_param) + '_type-' + str(Opts.scheduler_type) + '__epochs-' + str(epoch) + '++n_runs-' + str(n_runs)
            df[type_of_optimizer][str(epoch)].to_csv(full_name + base_file_name + '.csv', index = False)


        # Save best validation results to json
        if Opts.scheduler_param :
            validation_json_filename = model_string + "===best_val_params---" + type_of_optimizer + '_schd-' + str(Opts.scheduler_param) + '_type-' + str(Opts.scheduler_type) + '[' + str(Opts.scheduler_step) + '-' + str(Opts.scheduler_coeff) + ".json"
        else :
            validation_json_filename = model_string + "===best_val_params---" + type_of_optimizer + '_schd-' + str(Opts.scheduler_param) + '_type-' + str(Opts.scheduler_type) + ".json"
        with open(base_path_name + '/best-params/' + validation_json_filename, "w") as outfile :
            json.dump(best_validation_params[type_of_optimizer], outfile)


        # Save best test results to json
        if Opts.scheduler_param :
            test_json_filename = model_string + "===best_test_params---" + type_of_optimizer + '_schd-' + str(Opts.scheduler_param) + '_type-' + str(Opts.scheduler_type) + '[' + str(Opts.scheduler_step) + '-' + str(Opts.scheduler_coeff) + ".json"
        else :
            test_json_filename = model_string + "===best_test_params---" + type_of_optimizer + '_schd-' + str(Opts.scheduler_param) + '_type-' + str(Opts.scheduler_type) + ".json"
        with open(base_path_name + '/best-params/' + test_json_filename, "w") as outfile :
            json.dump(best_test_params[type_of_optimizer], outfile)



        # Generate heatmaps
        if generate_heatmaps_option :
            base_path_heatmap = data_path = os.path.split(os.getcwd())[0] + '/results' + '/plots'

            for current_params in string_params[type_of_optimizer] :
                print('Heatmap parameters : ', current_params, ' for ', type_of_optimizer, '\n')

                if Opts.scheduler_param :
                    base_filename_heatmap = '/' + model_string + "+++++val_params----" + type_of_optimizer + '_schd-' + str(Opts.scheduler_param) + \
                    '_type-' + str(Opts.scheduler_type) + '[' + str(Opts.scheduler_step) + '-' + str(Opts.scheduler_coeff) + '---PARAMS==' + current_params[0] + '_' + current_params[1]
                else :
                    base_filename_heatmap = '/' + model_string + "+++++val_params----" + type_of_optimizer + '_schd-' + str(Opts.scheduler_param) + \
                    '_type-' + str(Opts.scheduler_type) + '---PARAMS==' + current_params[0] + '_' + current_params[1]

                full_name_heatmap_mean = base_path_heatmap + base_filename_heatmap + '____mean'
                full_name_heatmap_std = base_path_heatmap + base_filename_heatmap + '____std'

                create_heatmap(df[type_of_optimizer][str(epochs[-1])], [current_params[0], current_params[1], "mean_validation_accuracy"], full_name_heatmap_mean, TypeStatistic.mean_accuracy)
                create_heatmap(df[type_of_optimizer][str(epochs[-1])], [current_params[0], current_params[1], "std_validation_accuracy"], full_name_heatmap_std, TypeStatistic.std_accuracy)



    # Generate plots
    if generate_plots_option :
        print('START GENERATING PLOTS FOR VALIDATION RESULTS for ' + type_of_optimizer + '... \n')
        generate_plots(best_validation_params, n_runs, dataset_type, base_path_name, 'validation', model_type, model_string)
        
        print('START GENERATING PLOTS FOR TEST RESULTS for' + type_of_optimizer + '... \n')
        generate_plots(best_test_params, n_runs, dataset_type, base_path_name, 'test', model_type, model_string)

