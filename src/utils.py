###########################
#          MODULES
###########################
import os, torch
import numpy as np
import matplotlib.pyplot as plt 









############################################
#               MISCELLANEOUS CLASSES
###########################################

# Choices for the datasets
class TypeDataset :
    MNIST = 'MNIST'
    CIFAR_10 = 'CIFAR_10'




# Choices for the optimizers
class TypeOptimizer :
    AMSGRAD = 'AMSGRAD'
    AAMMSU = 'AAMMSU'




# Types of VGG networks
class TypeVGG :
    VGG_11 = 'VGG11'
    VGG_13 = 'VGG13'
    VGG_16 = 'VGG16'
    VGG_19 = 'VGG19'

    

    
# Types of ResNet networks
class TypeResNet :
    ResNet_18 = 'ResNet_18'
    ResNet_34 = 'ResNet_34'
    ResNet_50 = 'ResNet_50'
    ResNet_101 = 'ResNet_101'
    ResNet_152 = 'ResNet_152'


    
# Types of nn models
class TypeModel :
    LR = 'LR'
    CNN = 'CNN'
    VGG = 'VGG'
    ResNet = 'ResNet'

    
# Types of statistics for computations (mean & std metrics)
class TypeStatistic :
    mean_accuracy = 'mean_accuracy'
    std_accuracy = 'std_accuracy'
    
    
# Type of scheduler for the optimizer
class Scheduler_type :
    step_lr = 'SLR'
    multi_step_lr = 'MLR'




# Option class for the settings of the experiments
class Opts :
    ngpu = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random_seed = False
    verbose = True
    split_ratio = 0.2
    epochs = [50, 75, 100, 150, 175, 200]
    n_runs = 5

    scheduler_param = True
    scheduler_coeff = 1e-1
    scheduler_type = Scheduler_type.multi_step_lr
    
    if scheduler_type == Scheduler_type.step_lr :
        scheduler_step = 30 * epochs[-1] / 100
    elif scheduler_type == Scheduler_type.multi_step_lr :
        scheduler_step = [50, 100, 150]









######################################
#          FUNCTION USED FOR PLOTS
######################################
def plot_values(x_axis_list, title_string, legend_strings, lists_mean, lists_std, save_path, n_runs) :
    x_axis_AAMMSU, x_axis_AMSGRAD = x_axis_list
    
    plt.figure(figsize = (10, 10))

    # Train & validation plots for the mean values of AAMMSU
    plt.plot(x_axis_AAMMSU, lists_mean[0][TypeOptimizer.AAMMSU], 'ro-')
    plt.plot(x_axis_AAMMSU, lists_mean[1][TypeOptimizer.AAMMSU], 'bs-')
    
    # Train & validation plots for the mean values of AMSGRAD
    plt.plot(x_axis_AMSGRAD, lists_mean[0][TypeOptimizer.AMSGRAD], 'ko-')
    plt.plot(x_axis_AMSGRAD, lists_mean[1][TypeOptimizer.AMSGRAD], 'gs-')

    # Train & validation plots for the error bands (mean - std, mean + std) of AAMMSU
    plt.fill_between(x_axis_AAMMSU, lists_mean[0][TypeOptimizer.AAMMSU] - lists_std[0][TypeOptimizer.AAMMSU], lists_mean[0][TypeOptimizer.AAMMSU] + lists_std[0][TypeOptimizer.AAMMSU], alpha = 0.2, color = 'red')
    plt.fill_between(x_axis_AAMMSU, lists_mean[1][TypeOptimizer.AAMMSU] - lists_std[1][TypeOptimizer.AAMMSU], lists_mean[1][TypeOptimizer.AAMMSU] + lists_std[1][TypeOptimizer.AAMMSU], alpha = 0.2, color = 'blue')

    # Train & validation plots for the error bands (mean - std, mean + std) of AMSGRAD
    plt.fill_between(x_axis_AMSGRAD, lists_mean[0][TypeOptimizer.AMSGRAD] - lists_std[0][TypeOptimizer.AMSGRAD], lists_mean[0][TypeOptimizer.AMSGRAD] + lists_std[0][TypeOptimizer.AMSGRAD], alpha = 0.2, color = 'black')
    plt.fill_between(x_axis_AMSGRAD, lists_mean[1][TypeOptimizer.AMSGRAD] - lists_std[1][TypeOptimizer.AMSGRAD], lists_mean[1][TypeOptimizer.AMSGRAD] + lists_std[1][TypeOptimizer.AMSGRAD], alpha = 0.2, color = 'green')

    plt.legend(legend_strings)
    plt.title(title_string)
    
    plt.savefig(save_path + '.jpeg')
    plt.savefig(save_path + '.eps')
    plt.clf()






    


##############################################################################
#          FUNCTION USED FOR GENERATING ORDERED PAIRS OF (mu, tilde_gamma)
##############################################################################
def generate_pairs(mu_lst, tilde_gamma_list) :
    pairs_list = []
    if len(mu_lst) == len(tilde_gamma_list) :
        for count, mu in enumerate(mu_lst) :
            if mu_lst[count] <= tilde_gamma_list[count] : 
                pairs_list.append((mu_lst[count], tilde_gamma_list[count]))
        return pairs_list
    else :
        return None







###########################################################################
#          FUNCTION USED FOR GENERATING TUPLES OF COEFFICIENTS
###########################################################################
def generate_tuples_from_dict(dict_of_lists, amsgrad = False, generate_pairs_option = True) :

    lr_lst = dict_of_lists["lr"]
    batch_size_lst = dict_of_lists["batch_size"] 
    eps_lst = dict_of_lists["eps"]
    beta_2_lst = dict_of_lists["beta_2"]
 
    tuple_list = []
    
    if amsgrad :
        for lr in lr_lst :
            for batch_size in batch_size_lst :
                for eps in eps_lst :
                    for beta_2 in beta_2_lst :
                        tuple_list.append((lr, batch_size, eps, beta_2))

    else :
        mu_lst = dict_of_lists["mu"]
        tilde_gamma_lst = dict_of_lists["tilde_gamma"]
        M_lst = dict_of_lists["M"] 
        nu_lst = dict_of_lists["nu"]
        
        if generate_pairs_option :
            pairs_list = generate_pairs(mu_lst, tilde_gamma_lst)

        for lr in lr_lst :
            for batch_size in batch_size_lst :
                for nu in nu_lst :
                    for eps in eps_lst :
                        for beta_2 in beta_2_lst :
                            for M in M_lst :
                                
                                if generate_pairs_option :
                                    for (mu, tilde_gamma) in pairs_list :
                                        tuple_list.append((lr, batch_size, M, nu, eps, beta_2, mu, tilde_gamma))

                                else :
                                    for mu in mu_lst :    
                                        for tilde_gamma in tilde_gamma_lst :
                                            if mu <= tilde_gamma :
                                                tuple_list.append((lr, batch_size, M, nu, eps, beta_2, mu, tilde_gamma))

    return tuple_list










##############################################################################
#          FUNCTION USED FOR GENERATING THE NAME OF THE MODEL STRING
##############################################################################

def generate_model_string(dataset_type, type_of_model) :
    return type_of_model.lower() + '-' + dataset_type.lower() 
