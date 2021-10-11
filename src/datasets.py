################################
#               MODULES
################################
import os, torchvision
import numpy as np

import torch.utils.data as Data
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from utils import TypeDataset











########################################################################
#          FUNCTION USED IN GENERATING PYTORCH DATALOADERS
#########################################################################
def generate_datasets(dataset_type, batch_size, validation_size, shuffle, verbose) :
    
    # Option for downloading the dataset
    download_dataset = True
    
    # The path where we put the donwloaded dataset
    data_path = os.path.split(os.getcwd())[0] + '/data'
    
    # Check if the dataset was downloaded or not
    if not(os.path.exists(data_path)) or not os.listdir(data_path) :
        download_dataset = True
    
    # Consider the mean & std values for the normalization of the datasets
    if dataset_type == TypeDataset.MNIST : 
        mean_vec = (0.1307, )
        std_vec = (0.3081, )
    elif dataset_type == TypeDataset.CIFAR_10 :
        mean_vec = (0.4914, 0.4822, 0.4465)
        std_vec = (0.2023, 0.1994, 0.2010)
    
    stats = (mean_vec, std_vec)

    # Apply PyTorch transformations for train / validation / test datasets
    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)]) 
    validation_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)]) 
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)]) 
    
    # Retrieve the PyTorch datasets
    if dataset_type == TypeDataset.MNIST : 
        torch_data = torchvision.datasets.MNIST
    elif dataset_type == TypeDataset.CIFAR_10 :
        torch_data = torchvision.datasets.CIFAR10
        
    # Download the PyTorch datasets
    train_data = torch_data(root = data_path, train = True, transform = train_transform, download = download_dataset)
    validation_data = torch_data(root = data_path, train = True, transform = validation_transform, download = download_dataset)
    test_data = torch_data(root = data_path, train = False, transform = test_transform)

    # Split the PyTorch train dataset into train / validation
    len_train = len(train_data)
    indices = list(range(len_train))
    split = int(np.floor(validation_size * len_train))

    # Option for shuffling the test dataset
    # In order to be consistent, we should have also set a np.random.seed (which can be a-priori set in the Opts class of utils.py), but different train / validation shuffling is useful for different simulations for variability of the results
    if shuffle == True :
        np.random.shuffle(indices)

    # Indices for splitting into train & validation datasets
    train_index, validation_index = indices[split: ], indices[ :split]

    # Apply the Sampler for train & validation datasets
    train_sampler = SubsetRandomSampler(train_index)
    validation_sampler = SubsetRandomSampler(validation_index)

    # Option used in printing the sizes of the datasets
    if verbose :
        print('Train data: ', len(train_index))
        print('Validation data: ', len(validation_index))
        print('Train data: ', len(test_data))
    
    # Retrieve the PyTorch dataloaders
    train_dataloader = Data.DataLoader(dataset = train_data, batch_size = batch_size, sampler = train_sampler)
    validation_dataloader = Data.DataLoader(dataset = validation_data, batch_size = batch_size, sampler = validation_sampler)
    test_dataloader = Data.DataLoader(dataset = test_data, batch_size = batch_size, shuffle = True)
    dataloaders = {'train': train_dataloader, 'validation': validation_dataloader, 'test': test_dataloader}
    
    # Return the PyTorch dataloaders
    return dataloaders