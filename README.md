# Contents of the *src* (source code) folder:
- *main.py*: this is the Python file used for running an experiment
- *config.json*: contains the parameters of the current experiment
- *utils.py*: contains various functions that are used in the other Python files, along with some settings (epochs, scheduler, gpu, number of simulations)
- *AAMMSU_optim.py*: it is the PyTorch optimizer that we have implemented 
- *nn_modules.py*: contains all the neural network modules: LogisticRegression, CNN, VGG, ResNet
- *datasets.py*: the main function returns the train/validation/test PyTorch dataloaders
- *compile.py*: contains the training & evaluation methods

# Experiment guide & remarks:

For an experiment, the options for the scheduler, number of gpus, number of epochs and the number of simulations can be set from the *Opts* class from the file *utils.py*. The other parameters of the experiment can be set in the file *config.json*, which contains a list of 3 elements: the string which represents the type of the dataset (this must be a name from the class *TypeDataset* from *utils.py*), the string which represents the type of model (this must be a name from the class *TypeModel* from *utils.py*) and a dictionary of parameters. The dictionary of parameters contains 2 keys (which are string names from the class *TypeOptimizer* from *utils.py*). For each key, the value is a sub-dictionary which contains the actual value of the parameters of the underlying optimizer (we have only 2 available optimizers: AAMMSU & AMSGrad). When we run an experiment from the file *main.py*, 3 folders are firstly created, namely *data*, *results* and *results/best-params*, in which we will store the chosen dataset, the results of the experiment and the parameters which lead to the best value of accuracy of the optimizers (in the *.json* format). At the same time, in the *main.py* file, we have the options for creating heatmaps for comparing parameters (the setting has the name *generate_heatmaps_option*) and plots for the visual comparison of the 2 optimizers (the setting has the name *generate_plots_option*). After the experiment is done, the comparison plots and the heatmaps (where the color bar represents the mean of the accuracy and the standard deviation of the accuracy) are saved in the folder *results/plots*, while the best validation/test parameters are saved in the folder *results/best-params*. Along with these, the resu

# References:
- Datasets:
  - MNIST: http://yann.lecun.com/exdb/mnist/
  - CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
- Data pre-processing:
  - normalization on MNIST: https://github.com/floydhub/mnist/blob/master/main.py
  - normalization on CIFAR-10: https://zhenye-na.github.io/2018/09/28/pytorch-cnn-cifar10.html
  - train/validation/test split: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb#file-data_loader-py
- Available networks:
  - Logistic Regression: https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_logistic_regression/
  - CNN: https://zhenye-na.github.io/2018/09/28/pytorch-cnn-cifar10.html
  - VGG: https://github.com/uclaml/Padam/blob/master/models/vgg.py
  - ResNet: https://github.com/uclaml/Padam/blob/master/models/resnet.py
- Basic PyTorch neural network implementations:
  - Neural networks on CIFAR-10: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
  - ConvNet classifier: https://github.com/jamespengcheng/PyTorch-CNN-on-CIFAR10/blob/master/ConvNetClassifier.py
  - FeedForward neural network: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py
- Optimizer implementations in PyTorch:
  - Adam & AMSGrad: https://github.com/pytorch/pytorch/tree/99711133403eff8474af0e710a45d367f4fb5e66/torch/optim
  - Adaptive optimizers: https://github.com/jettify/pytorch-optimizer/tree/master/torch_optimizer
 
