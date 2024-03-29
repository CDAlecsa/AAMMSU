<p align="center">
  <b> <h2> PyTorch implementation of AAMMSU optimizer: https://arxiv.org/pdf/2110.08531.pdf  </h2> </b><br>
</p>

## Contents of the *src* (source code) folder:
- *main.py*: this is the Python file used for running an experiment
- *config.json*: contains the parameters of the current experiment
- *utils.py*: contains various functions that are used in the other Python files, along with some settings (epochs, scheduler, gpu, number of simulations)
- *AAMMSU_optim.py*: it is the PyTorch optimizer that we have implemented 
- *nn_modules.py*: contains all the neural network modules: LogisticRegression, CNN, VGG, ResNet
- *datasets.py*: the main function returns the train/validation/test PyTorch dataloaders
- *compile.py*: contains the training & evaluation methods
- *main_comparison_optimizers.py*: contains the comparison of various optimization algorithms (including AAMMSU) for convex and non-convex objective functions
- *various_optimizers*: the folder containing the optimizers used in the aforementioned comparison (excluding AAMMSU which is given in *AAMMSU_optim.py*)

## Experiment guide & remarks:

For an experiment, the options for the scheduler, number of gpus, number of epochs and the number of simulations can be set from the *Opts* class from the file *utils.py*. The other parameters of the experiment can be set in the file *config.json*, which contains a list of 3 elements: the string which represents the type of the dataset (this must be a name from the class *TypeDataset* from *utils.py*), the string which represents the type of model (this must be a name from the class *TypeModel* from *utils.py*) and a dictionary of parameters. The dictionary of parameters contains 2 keys (which are string names from the class *TypeOptimizer* from *utils.py*). For each key, the value is a sub-dictionary which contains the actual value of the parameters of the underlying optimizer (we have only 2 available optimizers: AAMMSU & AMSGrad). When we run an experiment from the file *main.py*, 3 folders are firstly created, namely *data*, *results* and *results/best-params*, in which we will store the chosen dataset, the results of the experiment and the parameters which lead to the best value of accuracy of the optimizers (in the *.json* format). At the same time, in the *main.py* file, we have the options for creating heatmaps for comparing parameters (the setting has the name *generate_heatmaps_option*) and plots for the visual comparison of the 2 optimizers (the setting has the name *generate_plots_option*). After the experiment is done, the comparison plots and the heatmaps (where the color bar represents the mean of the accuracy and the standard deviation of the accuracy) are saved in the folder *results/plots*, while the best validation/test parameters are saved in the folder *results/best-params*. Along with these, a folder with the name of the model string will be created (see the function *generate_model_string* from *utils.py*), in which we will store the values of the train/validation/test accuracy & loss values at different number of epochs (the list of all the epochs for the model is given in the *Opts* class from *utils.py*). We also mention that we did not implemented weight initialization and we have used the default one from PyTorch. As the default VGG and ResNet modules we have chosen VGG-11 and ResNet-18 (you can change this in the class *create_model* from *compile.py*). Furthermore, the LogisticRegression module is implemented only for MNIST dataset (for the general case, one can replace the input and output dimensions from the linear layer), while the CNN, VGG and ResNet modules are implemented for the CIFAR1-0 dataset. We mention that the loss after one epoch is computed as the accumulated losses of the batches divided by total number of batches (these values are stored and are used for plots and heatmaps). On the other hand, the loss in the current batch is simply the loss value for the current batch divided by the current number of processed batches (these values are stored but used only for printing). Finally, the model.eval() command is applied on the test and also on the validation dataset.

## Comparison of optimizers:
We have also compared our AAMMSU optimizer with some adaptive methods. Our folder *various_optimizers* contains the optimization methods implemented in PyTorch which can be found in their original GitHub repositories:
- Adam: https://github.com/pytorch/pytorch/blob/main/torch/optim/adam.py
- Ranger: https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer/blob/master/ranger/ranger2020.py
- Apollo: https://github.com/XuezheMax/apollo/blob/master/optim/apollo.py
- AdaBelief: https://github.com/juntang-zhuang/Adabelief-Optimizer/blob/update_0.2.0/pypi_packages/adabelief_pytorch0.2.1/adabelief_pytorch/AdaBelief.py
- Madgrad: https://github.com/facebookresearch/madgrad/blob/main/madgrad/madgrad.py
The code regarding the comparison of the optimization algorithms can be found in *main_comparison_optimizers.py*. When running the file it creates a folder *optimizer_comparison_results* in which it stores the images containing the comparison of optimizers' trajectories.
Our comparison code is inspired by the implementation:
- https://github.com/jettify/pytorch-optimizer/blob/master/examples/viz_optimizers.py
For a collection containing links to codes & papers with respect to optimizers, see: https://github.com/zoq/Awesome-Optimizer

## References:
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
 
