# PyTorch implementation of AAMMSU optimizer - from the paper : ....

## Neural networks architectures that are available:
- Logistic Regression: <br />
https://towardsdatascience.com/logistic-regression-on-mnist-with-pytorch-b048327f8d19 <br /> https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_logistic_regression/ <br />
- CNN:
https://shonit2096.medium.com/cnn-on-cifar10-data-set-using-pytorch-34be87e09844 <br /> 
https://zhenye-na.github.io/2018/09/28/pytorch-cnn-cifar10.html
- VGG: https://github.com/uclaml/Padam/blob/master/models/vgg.py
- ResNet: https://github.com/uclaml/Padam/blob/master/models/resnet.py

## Datasets:
- MNIST: http://yann.lecun.com/exdb/mnist/
- CIFAR10: https://www.cs.toronto.edu/~kriz/cifar.html

## References which helped the current implementation of the optimizer:
https://github.com/pytorch/pytorch/tree/99711133403eff8474af0e710a45d367f4fb5e66/torch/optim <br />
https://github.com/jettify/pytorch-optimizer/tree/master/torch_optimizer

## For the train/validation/test split, we have used:
https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb#file-data_loader-py

## For the dataset pre-processing, see:
- normalization on MNIST: https://github.com/floydhub/mnist/blob/master/main.py
- normalization on CIFAR-10: https://zhenye-na.github.io/2018/09/28/pytorch-cnn-cifar10.html

## How to make experiments with the AAMMSU optimizer:
The config.json file must contain the following: 
- a string for the dataset type (the name must belong to a string variable from the class TypeDataset class in the utils.py)
- a string for the type of model (the name must belong to a string variable from the class TypeModel class in the utils.py)
- a dictionary which contain two sub-dictionaries of parameters (one for the AAMMSU optimizer and one for the AMSGRAD optimizer)
- each sub-dictionary contains, for each key (e.g. lr, eps, beta_2, ...) the list of the values we want in our experiments

For our experiments we have 2 plotting options: make actual plots and generate heatmaps for the parameters (see compile.py file) <br />
The number of runs for the simulations, the number of epochs and the scheduler options are given in the class Opts from utils.py <br />
The default VGG and ResNet modules are VGG-11 and ResNet-18 (you can change this in the class create_model from compile.py) <br />
The LogisticRegression module is implemented only for MNIST dataset (for the general case, one can replace the input and output dimensions from the linear layer)

## Remarks:
- The loss after one epoch is computed (as usually) as the accumulated losses of the batches divided by total number of batches (these values are stored and are used for plots and heatmaps) <br />
- The loss in the current batch is simply the loss value for the current batch divided by the current number of processed batches (these values are stored but used only for printing) <br />
- The model.eval() is applied on the test and also on the validation dataset
- The weight initialization is the default one from PyTorch

## nn implementations as references:
- https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
- https://github.com/jamespengcheng/PyTorch-CNN-on-CIFAR10/blob/master/ConvNetClassifier.py
- https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py


## Experiments :
- LogisticRegression only for MNIST
- CNN, VGG and ResNet only for CIFAR_10
