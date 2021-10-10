# AAMMSU optimizer (from the paper : ....)

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

