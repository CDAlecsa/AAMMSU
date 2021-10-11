################################
#               MODULES
################################
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from utils import TypeVGG, TypeResNet











################################
#           LR MODULE
################################
class LR(nn.Module) :

    # Constructor
    def __init__(self) :
        super(LR, self).__init__()
        self.linear = nn.Linear(784, 10)

    # Forward pass
    def forward(self, x) :
        outputs = self.linear(x)
        return outputs











################################
#           CNN MODULE
################################
class CNN(nn.Module) :

    # Constructor
    def __init__(self) :
        super(CNN, self).__init__()

        # Container for convolutional layers
        self.conv_layer = nn.Sequential(

            # Convolutional layer block 1
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # Convolutional layer block 2
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout2d(p = 0.05),

            # Convolutional layer block 3
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )


        # Container for linear layers
        self.fc_layer = nn.Sequential(
            nn.Dropout(p = 0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace = True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace = True),
            nn.Dropout(p = 0.1),
            nn.Linear(512, 10)
        )


    # Forward pass
    def forward(self, x) :
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x









############################
#           VGG MODULE
############################
cfg = {
    TypeVGG.VGG_11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    TypeVGG.VGG_13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    TypeVGG.VGG_16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    TypeVGG.VGG_19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



class VGG(nn.Module) :

    # Constructor
    def __init__(self, vgg_name, num_classes = 10) :
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)


    # Forward pass
    def forward(self, x) :
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


    # Define the VGG layers
    def _make_layers(self, cfg) :
        layers = []
        in_channels = 3
        for x in cfg :
            if x == 'M' :
                layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
            else :
                layers += [nn.Conv2d(in_channels, x, kernel_size = 3, padding = 1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace = True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size = 1, stride = 1)]
        return nn.Sequential(*layers)
    
    
    



    
    



#####################################
#           ResNet MODULE
#####################################


# Basic block class
class BasicBlock(nn.Module) :
    expansion = 1

    def __init__(self, in_planes, planes, stride = 1) :

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes :
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(self.expansion * planes)
            )


    # Forward pass
    def forward(self, x) :
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



# Bottleneck class
class Bottleneck(nn.Module) :
    expansion = 4

    def __init__(self, in_planes, planes, stride = 1) :
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes :
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    # Forward pass
    def forward(self, x) :
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# Actual ResNet class
class ResNet(nn.Module) :

    # Constructor
    def __init__(self, block, num_blocks, num_classes = 10) :
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride = 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride = 2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)


    # Define the ResNet layers
    def _make_layer(self, block, planes, num_blocks, stride) :
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides :
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    # Forward pass
    def forward(self, x) :
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out




# Different types of ResNet architectures
def ResNet18(num_classes = 10) :
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes = num_classes)

def ResNet34(num_classes = 10) :
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes = num_classes)

def ResNet50(num_classes = 10) :
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes = num_classes)

def ResNet101(num_classes = 10) :
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes = num_classes)

def ResNet152(num_classes = 10) :
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes = num_classes)


