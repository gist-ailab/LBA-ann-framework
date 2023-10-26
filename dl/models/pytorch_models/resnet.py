import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision import models


class ResNet_teacher(nn.Module):
    def __init__(self, mode):
        super(ResNet_teacher, self).__init__()
        
        if mode == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            
        elif mode == "resnet101":
            model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)

        elif mode == "resnet152":
            model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)

        elif mode == "resnext101_32x8d":
            model = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1)
    
        elif mode == "resnext101_64x4d":
            model = models.resnext101_64x4d(weights=models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1)
   
        
        model_parameter = []
        for name, param in model.named_children():
            model_parameter.append((name, param))
    
        self.block0 = nn.Sequential(OrderedDict(model_parameter[:4]))

        self.block1 = model.layer1
        self.block2 = model.layer2
        self.block3 = model.layer3
        
     
    def forward(self, x):
        
        x = self.block0(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x

class ResNet(nn.Module):
    def __init__(self, mode):
        super(ResNet, self).__init__()
        if mode == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        elif mode == "resnet34":
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        elif mode == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        
        model_parameter = []
        for name, param in model.named_children():
            model_parameter.append((name, param))
        
        self.block0 = nn.Sequential(OrderedDict(model_parameter[:4]))

        self.block1 = model.layer1
        self.block2 = model.layer2
        self.block3 = model.layer3
        self.block4 = model.layer4
        
        
    def forward(self, x):
        
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x4 = x.clone()
        
        x = self.block4(x)
 
        
        return x, x4


if __name__ == "__main__":
    input_t = torch.rand(size=(1,3,224,224)).cuda()
    model = ResNet(mode="resnet50").cuda()
    out = model(input_t)

