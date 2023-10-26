import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision import models


class VGG_teacher(nn.Module):
    def __init__(self, mode):
        super(VGG_teacher, self).__init__()

        if mode == "vgg16":
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        elif mode == "vgg19":
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)


        feature_params = []
        mp_indice = []
        for idx, (name, param) in enumerate(model.features.named_children()):
            feature_params.append(((name), param))
            if "MaxPool2d" in str(param):
                mp_indice.append(idx)

        self.block0 = nn.Sequential(OrderedDict(feature_params[:mp_indice[0]+1]))
        self.block1 = nn.Sequential(OrderedDict(feature_params[mp_indice[0]+1:mp_indice[1]+1]))
        self.block2 = nn.Sequential(OrderedDict(feature_params[mp_indice[1]+1:mp_indice[2]+1]))
        self.block3 = nn.Sequential(OrderedDict(feature_params[mp_indice[2]+1:mp_indice[3]+1]))

    def forward(self, x):
        x = self.block0(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
     
        return x



class VGG(nn.Module):
    def __init__(self, mode, num_classes):
        super(VGG, self).__init__()

        if mode == "vgg11":
            model = models.vgg11(pretrained=True)

        if mode == "vgg13":
            model = models.vgg13(pretrained=True)

        elif mode == "vgg16":
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        elif mode == "vgg19":
            model = models.vgg19(pretrained=True)

        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)

        feature_params = []
        mp_indice = []
        for idx, (name, param) in enumerate(model.features.named_children()):
            feature_params.append(((name), param))
            if "MaxPool2d" in str(param):
                mp_indice.append(idx)

        self.block0 = nn.Sequential(OrderedDict(feature_params[:mp_indice[0]+1]))
        self.block1 = nn.Sequential(OrderedDict(feature_params[mp_indice[0]+1:mp_indice[1]+1]))
        self.block2 = nn.Sequential(OrderedDict(feature_params[mp_indice[1]+1:mp_indice[2]+1]))
        self.block3 = nn.Sequential(OrderedDict(feature_params[mp_indice[2]+1:mp_indice[3]+1]))
        self.block4 = nn.Sequential(OrderedDict(feature_params[mp_indice[3]+1:]))

  
    def forward(self, x):
        x = self.block0(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x3 = x.clone()
        
        x = self.block4(x)


        return x, x3

if __name__ == "__main__":
    input_t = torch.rand(size=(1,3,224,224)).cuda()
    # model = VGG(mode="vgg16", num_classes=200).cuda()
    model = VGG_teacher(mode="vgg16").cuda()
    out = model(input_t)
    print(out.shape)
