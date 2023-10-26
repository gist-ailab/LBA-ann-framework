import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import inception, _utils
from typing import Any, Optional


@_utils.handle_legacy_interface(weights=("pretrained", inception.Inception_V3_Weights.IMAGENET1K_V1))
def inception_v3(*, weights: Optional[inception.Inception_V3_Weights] = None, progress: bool = True, **kwargs: Any) -> inception.Inception3:
    weights = inception.Inception_V3_Weights.verify(weights)
    
    
    if weights is not None:
        _utils._ovewrite_named_param(kwargs, "init_weights", False)
        _utils._ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        
    model = inception.Inception3(**kwargs)
        
    model.load_state_dict(weights.get_state_dict(progress=progress), strict=False)
    model.aux_logits = False
    model.AuxLogits = None
        
    return model


class Inception3_teacher(nn.Module):
    def __init__(self, num_classes):
        super(Inception3_teacher, self).__init__()

        model = inception_v3(aux_logits=False, pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,num_classes)


        C_indice = [3,6,10,15,18]

        feature_params = []
        for name, param in model.named_children():
            if name == "avgpool":
                break
            feature_params.append(((name), param))

        self.block0 = nn.Sequential(OrderedDict(feature_params[:C_indice[0]]))

        self.block1 = nn.Sequential(OrderedDict(feature_params[C_indice[0]:C_indice[1]]))
        self.block2 = nn.Sequential(OrderedDict(feature_params[C_indice[1]:C_indice[2]]))
        self.block3 = nn.Sequential(OrderedDict(feature_params[C_indice[2]:C_indice[3]]))
        self.block4 = nn.Sequential(OrderedDict(feature_params[C_indice[3]:C_indice[4]]))

     
    def forward(self, x):
        
        x = self.block0(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x


class Inception3(nn.Module):
    def __init__(self, num_classes):
        super(Inception3, self).__init__()

        model = inception_v3(pretrained=True)

        C_indice = [3,6,10,15,18]

        feature_params = []
        for name, param in model.named_children():
            if name == "avgpool":
                break
            feature_params.append(((name), param))

        self.block0 = nn.Sequential(OrderedDict(feature_params[:C_indice[0]]))

        self.block1 = nn.Sequential(OrderedDict(feature_params[C_indice[0]:C_indice[1]]))
        self.block2 = nn.Sequential(OrderedDict(feature_params[C_indice[1]:C_indice[2]]))
        self.block3 = nn.Sequential(OrderedDict(feature_params[C_indice[2]:C_indice[3]]))
        self.block4 = nn.Sequential(OrderedDict(feature_params[C_indice[3]:C_indice[4]]))


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
    model = Inception3(num_classes=200).cuda()
    out = model(input_t)
    print(out.shape)
