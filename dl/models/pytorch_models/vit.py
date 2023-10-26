import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision import models
import torch.nn.functional as F

from src.utils import minmax_norm

def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias: 
            torch.nn.init.xavier_uniform_(m.bias)


class ViT(nn.Module):
    def __init__(self, mode, num_classes):
        super(ViT, self).__init__()
        if mode == "vit_l_16":
            model = models.vit_l_16(weights=models.ViT_L_16_Weights.DEFAULT)

        elif mode == "vit_l_32":
            model = models.vit_l_32(weights=models.ViT_L_32_Weights.DEFAULT)
        
        elif mode == "vit_h_14":
            model = models.vit_l_32(weights=models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1)

        self._process_input = model._process_input
        self.class_token = model.class_token
        self.conv_proj = model.conv_proj
        self.encoder = model.encoder

        num_ftrs = model.hidden_dim
        self.heads = nn.Linear(num_ftrs, num_classes)
     
    def forward(self, x):
        x = self._process_input(x)
        n = x.shape[0]

        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        x = x[:, 0]

        out = self.heads(x)

        return out


class ViT_teacher(nn.Module):
    def __init__(self, mode):
        super(ViT_teacher, self).__init__()
        if mode == "vit_l_16":
            model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)

        elif mode == "vit_l_32":
            model = models.vit_l_32(weights=models.ViT_L_32_Weights.IMAGENET1K_V1)
        
        elif mode == "vit_h_14":
            model = models.vit_h_14(weights=models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1)

        self.num_ftrs = model.hidden_dim

        self._process_input = model._process_input
        self.class_token = model.class_token
        self.conv_proj = model.conv_proj
        self.encoder = model.encoder
     
    def forward(self, x):
        x = self._process_input(x)
        # print(x.shape)
        n = x.shape[0]

        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)
        
        # x = x.permute(0, 2, 1)[:,:,1:]
   
        # out = x.view(-1, self.num_ftrs, 7, 7)

        # out = x[:, 1 :  , :].reshape(x.size(0), 7, 7, x.size(2))
        # out = x[:, 1 :  , :].reshape(x.size(0), 14, 14, x.size(2))
        out = x[:, 1 :  , :].reshape(x.size(0), 16, 16, x.size(2))
        # print(out.shape)

        out = out.transpose(2, 3).transpose(1, 2)


        return out

if __name__ == "__main__":
    input_t = torch.rand(size=(1,3,224,224)).cuda()
    model = ViT(mode="vit_l_32", num_classes=200).cuda()
    # summary(model, (3, 224, 224))
    out = model(input_t)
    print(out.shape)
