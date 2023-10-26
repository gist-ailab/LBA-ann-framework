import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from .dinov2.models import vision_transformer


def _make_dinov2_model_name(arch_name: str, patch_size: int) -> str:
    compact_arch_name = arch_name.replace("_", "")[:4]
    return f"dinov2_{compact_arch_name}{patch_size}"

class Dinov2(nn.Module):
    def __init__(self, mode='vit_base'):
        super(Dinov2, self).__init__()
        arch_name = mode
        patch_size = 14
        
        if mode == "vit_base":
            model = vision_transformer.vit_base(patch_size=patch_size, 
                                                img_size=518, init_values=1.0,
                                                block_chunks=0)
            
        _DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
        
        
        model_name = _make_dinov2_model_name(arch_name, patch_size)
        
        url = _DINOV2_BASE_URL + f"/{model_name}/{model_name}_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        
        for p in model.parameters():
            p.requires_grad = False


        self.model = model
        
        self.norm = model.norm
        
     
    def forward(self, x):

        out = self.model.forward_features(x)
        cls_feature = out["x_clstoken"]
    
        return cls_feature
    
    def get_last_selfattention(self, x):
        return self.model.get_last_self_attention(x)
    
    
    def get_last_qkv_feats(self, x, nb_token=1024, feat_dim=768, vit_feat="k"):
        bs = x.shape[0]
        
        nb_token += 1

        feat_h = int(nb_token**0.5)
        feat_w = feat_h
        
        q, k, v = self.model.get_last_qkv(x)

        k = k.transpose(1, 2).reshape(bs, nb_token, -1)
        q = q.transpose(1, 2).reshape(bs, nb_token, -1)
        v = v.transpose(1, 2).reshape(bs, nb_token, -1)

        # Modality selection
        if vit_feat == "k":
            feats = k[:, 1:].transpose(1, 2).reshape(bs, feat_dim, feat_h * feat_w)
        elif vit_feat == "q":
            feats = q[:, 1:].transpose(1, 2).reshape(bs, feat_dim, feat_h * feat_w)
        elif vit_feat == "v":
            feats = v[:, 1:].transpose(1, 2).reshape(bs, feat_dim, feat_h * feat_w)
        elif vit_feat == "kqv":
            k = k[:, 1:].transpose(1, 2).reshape(bs, feat_dim, feat_h * feat_w)
            q = q[:, 1:].transpose(1, 2).reshape(bs, feat_dim, feat_h * feat_w)
            v = v[:, 1:].transpose(1, 2).reshape(bs, feat_dim, feat_h * feat_w)
            feats = torch.cat([k, q, v], dim=1)
            
        feats = F.normalize(feats, p=2, dim=0)
        feats = feats.transpose(1, 2)
        
        return feats
    
    
    def get_patchtokens(self, x):
        out = self.model.forward_features(x)
        patch_feature = out["x_patchtokens"]
        
        return patch_feature


if __name__ == "__main__":
    input_t = torch.rand(size=(1, 3, 224, 224)).cuda()
    model = Dinov2(mode="vit_base").cuda()
    # summary(model, (3, 224, 224))
    out = model(input_t)
    print(out.shape)