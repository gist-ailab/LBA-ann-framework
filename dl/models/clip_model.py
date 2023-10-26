import torch
import torch.nn as nn
from models.clip import clip


class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()

        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
     
    def forward(self, x):
        
        x = self.preprocess(x).to(self.device)
        
        if x.dim == 4:
            pass
        else:
            x = x.unsqueeze(0)
            
        feat = self.clip_model.encode_image(x)
    

        return feat


if __name__ == "__main__":
    input_t = torch.rand(size=(1, 3, 224, 224)).cuda()
    model = CLIP(mode="vit_base").cuda()
    # summary(model, (3, 224, 224))
    out = model(input_t)
    print(out.shape)