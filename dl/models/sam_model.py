import torch
import torch.nn as nn


from models.segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()

    
        # self.device = device
       
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        # sam.to(device=self.device)

        sam_predictor = SamAutomaticMaskGenerator(sam).predictor.model
        
        for p in sam_predictor.parameters():
            p.requires_grad = False
        
        self.img_encoder = sam_predictor.image_encoder
        
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.embed_layer = nn.Linear(256,256)
        
     
    def forward(self, x):

        x = self.img_encoder(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        
        embed_feat = self.embed_layer(x)

        return embed_feat


if __name__ == "__main__":
    input_t = torch.rand(size=(1, 3, 224, 224)).cuda()
    model = SAM(mode="vit_base").cuda()
    # summary(model, (3, 224, 224))
    out = model(input_t)
    print(out.shape)
