import torch.nn as nn

from models.maskrcnn.mask_rcnn import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights


class InstFrame(nn.Module):
    def __init__(self, num_classes, args):
        super(InstFrame, self).__init__()
        
        # MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn(weights=None,
                                  weights_backbone= "ResNet50_Weights.IMAGENET1K_V1", 
                                  num_classes=num_classes).to(args.device)
        
    def forward(self, x, target=None):
        
        if target==None:
            output = self.model(x)
            
        else:
            # print(target)
            output = self.model(x, target)
        
        return output