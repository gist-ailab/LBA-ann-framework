import torch
import torch.nn as nn

import os
import cv2
import natsort
from PIL import Image


from models.clip import clip
from models.segment_anything import sam_model_registry, SamPredictor


from models.dinov2_model import Dinov2_teacher

import PIL
from src.third_party.TokenCut.unsupervised_saliency_detection import utils
from torchvision import transforms


from tqdm import tqdm

import numpy as np




resize = (448, 448) 

transform = transforms.Compose([
                    transforms.Resize(size=(int(resize[0]), int(resize[1]))),
                    transforms.ToTensor(),
                    transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225)),])


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result



class LabelGenerator(nn.Module):
    def __init__(self, device) -> None:
        super(LabelGenerator, self).__init__()
        
        self.device = device
       
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        
        
        self.predictor = SamPredictor(sam)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        
        
        
    @torch.no_grad()    
    def get_mask(self, image, in_points):
        
        
        # in_points = torch.tensor(in_points)
        # in_labels = in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        
        in_points = np.array(in_points)
        in_labels = np.array([1 for i in range(len(in_points))])
        
        # print(in_points)
        # print(in_labels)
        
        # exit()
        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
                                point_coords=in_points,
                                point_labels=in_labels,
                                multimask_output=True,)
        
        return masks
    
   
        
    
    

if __name__ == "__main__":
    input_t = torch.rand(size=(1,3,224,224)).cuda()
    model = LabelGenerator().cuda()