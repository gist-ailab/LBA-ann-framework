import torch
import cv2
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from loader.coco import CocoDataset

from torchvision import transforms

from models.dinov2_model import Dinov2_teacher

from PIL import Image

from tqdm import tqdm


transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
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


def main():
    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
    
    
    dinov2 = Dinov2_teacher().to(device)
    
    class_name = "person"
    
    crop_images_root = "coco_select"
    crop_images_path = os.path.join(crop_images_root, class_name)
    
    image_idx = 158
       
    samp_images_root = "sample_images"
    


    
    with torch.no_grad():
        crop_img = Image.open(os.path.join(crop_images_path, str(image_idx)+".jpg")).convert('RGB')
        crop_img = expand2square(crop_img, (0, 0, 0))
        crop_img = transform(crop_img).unsqueeze(0).to(device)
        
        crop_feat = dinov2(crop_img)
        
        # patch_feat = dinov2.get_patchtokens(crop_img)
        # crop_feat = patch_feat.mean(1)
        
        crop_feat = crop_feat / crop_feat.norm(dim=-1, keepdim=True)
        
        
        samp_features = []
        
        class_n = sorted(os.listdir(samp_images_root))
        
        for cls_n in tqdm(class_n):
            
            
            
            # img_n = sorted(os.listdir(os.path.join('coco_select', cls_n)))[24]
            # image = Image.open(os.path.join('coco_select', cls_n, img_n))
            # image = expand2square(image, (0, 0, 0))
            # image = transform(image).unsqueeze(0).to(device)
            
            # samp_feat = dinov2(image)
            
            # # patch_feat = dinov2.get_patchtokens(image)
            # # samp_feat = patch_feat.mean(1)
        

            # samp_feat = samp_feat / samp_feat.norm(dim=-1, keepdim=True)
            # samp_features.append(samp_feat)
            
            cls_features = []
            # for img_n in sorted(os.listdir(os.path.join('coco_select', cls_n)))[:30]:
            for img_n in sorted(os.listdir(os.path.join(samp_images_root, cls_n))):
                image = Image.open(os.path.join(samp_images_root, cls_n, img_n))
                image = expand2square(image, (0, 0, 0))
                image = transform(image).unsqueeze(0).to(device)
                
                samp_feat = dinov2(image)
                
                # patch_feat = dinov2.get_patchtokens(image)
                # samp_feat = patch_feat.mean(1)
            

                samp_feat = samp_feat / samp_feat.norm(dim=-1, keepdim=True)
                cls_features.append(samp_feat)
            
            avg_feature = torch.cat(cls_features).mean(0, keepdim=True)
            # avg_feature = avg_feature / avg_feature.norm(dim=-1, keepdim=True)
   
            samp_features.append(avg_feature)
            
        samp_features = torch.cat(samp_features, dim=0)
        
  
        cos = (100 *crop_feat @ samp_features.T).softmax(dim=1)
        
        output = []
        for idx, sim in enumerate(cos):
            val, cls = torch.topk(sim,10)
            for v, c in zip(val, cls):
                print({"class": class_n[c.item()],
                           "class_idx": c.item(),
                           "score": v.item(),
                           })
        print()
            
if __name__ == "__main__": 
    main()