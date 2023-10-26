import torch
import torch.nn.functional as F

from models.segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from models.dinov2_model import Dinov2_teacher
from torchvision import transforms


from loader.coco import CocoDataset
from models.label_gen import LabelGenerator


import os

import numpy as np

from PIL import Image

from tqdm import tqdm

from models.clip import clip


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
    
    
    
import matplotlib.pyplot as plt
import matplotlib.patches as patches
    
def show_anns(anns, indice):
    if len(anns) == 0:
        return
    # sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    
    # print(output)
    # print("aaa",len(output), len(anns))
    # exit()
    for idx, ann in enumerate(anns):
        if idx not in indice:
            continue
        m = ann['segmentation']
        # if pred_score < 50:
        #     continue
        
        rand_color = np.random.random(3)
        color_mask = np.concatenate([rand_color, [0.35]])
        
        x,y,w,h = ann['bbox']
        
        img[m] = color_mask
        
        ax.add_patch(patches.Rectangle(xy=(x,y),width=w,height=h, color=rand_color, fill=False))
        
    ax.imshow(img)



def main():
    
    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
    
    root = "/mnt/d/Datasets/coco2017/train2017"
    json_path = "/mnt/d/Datasets/coco2017/annotations/instances_train2017.json"
    
    dataset = CocoDataset(root=root, json=json_path)
    
    random_idx = np.random.randint(dataset.__len__())
    random_idx = 102309
    print(random_idx)
    
    # image_name = "000000056664.jpg"
    
    image, target, _ = dataset.__getitem__(random_idx)
    
    sample_name = "person"
    sample_root = "sample_images"
    
    
    # sample_path_ = os.path.join(sample_root, sample_name, "4.jpg")
    sample_path = os.path.join(sample_root, sample_name, "2.jpg")
    
    dinov2 = Dinov2_teacher().to(device)
    
    # clip_model, preprocess = clip.load("ViT-B/32", device=device)
    
    
    with torch.no_grad():
            
        # u_image = np.array(Image.open(os.path.join(root, image_name)))
        # u_image = np.array(Image.open(sample_path_))  
        u_image = np.array(image)     
        s_image = np.array(Image.open(sample_path))        
        
        model = LabelGenerator(device)
        
        u_crop_images, masks = model.get_crop_img(u_image, u_image)
        s_crop_images, _ = model.get_crop_img(s_image, s_image)
        
        
        
        crop_features = []
        
        for crop_img in tqdm(u_crop_images):    
            image = Image.fromarray(crop_img)
            image = expand2square(image, (0, 0, 0))
            

            image = transform(image).unsqueeze(0).to(device)
            patch_feat = dinov2.get_patchtokens(image)
            crop_feat = patch_feat.mean(1)
            
            # crop_feat = dinov2(image)
            
            crop_feat = crop_feat / crop_feat.norm(dim=-1, keepdim=True)
           
            crop_features.append(crop_feat) 
            
        crop_features = torch.cat(crop_features, dim=0)
        # crop_features = crop_features / crop_features.norm(dim=-1, keepdim=True)
        
        print(crop_features.shape)
        
        
        sample_features = []
        for crop_img in tqdm(s_crop_images):    
            image = Image.fromarray(crop_img)
            image = expand2square(image, (0, 0, 0))
            
            image = transform(image).unsqueeze(0).to(device)
            
            patch_feat = dinov2.get_patchtokens(image)
            sample_feat = patch_feat.mean(1)
            
            # sample_feat = dinov2(image)
            
            sample_feat = sample_feat / sample_feat.norm(dim=-1, keepdim=True)
           
            sample_features.append(sample_feat) 
                
        sample_features = torch.cat(sample_features, dim=0)
        # sample_features = sample_features / sample_features.norm(dim=-1, keepdim=True)
        
        print(sample_features.shape)
        
        cos = (100 * crop_features @ sample_features.T)
        
        cos = cos.T.softmax(dim=-1)
        # print(cos)
        # exit()
        print(cos.shape)
        
        m_val, m_indice = cos.max(dim=1)

        th = 0.9
        temp_m_val = []
        temp_m_indice = []
        
        for val, idx in zip(m_val, m_indice):
            if val >= th:
                temp_m_val.append(val.cpu().item())
                temp_m_indice.append(idx.cpu().item())
                
        m_val, m_indice = temp_m_val, temp_m_indice
        print(m_val)
        # val, indice = torch.topk(m_indice, 5)
        # m_cos = cos.mean(dim=1)
        

        
        # print(m_cos.shape)
        # val, indice = torch.topk(m_cos, 5)
        
        
        # print(np.unique(m_indice, return_counts=True))
        # u_indice = np.unique(m_indice, return_counts=True)[0]
        
        # val, indice = torch.topk(u_indice, 5)
        # print(indice)
        
        plt.figure(figsize=(20,20))
        plt.imshow(u_image)
        show_anns(masks, m_indice)
        plt.axis('off')
        plt.savefig("match.jpg")
        
        
main()
        