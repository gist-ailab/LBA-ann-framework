import torch
import torch.nn.functional as F

import cv2
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from loader.coco import CocoDataset

from torchvision import transforms
from models.label_gen_pointed import LabelGenerator

from PIL import Image
from models.dinov2_model import Dinov2_teacher

import skimage


def show_anns(masks, in_points):

    # sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((masks[0].shape[0], masks[0].shape[1], 4))
    img[:,:,3] = 0
    
    # print(output)
    # print("aaa",len(output), len(anns))
    # exit()
    
    
    marker_size=375
    in_points = np.array(in_points)
    ax.scatter(in_points[:, 0], in_points[:, 1], color='green', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)
    
    for idx, mask in enumerate(masks):
                
        rand_color = np.random.random(3)
        color_mask = np.concatenate([rand_color, [0.35]])
        
        img[mask] = color_mask
        
    ax.imshow(img)


def minmax_norm(A):
    batch_size, _,  height, width = A.shape
    AA = A.clone()
    AA = AA.view(A.size(0), -1)
    AA -= AA.min(1, keepdim=True)[0]
    AA /= AA.max(1, keepdim=True)[0]
    AA = AA.view(batch_size, 1, height, width)

    return AA


def get_points(attentions, image, w, h):
    
    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    attentions[:, 283] = 0 

    patch_size = 14
    w_featmap = image.shape[-2] // patch_size
    h_featmap = image.shape[-1] // patch_size
    
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = F.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0]
    
    att_map = torch.mean(attentions.unsqueeze(0), dim=1, keepdim=True)
    
    att_map = F.sigmoid(att_map)
    att_map = minmax_norm(att_map)
    att_map = att_map.swapaxes(2,3)
    
    th = 0.5
    att_map[att_map<th] = 0
    att_map[att_map>th] = 1
    
    
    att_map = F.interpolate(att_map, size=(w, h), mode="bilinear", align_corners=True)

    lbl_0 = skimage.measure.label(att_map[0][0].detach().cpu().numpy()) 
    props = skimage.measure.regionprops(lbl_0)
    
    c_points = []
    for prop in props:
        cx, cy = prop.centroid
        c_points.append([cx,cy])
        
    return c_points
    

resize = (448, 448) 
transform = transforms.Compose([
                    transforms.Resize(size=(int(resize[0]), int(resize[1]))),
                    transforms.ToTensor(),
                    transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225)),])



def main():
    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
    
    root = "/mnt/d/Datasets/coco2017/train2017"
    json_path = "/mnt/d/Datasets/coco2017/annotations/instances_train2017.json"
    
    dataset = CocoDataset(root=root, json=json_path)
    
    
    model = LabelGenerator(device)
    dino_vit = Dinov2_teacher().to(device)
    
    
    
    random_idx = np.random.randint(dataset.__len__())
    # random_idx = 323268
    print(random_idx)

    image, target, _ = dataset.__getitem__(random_idx)
    w,h = image.size
    # image_name = "000000496575.jpg"
        
    with torch.no_grad():
        # image = np.array(Image.open(os.path.join(root, image_name)))
        
        image_t = transform(image).to(device)
        image_t = image_t.unsqueeze(0)

        attentions = dino_vit.get_last_selfattention(image_t)
        in_points = get_points(attentions, image_t, w, h)
        
        
        image = np.array(image)
        masks = model.get_mask(image, in_points)
        
        print(masks.shape)
        
        # exit()

   
        plt.figure(figsize=(20,20))
        plt.imshow(image)
        show_anns(masks, in_points)
        plt.axis('off')
        plt.savefig("vis_point.jpg")
        
        # model.get_sim()
    
    
if __name__ == "__main__": 
    main()