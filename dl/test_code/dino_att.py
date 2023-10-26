import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import cv2
import numpy as np


from models.pytorch_models.resnet import ResNet
from loader.coco import CocoDataset

from models.dinov2_model import Dinov2_teacher

import torchvision

from PIL import Image

import skimage

def att_visualization(att_map_copy, path, c_points):
    
    with torch.no_grad():

        ori_np_img = cv2.imread(path)
        # ori_np_img =  np.transpose(ori_np_img, (1, 0, 2))
        
        ori_np_img_copy = ori_np_img.copy()
        
        att_map_copy = att_map_copy.cpu().detach().numpy()
        att_map_copy = att_map_copy * 255
        
        att_map_copy =  np.transpose(att_map_copy, (2, 1, 0))
        
        # print(att_map_copy.shape)
        # exit()

        heatmap_img = cv2.applyColorMap(att_map_copy.astype(np.uint8), cv2.COLORMAP_JET)
        
        for c_p in c_points:
            x, y = c_p
            print(x,y)
            cv2.circle(heatmap_img, (int(x), int(y)), 1, [0,0,255], 1)
        
        # print(heatmap_img.shape)
        # print(ori_np_img_copy.shape)
        # exit()
        
        np_img = cv2.addWeighted(heatmap_img, 0.5, ori_np_img_copy.astype(np.uint8), 0.5, 0)
          
        addh = cv2.hconcat([ori_np_img.astype(np.uint8), np_img])
        
        
        

        cv2.imwrite('compare_img.png', addh)



ToTensor = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(
                                (0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),])


resize = (448, 448) 
transform = transforms.Compose([
                    transforms.Resize(size=(int(resize[0]), int(resize[1]))),
                    transforms.ToTensor(),
                    transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225)),])



def minmax_norm(A):
    batch_size, _,  height, width = A.shape
    AA = A.clone()
    AA = AA.view(A.size(0), -1)
    AA -= AA.min(1, keepdim=True)[0]
    AA /= AA.max(1, keepdim=True)[0]
    AA = AA.view(batch_size, 1, height, width)

    return AA

def main():
    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
    
    root = "/mnt/d/Datasets/coco2017/train2017"
    json_path = "/mnt/d/Datasets/coco2017/annotations/instances_train2017.json"
    
    dataset = CocoDataset(root=root, json=json_path)
    
    # dataset_idx = 9979
    
    random_idx = np.random.randint(dataset.__len__())
    # random_idx = 166152
    print(random_idx)
    image, target, path = dataset.__getitem__(random_idx)
    
    # ToTensor = transforms.Compose([transforms.ToTensor(),])
    
    ToTensor = transform
    
    # model = ResNet("resnet50", )
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT", 
    #                                                            progress=True, 
    #                                                            num_classes=91)
    
    
    model = Dinov2_teacher()
    
    
    model = model.to(device)
    
    
    
    # path = "/home/jwk9284/Workspace/issl/sample_images_raw/surfboard/0.jpg"
    # image = Image.open(path).convert('RGB')
    w,h = image.size
    
    with torch.no_grad():        
        image = ToTensor(image).to(device)
        image = image.unsqueeze(0)
        
        
        attentions = model.get_last_selfattention(image)
        print(attentions.shape)
        
        
        # exit()
        
        nh = attentions.shape[1] # number of head

        # we keep only the output patch attention
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
        attentions[:, 283] = 0 

        patch_size = 14
        w_featmap = image.shape[-2] // patch_size
        h_featmap = image.shape[-1] // patch_size
        
    
        
        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        
        
        attentions = F.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0]
        print(attentions.shape)
        # exit()
        
        # att_map = attentions[-1].reshape(-1, 1, 448, 448)
        att_map = torch.mean(attentions.unsqueeze(0), dim=1, keepdim=True)
        print(att_map.shape)
        
        print()
        
        # print(att_map.shape)
        # exit()
        
        att_map = F.sigmoid(att_map)
        att_map = minmax_norm(att_map)
        att_map = att_map.swapaxes(2,3)
        
        th = 0.5
        att_map[att_map<th] = 0
        att_map[att_map>th] = 1
        
        
        
        att_map = F.interpolate(att_map, size=(w, h), mode="bilinear", align_corners=True)
        
        print(att_map.shape)
        
        lbl_0 = skimage.measure.label(att_map[0][0].detach().cpu().numpy()) 
        props = skimage.measure.regionprops(lbl_0)
        
        c_points = []
        for prop in props:
            cx, cy = prop.centroid
            c_points.append([cx,cy])
        
        att_map = att_map[0]

        att_visualization(att_map, path, c_points)
        
        


if __name__ == "__main__":
    main()
