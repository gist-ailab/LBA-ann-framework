import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os

import numpy as np
from PIL import Image

from pycocotools.coco import COCO
from pycocotools import mask as pymask

import albumentations as A
from albumentations.pytorch import ToTensorV2


def erase_temp_labeled_img(coco, img_indices):
    # erase teeth3 label image
    
    erase_list = []
    erase_list_indice = []
    
    for img_idx in img_indices:
        count = 0
        
        img_meta = coco.imgs[img_idx]
        # print(img_meta)
        
        ann_indices = coco.getAnnIds(img_idx)
        # coco_annotations = self.coco.loadAnns(self.coco.getAnnIds(img_idx))
        # print(coco_annotations)
        
        for ann_id in ann_indices:
            if coco.anns[ann_id]['category_id'] == 1:
                count += 1
    
            # print(self.coco.anns[ann_id]['category_id'])
        
        # print(self.coco.getAnnIds(img_idx))
        # exit()
        if count > 0:
            # print(img_meta['file_name'])
            erase_list.append(img_meta['file_name'])
            erase_list_indice.append(img_idx)
            # print()
        
        
        # print(erase_list)
        # print(len(erase_list))
        
    return erase_list_indice

class CocoDataset(data.Dataset):
    def __init__(self, root, json, train=True):
        self.root = root
        self.coco = COCO(json)
  
        img_indices = list(self.coco.imgs.keys())
        
        print(len(self.coco.dataset['categories']))
        # exit()
        
        erase_list_indice = erase_temp_labeled_img(self.coco, img_indices)
      
        self.img_indices = [img_idx for img_idx in img_indices if img_idx not in erase_list_indice]

        
        only_labels = []
        for img_idx in self.img_indices:
            coco_annotations = self.coco.loadAnns(self.coco.getAnnIds(img_idx))
            
            for idx, a in enumerate(coco_annotations):
                only_labels.append(a['category_id'] - 1)
                
        print(np.unique(only_labels, return_counts=True))

            

        
        self.transform = None
        
        
        
    def seg2mask(self, target, h, w):
        rle = pymask.frPyObjects(target['segmentation'], h, w)
        mask = pymask.decode(rle)
        
        mask = mask.transpose(2, 0, 1)
                     
        return mask


    def __getitem__(self, index):
            
        img_meta = self.coco.imgs[index]
        path = img_meta['file_name']
        # print(os.path.join(self.root, path))
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        
        if self.transform is not None:
            transformed = self.transform(image=np.array(image))
            image = transformed['image']  
            
        coco_annotations = self.coco.loadAnns(self.coco.getAnnIds(index))
 
        boxes = []
        labels = []
        masks = []
        img_id = index
        area = []
        iscrowd = []
        
        for idx, a in enumerate(coco_annotations):
            mask = self.seg2mask(a, img_meta['height'], img_meta['width'])
            masks.append(mask)
            area.append(a["area"])
            boxes.append(a['bbox'])
            labels.append(a['category_id'] - 1)
            iscrowd.append(a["iscrowd"])
            

        if self.transform is not None:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor([labels])
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            img_id = torch.tensor([img_id])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = img_id
        target["area"] = area
        target["iscrowd"] = iscrowd    
        
        return image, target, os.path.join(self.root, path)    
        
      

    def __len__(self):
        return len(self.img_indices)



if __name__ == "__main__":
    s = CocoDataset(root="./panorama_dataset/images", 
                    json="./panorama_dataset/annotations/panorama_coco.json")

    i = 0
    print(s.__getitem__(i)[1])
    print()