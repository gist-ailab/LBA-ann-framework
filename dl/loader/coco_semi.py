import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os

import numpy as np
from PIL import Image

from pycocotools.coco import COCO
from pycocotools import mask as pymask

from loader import presets
from loader import transforms as T

def collate_fn(samples):
    
    images = []
    targets = []

    for image, target in samples:

        images.append(image)
        targets.append(target)
   

    return images, targets


class CocoDataset(data.Dataset):
    def __init__(self, root, json, rand_per=0.1, train=True):
        self.root = root
        self.coco = COCO(json)
  
        img_indices = list(self.coco.imgs.keys())
         
        labeled_count = int(len(img_indices)*rand_per)
        rand_choice = np.random.choice(len(img_indices), labeled_count, replace=False)
        
        img_indices = img_indices[rand_choice]
        
        
        if train:
            filter_list = []
            for img_idx in img_indices:
                coco_annotations = self.coco.loadAnns(self.coco.getAnnIds(img_idx))
    
                if len(coco_annotations) == 0:
                    filter_list.append(img_idx)
                    continue
                
            
            for img_idx in img_indices:
                coco_annotations = self.coco.loadAnns(self.coco.getAnnIds(img_idx))
                for idx, a in enumerate(coco_annotations):
                    
                    bbox = a['bbox']
                    bbox = np.array(bbox)
                    
                    if bbox[2] < 1 or bbox[3] < 1:
                        filter_list.append(img_idx)
            
            
            self.img_indices = [i for i in img_indices if i not in filter_list]
            
        else:
            self.img_indices = img_indices
        
        
        self.transforms = presets.DetectionPresetTrain(
                        data_augmentation="val", backend="pil", use_v2=False)
        


    def convert_coco_poly_to_mask(self, segmentations, height, width):
        masks = []
        for polygons in segmentations:
            rles = pymask.frPyObjects(polygons, height, width)
            mask = pymask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            mask = mask.any(dim=2)
            masks.append(mask)
        if masks:
            masks = torch.stack(masks, dim=0)
        else:
            masks = torch.zeros((0, height, width), dtype=torch.uint8)
        return masks
    

    def __getitem__(self, index):
        image_id = self.img_indices[index]
        # image_id = 245576
        
        img_meta = self.coco.imgs[image_id]
        
        path = img_meta['file_name']
        print(os.path.join(self.root, path))
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        w, h = image.size
        
        anno = self.coco.loadAnns(self.coco.getAnnIds(image_id))
        anno = [obj for obj in anno if obj["iscrowd"] == 0]
        
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        
        
        segmentations = [obj["segmentation"] for obj in anno]
        masks = self.convert_coco_poly_to_mask(segmentations, h, w)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        # target["image_id"] = image_id

        # # for conversion to coco api
        # area = torch.tensor([obj["area"] for obj in anno])
        # iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        # target["area"] = area
        # target["iscrowd"] = iscrowd
        
        
        image, target = self.transforms(image, target)
        
        
        raw_image = Image.open(os.path.join(self.root, path)).convert('RGB')
        
        return image, target, raw_image#, path
        
      

    def __len__(self):
        return len(self.img_indices)



if __name__ == "__main__":
    s = CocoDataset(root="/mnt/d/Datasets/coco2017/train2017", 
                    json="/mnt/d/Datasets/coco2017/annotations/instances_train2017.json")

    i = 17710
    print(s.__getitem__(i)[1]['masks'][0].shape)
    print()