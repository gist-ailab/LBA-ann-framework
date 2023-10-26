import torch
import cv2
import os
import numpy as np


from loader.coco import CocoDataset
from tqdm import tqdm

from pycocotools import mask as pymask

import pickle

from models.segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def sam_load(device):
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    mask_generator = SamAutomaticMaskGenerator(
                            sam,
                            points_per_side = 32,
                            points_per_batch= 128,
                            pred_iou_thresh = 0.95,
                            stability_score_thresh= 0.9,
                            stability_score_offset = 1.0,
                            box_nms_thresh= 0.9,
                            crop_n_layers = 1,
                            crop_nms_thresh = 0.95,
                            crop_overlap_ratio = 512 / 1500,
                            crop_n_points_downscale_factor = 2,
                            min_mask_region_area = 100)
    
    return mask_generator




def main():
    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
    
    root = "/mnt/d/Datasets/coco2017/train2017"
    json_path = "/mnt/d/Datasets/coco2017/annotations/instances_train2017.json"
    
    dataset = CocoDataset(root=root, json=json_path)


    with torch.no_grad():
        # model = LabelGenerator(device)
        
        model = sam_load(device)
        
            
        for idx in tqdm(range(dataset.__len__())):

            image, target, img_name = dataset.__getitem__(idx)

            
            image = np.array(image)
            preds = model.generate(image)
            
            temp = []
            for pred in preds:
                # print(pred["segmentation"])
                # exit()
                rle = pymask.encode(pred["segmentation"])
                # ex) {'size': [360, 640], 'counts': b'e`V39V9m1H5N10000000001O2O01O2M4K`ce3'}
                temp.append(rle)
                
                s = pymask.decode(rle)
                # print(s.shape)
                # exit()
        
            with open("sam_masks/{}.pickle".format(img_name.split(".")[0]),'wb') as fw:
                pickle.dump(temp, fw)
     
    
    
if __name__ == "__main__": 
    main()