import torch
import cv2
import os
import numpy as np
import argparse


import torch.nn.functional as F

import matplotlib.pyplot as plt

from loader.coco import CocoDataset

from models.ssl_model import InstFrame


def show_anns(masks):

    ax = plt.gca()
    ax.set_autoscale_on(False)

    # print(masks)
    # exit()
    img = np.ones((masks[0].shape[0], masks[0].shape[1], 4))
    img[:,:,3] = 0

    
    # print(output)
    # print("aaa",len(output), len(anns))
    # exit()
    for idx, mk in enumerate(masks):

        # class_name = output[idx]['class']
        # pred_score = output[idx]['score'] * 100
        
        # if pred_score < 90:
        #     continue
        
        # mk = mk.detach().cpu()
        th = 0.5
        mk[mk>=th] = 1
        mk[mk<th] = 0
        
        # m = mk.type(torch.bool)
        m = mk.astype(np.bool_)
                
        rand_color = np.random.random(3)
        color_mask = np.concatenate([rand_color, [0.35]])
        
        img[m] = color_mask
        # break
        
    ax.imshow(img)


def main(args):

    GPU_NUM = args.gpu_num    
    args.device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    
    root = "/mnt/d/Datasets/coco2017/val2017"
    json_path = "/mnt/d/Datasets/coco2017/annotations/instances_val2017.json"
    
    dataset = CocoDataset(root=root, json=json_path, train=False)
    
    random_idx = np.random.randint(dataset.__len__())
    # random_idx = 2111
    print(random_idx)
    
    
    model = InstFrame(num_classes=91, args=args)
    
    
    # ckp_path = "checkpoints/Wed_Aug_16_18-11-52_2023/epoch4.pth"
    # state_dict = torch.load(ckp_path, map_location=args.device)
    # model.load_state_dict(state_dict)
    
    image, target, raw_image = dataset.__getitem__(random_idx)
    
    # print(unlabel_pack)
    
    # exit()
    # image_name = "000000442078.jpg"
    

    with torch.no_grad():
        model.eval()
        
        # image = np.array(labeled_image)
        # masks = np.array(labeled_pseudo)
        
        # exit()
        image = image.to(args.device)
        image = image.unsqueeze(0)
        target = [target]
        target = [{k: v.to(args.device) for k, v in t.items()} for t in target]
        
        output = model(image)
        
        # masks = torch.sigmoid(output)
        # masks = torch.mean(output, dim=1)
        # print(masks.shape)
        # exit()
        
        # print(output)
        # exit()
        masks = output[0]["masks"].squeeze(1).detach().cpu().numpy()
        # print(masks.shape)
        
        # print(masks)
        # exit()

        
        raw_image = np.array(raw_image)
        h, w, _  = raw_image.shape

        # masks = F.interpolate(masks.unsqueeze(0), (h,w))
        # masks = masks.squeeze(0)
        
        
        
        plt.figure(figsize=(20,20))
        plt.imshow(raw_image)
        show_anns(masks)
        plt.axis('off')
        plt.savefig("vis.jpg")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-num", type=int, default=0, help="gpu id number")
    
    parser.add_argument("--dataset-root", type=str, default="/mnt/d/Datasets", help="dataset name")
    parser.add_argument("--dataset-name", type=str, default="coco", help="dataset name")
    
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--save-dir", type=str, default="checkpoints")
 
    args = parser.parse_args()


    main(args)