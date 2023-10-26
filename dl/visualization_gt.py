import torch
import cv2
import os
import numpy as np
import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torchvision.ops.boxes import masks_to_boxes

from loader.coco import CocoDataset


def show_anns(masks, labels, cate):

    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((masks[0].shape[0], masks[0].shape[1], 4))
    img[:,:,3] = 0

    
    # print(output)
    # print("aaa",len(output), len(anns))
    # exit()

    # labels += [0] * (200 - len(labels))     
    
    for idx, lb in enumerate(labels):

        class_name = cate[lb]
        # print(class_name, lb)
        # if class_name == None:
        #     continue
        
        mk = masks[idx]
        
        x1,y1,x2,y2 = masks_to_boxes(mk.unsqueeze(0))[0]
        
        x, y = int(x1), int(y1)
        w, h = int(x2-x1), int(y2- y1)
        
        mk = mk.type(torch.bool)
    
        rand_color = np.random.random(3)
        color_mask = np.concatenate([rand_color, [0.35]])

        img[mk] = color_mask
        
        ax.add_patch(patches.Rectangle(xy=(x,y),width=w,height=h, color=rand_color, fill=False))
        
        # print_txt = "class : {}".format(class_name)
        # ax.text(x, y, print_txt, backgroundcolor=rand_color)
        
    ax.imshow(img)


def main(args):

    GPU_NUM = args.gpu_num    
    args.device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    
    root = "/mnt/d/Datasets/coco2017/val2017"
    json_path = "/mnt/d/Datasets/coco2017/annotations/instances_val2017.json"
    
    dataset = CocoDataset(root=root, json=json_path)
    
    random_idx = np.random.randint(dataset.__len__())
    # random_idx = 6300
    # random_idx = 1000
    random_idx = 4430
    print(random_idx)
    
    cate = dataset.coco.dataset['categories']
    # print(cate)
    # exit()
    
    cate_list = [None] * 91

    
    for c in cate:
        c_idx = c["id"]
        cate_list[c["id"]] = c["name"]
   
    labeled_pack = dataset.__getitem__(random_idx)

    with torch.no_grad():


        # image = np.array(labeled_image)
        # masks = np.array(labeled_pseudo)
        
    
        image, target,_ = labeled_pack

        masks = target["masks"]
        labels = target["labels"]

        
        
        raw_image = np.array(image)
        raw_image = raw_image.transpose((1,2,0))
        # print(raw_image.shape)
        # exit()
        
        
        plt.figure(figsize=(20,20))
        plt.imshow(raw_image)
        show_anns(masks, labels, cate_list)
        plt.axis('off')
        # plt.savefig("vis.jpg")
        plt.savefig("vis_ori.jpg")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-num", type=int, default=0, help="gpu id number")
    
    parser.add_argument("--dataset-root", type=str, default="/mnt/d/Datasets", help="dataset name")
    parser.add_argument("--dataset-name", type=str, default="coco", help="dataset name")
    
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--save-dir", type=str, default="checkpoints")
 
    args = parser.parse_args()


    main(args)