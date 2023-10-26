import pickle
import os


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from pycocotools import mask as pymask

def show_anns(masks):

    ax = plt.gca()
    ax.set_autoscale_on(False)

    
    img = np.ones((masks[0].shape[0], masks[0].shape[1], 4))
    # print(img.shape)
    # exit()
    img[:,:,3] = 0
    
    # print(output)
    # print("aaa",len(output), len(anns))
    # exit()
    # print(len(masks))
    # exit()
    for idx, mask in enumerate(masks):
        mask = np.array(mask, dtype=bool)
        # m = mask
        # print(mask)
        
        # class_name = output[idx]['class']
        # pred_score = output[idx]['score'] * 100
        
        # if pred_score < 90:
        #     continue
        
        rand_color = np.random.random(3)
        color_mask = np.concatenate([rand_color, [0.8]])
        
        # x,y,w,h = ann['bbox']
        
        img[mask] = color_mask
        
        # ax.add_patch(patches.Rectangle(xy=(x,y),width=w,height=h, color=rand_color, fill=False))
        
        # print_txt = "{}: {}%".format(class_name, round(pred_score))
        # ax.text(x, y, print_txt, backgroundcolor=rand_color)
        
    ax.imshow(img)

def main():
    path = "sam_masks"
    root = "/mnt/d/Datasets/coco2017/train2017"
    
    label_file_list = os.listdir(path)
    
    # for file_n in label_file_list:
    #     print(file_n.split('.')[0])
    
    
    # random_idx = np.random.randint(len(label_file_list))
    
    # file_n = label_file_list[random_idx]
    # print(file_n.split('.')[0])
    
    file_n = "000000023309.pickle"
    
    image = np.array(Image.open(os.path.join(root, file_n.split('.')[0] +".jpg")))
    
    with open(os.path.join(path, file_n), 'rb') as fr:
        rles = pickle.load(fr)
    
    
    
    masks = []
    print(len(rles))
    # exit()
    for rle in rles:
        mask = pymask.decode(rle)
        # print(mask.shape)
        # print(np.unique(mask, return_counts=True))
        # exit()
        # print(mask)
        # exit()
        bbox = pymask.toBbox(rle)
        # print(bbox)
        masks.append(mask)
        # continue
    # exit()
    
    # print(len(masks))
    # exit()
    assert len(rles) == len(masks), "size error!!!" 
    
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig("sam_mask_vis.jpg")
    exit()
    

main()