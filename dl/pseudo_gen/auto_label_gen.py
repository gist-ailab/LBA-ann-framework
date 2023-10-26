import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import cv2

from PIL import Image

from models.segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from models.dinov2_model import Dinov2

from torchvision import transforms

from torchvision.transforms import functional as vF


from tqdm import tqdm

import numpy as np




resize = (224, 224) 

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



class LabelGenerator(nn.Module):
    def __init__(self, device) -> None:
        super(LabelGenerator, self).__init__()
        
        self.device = device
       
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        
        self.mask_generator = SamAutomaticMaskGenerator(
                                sam,
                                points_per_side = 16,
                                points_per_batch= 32,
                                pred_iou_thresh = 0.9,
                                stability_score_thresh= 0.9,
                                stability_score_offset = 1.0,
                                box_nms_thresh= 0.5,
                                crop_n_layers = 1,
                                crop_nms_thresh = 0.5,
                                crop_overlap_ratio = 512 / 1500,
                                crop_n_points_downscale_factor = 2,
                                min_mask_region_area = 10000)
        
        
        self.img_encoder = self.mask_generator.predictor.model.image_encoder
            
        self.dinov2 = Dinov2().to(self.device)
        
        
    @torch.no_grad()    
    def get_mask(self, image):
        masks = self.mask_generator.generate(image)
        return masks
    
    
    # @torch.no_grad()    
    # def get_mask(self, image):
    #     self.predictor.set_image(image)
        
    #     input_point = None
    #     input_label = None
        
    #     masks, scores, logits = self.predictor.predict(
    #     point_coords=input_point,
    #     point_labels=input_label,
    #     multimask_output=True)
    
    #     return masks
    
    
    @torch.no_grad()  
    def get_crop_img_save(self, image, save_root="crop_images"):
        crop_images = []
        
        masks = self.get_mask(image)
        
        
        for idx, mask in enumerate(masks):
            x,y,w,h = mask['bbox']
            seg_mask = mask["segmentation"]
            image_copy = image.copy() * np.expand_dims(seg_mask, -1)
            crop_img = image_copy[y:y+h, x:x+w]
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(save_root, ),  crop_img)
            
            
            crop_images.append(crop_img)
        
        return crop_images, masks
    
    @torch.no_grad()  
    def get_crop_img(self, image, raw_image):
        crop_images = []
        
        raw_image = np.array(raw_image)
        
        masks = self.get_mask(image)
        
        print(raw_image.shape)
        rh, rw, _ = raw_image.shape  # raw image
        
        fh, fw, _ = image.shape # resize image

        a, b = rw/fw, rh/fh
        
        for idx, mask in enumerate(masks):
            x,y,w,h = mask['bbox']
            seg_mask = mask["segmentation"]
            
            masks[idx]['bbox'] = [x*a, y*b, w*a, h*b]
            
            seg_mask = seg_mask.astype(np.uint8)
            seg_mask = cv2.resize(seg_mask, dsize=(rw, rh), interpolation=cv2.INTER_CUBIC)

            masks[idx]['segmentation'] = seg_mask.astype(np.bool_)
            
            
            image_copy = raw_image.copy() #* np.expand_dims(seg_mask, -1)
            
            
            
            # if (h < 224) or (w < 224):
            #     continue
            # print(y,y+h, x,x+w)
    
            # exit()
            x1, y1 = x*a, y*b
            x2, y2 = (x+w)*a, (y+h)*b
        
            crop_img = image_copy[int(y1):int(y2), int(x1):int(x2)]
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            crop_images.append(crop_img)
            
            # cv2.imwrite(os.path.join("crop_images", str(idx)+".jpg"),  crop_img)
        
        return crop_images, masks
    
    
    
    @torch.no_grad()
    def get_img_embedd(self, image):
        self.mask_generator.predictor.set_image(image)
        return self.mask_generator.predictor.get_image_embedding()
    
    
    def get_sp_feature(self, sp_masks, feats):
        
        # sp_masks          => b x 300 x 32 x 32
        # sp_masks(resized) => b x 300 x 1024 x   1
        # feats             => b x   1 x 1024 x 768
        #---------------------------------------------
        # sp_features       => b x 300 x 1024 x 768
        # sp_features       => b x 300 x  768
        
        bz, p_size, h, w = sp_masks.shape
        sp_masks = sp_masks.view(bz, p_size, h*w)
        
        sp_features = feats.unsqueeze(1) * sp_masks.unsqueeze(-1)
  
        # sp_features = torch.mean(sp_features, dim=2)
        
        
        # print(sp_features.shape)
        
        # exit()
        
        sp_features = torch.sum(sp_features, dim=2)
        d = torch.sum(sp_masks, dim=2, keepdim=True) + 1e-7
        sp_features = sp_features / d.detach()
        
        # sp_features = sp_features / sp_features.norm(dim=-1, keepdim=True)
        sp_features = F.normalize(sp_features, p=2, dim=1)
        
        return sp_features
    
    
    
    @torch.no_grad()
    def get_sim(self, image):
        
        transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
        image_dino = transform(image).to(self.device)
        # print(image_dino.shape)
        # exit()
        patch_feats = self.dinov2.get_patchtokens(image_dino.unsqueeze(0))
        print(patch_feats.shape)
        
        image_sam = np.array(image)
        target = self.mask_generator.generate(image_sam)
        # print(target)
        # exit()
        sam_masks_raw = np.array([gt["segmentation"] for gt in target])

        sam_masks = torch.as_tensor(sam_masks_raw).to(self.device)
        sam_masks = vF.resize(
                    sam_masks, [256, 256]
                )
        
        
        maxpool = nn.MaxPool2d(kernel_size=8, stride=8)
        pool_sam_masks = maxpool(sam_masks.float()).unsqueeze(0)
       
     
        mask_feature = self.get_sp_feature(pool_sam_masks, patch_feats)
        mask_feature = mask_feature.squeeze(0)

       
        
        sample_features = []
        
        class_n = sorted(os.listdir('coco_select'))
        
        for cls_n in tqdm(class_n):    

            cls_features = []
            num = 10
            for img_n in sorted(os.listdir(os.path.join('coco_select', cls_n))[:num]):
                s_image = Image.open(os.path.join('coco_select', cls_n, img_n))
                s_image = expand2square(s_image, (0, 0, 0))
                s_image = transform(s_image).unsqueeze(0).to(self.device)
                
                # sample_feat = self.dinov2(image)
                
                patch_feat = self.dinov2.get_patchtokens(s_image)
                
                # sample_feat = torch.sum(patch_feat, dim=1)
                # sample_feat = F.normalize(sample_feat, p=2, dim=1)
             
                sample_feat = patch_feat.mean(1)
                # sample_feat = sample_feat / sample_feat.norm(dim=-1, keepdim=True)
                
                cls_features.append(sample_feat)
            
            avg_feature = torch.cat(cls_features).mean(0, keepdim=True)
            # avg_feature = avg_feature / avg_feature.norm(dim=-1, keepdim=True)
            
            # avg_feature = torch.cat(cls_features)
  
            # avg_feature = torch.mean(avg_feature, dim=0, keepdim=True)
            # avg_feature = avg_feature / avg_feature.norm(dim=-1, keepdim=True)
            avg_feature = F.normalize(avg_feature, p=2, dim=0)
            
   
            sample_features.append(avg_feature)

        sample_features = torch.cat(sample_features, dim=0)
        print(sample_features.shape)

        # W = sample_features @ sample_features.T
        # W = W * (W > 0)
        # W = W / W.max()
        # th = 0.5
        # W[W>=th] = 1
        # W[W<th] = 0
        
        # print(torch.sum(W))
        
        cos = mask_feature@ sample_features.T
        
        cos = cos * (cos>0)
        cos = cos/cos.max()
        
        
        
        # cos = torch.zeros(size=(mask_feature.shape[0], sample_features.shape[0]))
        
        # for i in range(len(mask_feature)):
        #     for j in range(len(sample_features)):
        #         cos[i, j] = get_cosine_similarity(mask_feature[i], sample_features[j])
        
        
        # cos = (100 *cos).softmax(dim=1)
        # print(cos)
        # exit()
        
    
        # cos = (100 *crop_features @ sample_features.T).softmax(dim=1)
        
        
        output = []
        for idx, sim in enumerate(cos):
            val, cls = torch.topk(sim,1)
            output.append({"class": class_n[cls.item()],
                           "class_idx": cls.item(),
                           "score": val.item(),
                           })
            
        return output, target
            
def get_cosine_similarity(x1, x2):
    return (x1 * x2).sum() / ((x1**2).sum()**.5 * (x2**2).sum()**.5)
    
    

if __name__ == "__main__":
    input_t = torch.rand(size=(1,3,224,224)).cuda()
    model = LabelGenerator().cuda()