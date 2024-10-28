from src.segment_anything2.sam2.build_sam import build_sam2
from src.segment_anything2.sam2.sam2_image_predictor import SAM2ImagePredictor

# from src.marineinst_sam import sam_model_registry, SamPredictor

# from src.segment_anything import sam_model_registry, SamPredictor


import torch


import cv2
import numpy as np



def mask2poly(mask):
    
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return np.array([])
    
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.001 * cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, epsilon, True)

    polygon = polygon.reshape(-1, 2)
    
    return polygon


class Sam2Labler():
    def __init__(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"using device: {device}")

        # if device.type == "cuda":
        #     # use bfloat16 for the entire notebook
        #     torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        #     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        #     if torch.cuda.get_device_properties(0).major >= 8:
        #         torch.backends.cuda.matmul.allow_tf32 = True
        #         torch.backends.cudnn.allow_tf32 = True
                
        # elif device.type == "mps":
        #     print(
        #         "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        #         "give numerically different outputs and sometimes degraded performance on MPS. "
        #         "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        #         )
        
        
        
        sam2_checkpoint = "src/segment_anything2/checkpoints/sam2_hiera_tiny.pt"
        model_cfg = "../sam2_configs/sam2_hiera_t.yaml"
    

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        
        
        # # sam_checkpoint = "src/marineinst_sam/marineinst_vit-h_stage1.pth"
        # sam_checkpoint = "src/segment_anything/sam_vit_h_4b8939.pth"
        # model_type = "vit_h"
    
        # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        # sam.to(device=device)
        
        # self.predictor = SamPredictor(sam)
        
    
    def set_image(self, image):
        print(image.shape)
        self.predictor.set_image(image=image)


    def box2polygon(self, input_box):
        input_box = np.array(input_box)

        masks, scores, _ = self.predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=input_box[None, :],
                            multimask_output=False,
                            )
        
        polygon = mask2poly(masks[0])
        return polygon


    def point2polygon(self,input_point):
        input_point = np.asarray(input_point)
        input_label = np.array([1])
        print(input_point)
        
        masks, scores, _ = self.predictor.predict(
                                    point_coords=input_point,
                                    point_labels=input_label,
                                    multimask_output=True,
                                    )
        
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        polygon = mask2poly(masks[0])
        return polygon