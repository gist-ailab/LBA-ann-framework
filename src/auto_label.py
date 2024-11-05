from src.segment_anything2.sam2.build_sam import build_sam2
from src.segment_anything2.sam2.sam2_image_predictor import SAM2ImagePredictor


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
        
        sam2_checkpoint = "src/segment_anything2/checkpoints/sam2_hiera_tiny.pt"
        model_cfg = "../sam2_configs/sam2_hiera_t.yaml"
    

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        
        
    
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