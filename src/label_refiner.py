import cv2
import numpy as np

import torch



class LabelRefiner():
    def __init__(self):

        self.refiner = None


    def search_noise(self, shapes):
        
        shape_num = len(shapes)
        idx = np.random.randint(0, shape_num, size=1)[0]
        
        return idx