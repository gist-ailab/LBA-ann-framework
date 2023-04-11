import os
import json

class JsonLoader():
    def __init__(self):
        super().__init__()
        self.filename=[]
        self.path=[]
        
        self.dataset_path = "D:/Datasets/coco2017/val2017/"
        # self.current_im = "D:/Datasets/coco2017/val2017/000000000724.jpg"
        self.ann_dict = {}
        
        self.json_path = "instances_val2017.json"

        self.load_json()
    
    
    def make_json(self, categories_list=None):
        init_json = {"info":[], "licenes":[], "images":[], "annotations":[], "categories":[]}
        
        categories_list=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 
                         'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
                         'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 
                         'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
                         'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
                         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
                         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 
                         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
                         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
                         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 
                         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 
                         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
                         'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        
        for idx, f in enumerate(os.listdir(self.dataset_path)):
            init_json["images"].append({
                "file_name" : f,
                "height" : 0,
                "width" : 0,
                "id" : idx
            })
            
        for idx, cate in enumerate(categories_list):
            init_json["categories"].append({
                "id" : idx+1,
                "name" : cate 
            })
            
        return init_json
        
    
    def load_json(self):
        
        if True:
            with open(self.json_path, 'r') as file:
                data = json.load(file)
                    
            self.ann_json = data
            
        else:
            self.ann_json = self.make_json(self)
    
            
        self.image2idx = {}
        for img_dict in self.ann_json["images"]:
    
            file_name = img_dict["file_name"]
            img_idx = img_dict["id"]
            self.image2idx[file_name] = img_idx
            
        self.categories = self.ann_json["categories"]
        self.class_list = [cat['name'] for cat in self.categories]

        
    def load_ann(self):
        image_idx = self.image2idx[self.current_im.split("/")[-1]]

        image_ann = []
        for idx, ann_dict in enumerate(self.ann_json["annotations"]):
            if ann_dict["image_id"] == image_idx:
                image_ann.append(ann_dict)
        
        self.image_ann = image_ann
        

    def save_ann(self):
        self.ann_dict
        self.image_ann
        
        
    def update_ann(self):
        self.image_ann

