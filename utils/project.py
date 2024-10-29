import os
import json

from natsort import natsorted

from PySide6 import QtGui
from PySide6 import QtCore
from widgets.shape import Shape

class AnnProjectManager():
    def __init__(self, path):
        
        self.project_folder = path
        self.image_folder = os.path.join(self.project_folder, "images").replace('\\', '/')
        self.ann_save_path = os.path.join(self.project_folder,"anns").replace('\\', '/')
        
        self.load_meta()
        self.load_img_list()
        self.load_ann_list()
        
        # self.filename=[]
        # self.path=[]
        # self.current_im=[]
        
    def load_meta(self):
        with open(os.path.join(self.project_folder, "meta.json"), 'r', encoding="utf-8") as f:
            meta_data = json.load(f)
            
        self.categories = [cate["name"] for cate in meta_data["categories"]]
        self.colors = [cate["fill_color"] for cate in meta_data["categories"]]
        self.name2color = {cate["name"]:cate["fill_color"] for cate in meta_data["categories"]}
        
        
        self.info = meta_data["info"]
        f.close()
        
        
    def load_img_list(self):
        self.image_list = natsorted(os.listdir(self.image_folder))
    
    def load_ann_list(self):
        self.anns_list = []
        for i_img_name in self.image_list:
            i_json_name = i_img_name.split('.')[0] + ".json"
            i_json_path = os.path.join(self.ann_save_path, i_json_name).replace('\\', '/')

            if os.path.exists(i_json_path):
                with open(i_json_path, 'r', encoding="utf-8") as f:
                    json_data = json.load(f)
                    
                    
                shapes: list[Shape] = []
                for shape_dict in json_data["shapes"]:
                    
                    color = self.name2color[shape_dict["label"]] + [128]
                    qcolor = QtGui.QColor(*color)
                    shape = Shape(label=shape_dict["label"], 
                                  shape_type=shape_dict["shape_type"],
                                  line_color=qcolor)
                    
                    shape.line_color = QtGui.QColor(0, 255, 0, 0)
                    shape.fill_color = qcolor

                    for point in shape_dict["points"]:
                        shape.addPoint(QtCore.QPoint(*point))
                    shapes.append(shape)
                            
                self.anns_list.append(shapes)
                f.close()
                
            else:
                shapes: list[Shape] = []
                # shapes format : {"label":[], "shape_type":"..."}
                self.anns_list.append(shapes)
                
            
                
    def save_ann_shapes(self, idx, width, height):
        image_name = self.image_list[idx]
        shapes = self.anns_list[idx]
        
        save_dict = {"shapes":[], 
                    "imagePath": image_name,
                    "imageData": None,
                    "imageHeight": height,
                    "imageWidth": width}
        
        for i_shape in shapes:
            i_dict = {
            "label": i_shape.label,
            "points": [[point.x(), point.y()] for point in i_shape.points],
            "group_id": i_shape.group_id,
            "shape_type": i_shape.shape_type,
            "flags": i_shape.flags
            }
            save_dict["shapes"].append(i_dict)
        
        save_path = os.path.join(self.ann_save_path, image_name.split(".")[0]+".json")
        with open(save_path, 'w', encoding="utf-8") as output_json_file:
            json.dump(save_dict, output_json_file, ensure_ascii = False, indent="\t")
        output_json_file.close()
        