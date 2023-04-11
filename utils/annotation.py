from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *


class Annotator():
    def __init__(self):
        super().__init__()

        self.ann_dict = {}
        self.image_ann = [] # [{'iscrowd': 0, 'image_id': 724, 'bbox': [120.07, 71.83, 134.49, 153.08], 'category_id': 13, 'id': 268988}, {}, {}, ...]
        
        
        self.temp_ann_dict = {}
        
        self.begin, self.destination = QPoint(), QPoint()
        self.xh, self.yh = 0, 0
        
        self.obj_idx = 0
        
        # j_loder = JsonLoader()
        # j_loder.load_json()
        
        self.initUI()
        self.imsp_w, self.imsp_h = self.width()*(2/3), self.height()
        
        
        img_w, img_h = self.image.width(), self.image.height()
        self.xh, self.yh = abs(self.imsp_w - img_w)//2 , abs(self.imsp_h - img_h)//2
        
        
    def add_obj(self):
        self.obj_idx += 1
        self.obj_list.insertItem(self.obj_idx, "object_class_" + str(self.obj_idx))
        
    
    def save_obj(self):
        self.image_ann
        # print(self.image_ann)
        
        
    def remove_bbox(self):
        idx = self.obj_list.currentRow()
        
        x, y, w, h = self.image_ann[idx]['bbox']
  
        copy_image_ann = self.image_ann.copy()
        del copy_image_ann[idx]
        
        self.image = QPixmap(self.path).scaled(QSize(self.imsp_w + 20, self.imsp_h - 20), 
                                               aspectMode=Qt.KeepAspectRatio)
        
        for ann in copy_image_ann:
            x, y, w, h= ann['bbox']
            begin = QPoint(x,y)
            destination = QPoint(x+w, y+h)
            color = QColor(0, 0, 0)
            self.paint_bbox(begin, destination, color)

            
        self.obj_list.setFocus()
        
        self.update()
        
        self.image_ann = copy_image_ann.copy()
        
        self.obj_list.takeItem(idx)
        
        
        
    def record_bbox_ann(self):
        # print(self.begin.toTuple(), self.destination.toTuple())
        x1, y1 = self.begin.toTuple()
        x2, y2 = self.destination.toTuple()
        
        temp = {'iscrowd': 0,'image_id' :0,
                'bbox':[min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1)], 
                'category_id': 0, 'id': 0}
        
        self.image_ann.append(temp)
        # print(self.image_ann)
        
        
    def load_ann(self):
        self.image_ann
        
        
    def ann_init(self):
        # self.temp_ann_dict[self.path] = self.image_ann.copy()
        
        self.image_ann = []
        self.obj_list.clear()
        self.obj_idx = 0
        
        # print(self.path)
        # print(self.temp_ann_dict)
        # if self.path == []:
        #     pass
        
        # else:
        #     self.image_ann = self.temp_ann_dict[str(self.path)]
        
    
    def ann_temp_save(self):
        print(self.path)
        self.temp_ann_dict[str(self.path)] = self.image_ann.copy()
        
     
    def select_ann(self):
        idx = self.obj_list.currentRow()
        
        x, y, w, h = self.image_ann[idx]['bbox']
  
        copy_image_ann = self.image_ann.copy()
        
        self.image = QPixmap(self.path).scaled(QSize(self.imsp_w + 20, self.imsp_h - 20), aspectMode=Qt.KeepAspectRatio)
        
        for i, ann in enumerate(copy_image_ann):
            x, y, w, h= ann['bbox']
            begin = QPoint(x,y)
            destination = QPoint(x+w, y+h)
            
            if i == idx:
                color = QColor(255, 0, 0)
            else:
                color = QColor(0, 0, 0)
                
            self.paint_bbox(begin, destination, color)
            
        self.update()
        
        