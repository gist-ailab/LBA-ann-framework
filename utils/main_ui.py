import os

from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *

from .mouse_event import BboxDrawer
from .json_loader import JsonLoader

class Ui_Dialog(BboxDrawer):

    def __init__(self):
        super().__init__()
        self.filename=[]
        self.path=[]
        self.current_im=[]
        
        self.ann_dict = {}
        self.image_ann = [] # [{'iscrowd': 0, 'image_id': 724, 'bbox': [120.07, 71.83, 134.49, 153.08], 'category_id': 13, 'id': 268988}, {}, {}, ...]
        
        self.begin, self.destination = QPoint(), QPoint()
        self.xh, self.yh = 0, 0
        
        self.obj_idx = 0
        
        # j_loder = JsonLoader()
        # j_loder.load_json()
        
        self.initUI()
        self.imsp_w, self.imsp_h = self.width()*(2/3), self.height()
        
        
        img_w, img_h = self.image.width(), self.image.height()
        self.xh, self.yh = abs(self.imsp_w - img_w)//2 , abs(self.imsp_h - img_h)//2
        
        
    def del_obj(self):
        self.image_ann
        
    
    def add_obj(self):
        self.obj_idx += 1
        # self.image_ann[self.obj_idx]
        
        self.obj_list.insertItem(self.obj_idx, "object_class_" + str(self.obj_idx))
        
    
    def save_obj(self):
        self.image_ann
        # print(self.image_ann)
        
    def select_obj(self):
        self.obj_list
        
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
        self.image_ann = []
        self.obj_list.clear()
        self.obj_idx = 0
        
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
        
        
        

    def initUI(self, ui_w=1200, ui_h=800):
        self.setMinimumSize(ui_w, ui_h)
        #########################################
        #########################################
        
        self.image_viewer = QVBoxLayout()
        self.viewer= QLabel(self)
        
        self.image = QPixmap(820,780)
        self.image.fill(Qt.gray)
        
        #########################################
        #########################################
        #dataset list layout
        dset_loader = QVBoxLayout() # dataset loading part
        self.file_list = QListView()
        
        self.fileModel = QFileSystemModel()
        
        dset_loader.addWidget(self.file_list)
        
        pn_select = QHBoxLayout()
        self.loadButton = QPushButton('Load')
        self.prevButton = QPushButton('Previous')
        self.nextButton = QPushButton('Next')
        
        pn_select.addStretch(1)
        pn_select.addWidget(self.loadButton)
        pn_select.addStretch(1)
        pn_select.addWidget(self.prevButton)
        pn_select.addStretch(1)
        pn_select.addWidget(self.nextButton)
        pn_select.addStretch(1)
        
        dset_loader.addLayout(pn_select)
        #########################################
        # ann obj list layout
        annlist_viewer = QVBoxLayout()
        self.obj_list = QListWidget()
        annlist_viewer.addWidget(self.obj_list)
        
        pn2_select = QHBoxLayout()
        self.saveButton = QPushButton('Save')
        self.delButton = QPushButton('Del')
        
        pn2_select.addStretch(1)
        pn2_select.addWidget(self.saveButton)
        pn2_select.addStretch(1)
        pn2_select.addWidget(self.delButton)
        pn2_select.addStretch(1)
        
        annlist_viewer.addLayout(pn2_select)
        #########################################
        
        self.ann_info = QListView()
        
        #########################################
        
        ann_tool = QVBoxLayout()
        ann_tool.addLayout(dset_loader)
        ann_tool.addLayout(annlist_viewer)
        
        ann_tool.addWidget(self.ann_info)
        #########################################
        #########################################
        
        self.main_layout = QHBoxLayout()
        self.main_layout.addLayout(self.image_viewer, 70)
        self.main_layout.addLayout(ann_tool, 30)
        
        self.setLayout(self.main_layout)
        self.retranslateUi()
        self.show()
        
    
    def open_dialog_box(self):
        self.filename=QFileDialog.getExistingDirectory()

        self.fileModel.setRootPath(self.filename)
        self.file_list.setModel(self.fileModel)
        self.file_list.setRootIndex(self.fileModel.index(self.filename))
        
        self.file_list.clicked.connect(self.on_clicked)
    
        
    def retranslateUi(self):
        self.loadButton.clicked.connect(self.open_dialog_box)
        self.nextButton.clicked.connect(self.next_im)
        self.prevButton.clicked.connect(self.previous_im)
        
        self.saveButton.clicked.connect(self.save_obj)
        self.delButton.clicked.connect(self.remove_bbox)
        
        self.obj_list.clicked.connect(self.select_ann)
        
        
    def next_im(self):
        directory=self.filename
        list_1=[]
        
        if directory==[] or self.current_im==[]:
            pass
        
        else:
            for f in os.listdir(directory):
                fpath = directory+'/'+f
                list_1.append(fpath)

            n=list_1.index(self.current_im)

            if n==(len(list_1)-1):
                k=0
            else:
                k=n+1

            self.current_im=list_1[k]
            self.on_clicked_1()
            self.image_ann = []
            
            
    def previous_im(self):
        directory=self.filename
        list_1=[]
        
        if directory==[] or self.current_im==[]:
            pass
        
        else:
            for f in os.listdir(directory):
                fpath = directory+'/'+f
                list_1.append(fpath)

            n=list_1.index(self.current_im)

            self.current_im=list_1[n-1]
            self.on_clicked_1()
            self.image_ann = []
            
        
    

        