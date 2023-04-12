import os

from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *

from .mouse_event import BboxDrawer
from .json_loader import JsonLoader
from .annotation import Annotator

class Ui_Dialog(BboxDrawer, Annotator):

    def __init__(self):
        super().__init__()
        self.filename=[]
        self.path=[]
        self.current_im=[]
        
        self.ann_dict = {}
        self.image_ann = [] # [{'iscrowd': 0, 'image_id': 724, 'bbox': [120.07, 71.83, 134.49, 153.08], 'category_id': 13, 'id': 268988}, {}, {}, ...]
        
        
        self.temp_ann_dict = {}
        
        self.begin, self.destination = QPoint(), QPoint()
        self.xh, self.yh = 0, 0
        
        self.obj_idx = 0
        
        self.initUI()
        self.imsp_w, self.imsp_h = self.width()*(2/3), self.height()
        
        
        img_w, img_h = self.image.width(), self.image.height()
        self.xh, self.yh = abs(self.imsp_w - img_w)//2 , abs(self.imsp_h - img_h)//2
        

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
        # self.ann_temp_save()
        
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
        # self.ann_temp_save()
        
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
            
        
    

        