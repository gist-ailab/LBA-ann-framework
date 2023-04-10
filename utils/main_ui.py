import os

from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *

from .mouse_event import BboxDrawer

class Ui_Dialog(BboxDrawer):

    def __init__(self):
        super().__init__()
        self.filename=[]
        self.path=[]
        self.current_im=[]
        
        self.ann_dict = {}
        self.image_ann = [] # {'iscrowd': 0, 'image_id': 724, 'bbox': [120.07, 71.83, 134.49, 153.08], 'category_id': 13, 'id': 268988}, {}, {}, ...
        
        self.begin, self.destination = QPoint(), QPoint()
        self.initUI()
    
    
    def record_bbox_ann(self):
        print(self.begin.toTuple(), self.destination.toTuple())
        x1, y1 = self.begin.toTuple()
        x2, y2 = self.destination.toTuple()
        
        temp = {'iscrowd': 0,'image_id' :0,
                'bbox':[min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1)], 'category_id': 0, 'id': 0}
        
        self.image_ann.append(temp)
        print(self.image_ann)
        
    def save_bbox_ann(self):
        
        self.image_ann
    

    def initUI(self, ui_w=1200, ui_h=800):
        self.setMinimumSize(ui_w, ui_h)
        # self.setWindowTitle('Box Layout')
        # self.setGeometry(1000, 600, 300, 200)
        
        #########################################
        #########################################
        
        image_viewer = QVBoxLayout()
        self.viewer= QLabel()

        self.image = QPixmap()
        self.viewer.setPixmap(self.image)
        self.viewer.setAlignment(Qt.AlignCenter)
        
        image_viewer.addWidget(self.viewer)
        
        
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
        self.obj_list = QListView()
        self.ann_info = QListView()
        #########################################
        
        ann_tool = QVBoxLayout()
        ann_tool.addLayout(dset_loader)
        
        ann_tool.addWidget(self.obj_list)
        ann_tool.addWidget(self.ann_info)
        #########################################
        #########################################
        
        self.main_layout = QHBoxLayout()
        self.main_layout.addLayout(image_viewer, 70)
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