import os

from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *


from widgets.canvas import Canvas

class MainWindow(QMainWindow):

    def __init__(self, ui_w=1200, ui_h=800):
        super().__init__()
        self.setMinimumSize(ui_w, ui_h)
        self.label_hist = []
        
        
        if self.label_hist:
            self.default_label = self.label_hist[0]
            
        # self.settings = Settings()
        
        self.canvas = Canvas(parent=self)
        # self.canvas.set_drawing_shape_to_square(settings.get(SETTING_DRAW_SQUARE, False))
        # self.canvas.newShape.connect(self.new_shape)
        
        self.main_layout = QHBoxLayout()
        self.central_widget = QWidget(self)
        
        self.filename=[]
        self.path=[]
        self.current_im=[]
        
        self.initUI()
        self.show()
        

    def initUI(self):
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)
        
        self.main_layout.addWidget(self.canvas)

        self.canvas.mode = Canvas.CREATE
        # self.canvas.set_editing(True)
        
        #dataset list layout
        dset_loader = QVBoxLayout() # dataset loading part
        self.file_list = QListView()
        
        self.fileModel = QFileSystemModel()
        
        label1 = QLabel('File list', self)
        dset_loader.addWidget(label1)
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
        
        label2 = QLabel('Object list', self)
        annlist_viewer.addWidget(label2)
        annlist_viewer.addWidget(self.obj_list)
        
        pn2_select = QHBoxLayout()
        self.saveButton = QPushButton('Save')
        self.editButton = QPushButton('Edit')
        self.delButton = QPushButton('Del')
        
        pn2_select.addStretch(1)
        pn2_select.addWidget(self.saveButton)
        pn2_select.addStretch(1)
        pn2_select.addWidget(self.editButton)
        pn2_select.addStretch(1)
        pn2_select.addWidget(self.delButton)
        pn2_select.addStretch(1)
        
        annlist_viewer.addLayout(pn2_select)
        #########################################
        
        ann_info = QVBoxLayout()
        self.class_list = QListWidget()
        label3 = QLabel('Object class list', self)
        ann_info.addWidget(label3)
        ann_info.addWidget(self.class_list)
        
        #########################################
    
        ann_tool = QVBoxLayout()
        ann_tool.addLayout(dset_loader)
        ann_tool.addLayout(annlist_viewer)
        
        ann_tool.addLayout(ann_info)
        #########################################
        #########################################

        self.main_layout.addWidget(self.canvas, 70)
        self.main_layout.addLayout(ann_tool, 30)
        self.retranslateUi()
        
    
    def open_dialog_box(self):
        self.filename=QFileDialog.getExistingDirectory()

        self.fileModel.setRootPath(self.filename)
        self.file_list.setModel(self.fileModel)
        self.file_list.setRootIndex(self.fileModel.index(self.filename))
        
        self.file_list.clicked.connect(self.on_clicked)
        
        self.categories_list=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 
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
        
        for idx, cls in enumerate(self.categories_list):
            self.class_list.insertItem(idx, cls)
    
        
    def retranslateUi(self):
        self.loadButton.clicked.connect(self.open_dialog_box)
        self.nextButton.clicked.connect(self.next_im)
        self.prevButton.clicked.connect(self.previous_im)
        
        # self.saveButton.clicked.connect(self.save_obj)
        
        # self.editButton.clicked.connect(self.remove_obj)
        
        self.delButton.clicked.connect(self.remove_obj)
        
        
    
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
        
        
    def on_clicked(self, index):
  
        self.path = self.fileModel.fileInfo(index).absoluteFilePath()
        pixmap = QPixmap(self.path)
        scaled_pixmap = pixmap.scaled(self.canvas.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.canvas.loadPixmap(scaled_pixmap)
        self.current_im=self.path
        self.obj_list.clear()
        self.update()
		
    def on_clicked_1(self):
        
        self.path = self.current_im
        pixmap = QPixmap(self.path)
        scaled_pixmap = pixmap.scaled(self.canvas.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.canvas.loadPixmap(scaled_pixmap)
        self.obj_list.clear()
        self.update()
        
        
    def select_obj(self):
        idx = 0
        
        self.canvas.shapes[idx]
        self.canvas.select_shape()
    
    def de_select_obj(self):
        self.canvas.de_select_shape()
        
        
    def remove_obj(self):
        
        idx = self.obj_list.currentRow()
        self.update_obj_list("remove", idx)
        self.canvas.select_shape(self.canvas.shapes[idx])
        self.canvas.delete_selected()

        
    def update_obj_list(self, action, idx=None):
        
        cls_idx = self.class_list.currentRow()
        class_name = self.categories_list[cls_idx]
            
        if action=="remove":
            self.obj_list.takeItem(idx)
            
        elif action=="add":
            self.obj_list.addItem(class_name)
        
        
    

        