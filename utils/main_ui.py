import sys
import os

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import  QWidget, QPushButton, QHBoxLayout, QVBoxLayout

from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *

# from .ann_drawer import BboxDrawer

class Ui_Dialog(QWidget):

    def __init__(self):
        super().__init__()
        self.filename=[]
        self.path=[]
        self.current_im=[]
        
        # self.pix = QPixmap(600,600)
        # self.pix.fill(Qt.white)
        
        self.begin, self.destination = QPoint(), QPoint()
        
        self.initUI()
        
        

    def initUI(self, ui_w=1200, ui_h=800):
        self.setMinimumSize(ui_w, ui_h)
        # self.setWindowTitle('Box Layout')
        # self.setGeometry(1000, 600, 300, 200)
        
        #########################################
        #########################################
        
        image_viewer = QVBoxLayout()
        self.viewer= QtWidgets.QLabel()

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
        # QtCore.QMetaObject.connectSlotsByName()
    
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(QPoint(), self.image)
        
        if not self.begin.isNull() and not self.destination.isNull():
            rect = QRect(self.begin, self.destination)
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            painter.drawRect(rect.normalized())
    
    def mousePressEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.begin = event.pos()
            self.destination = self.begin
            self.update()
            
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:		

            self.destination = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() & Qt.LeftButton:
            rect = QRect(self.begin, self.destination)
            # print(self.begin.toTuple(), self.destination.toTuple())
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            painter.drawRect(rect.normalized())

            self.begin, self.destination = QPoint(), QPoint()
            self.update()
   
    
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
    
    
    def on_clicked(self, index):
        self.path = self.fileModel.fileInfo(index).absoluteFilePath()
        self.image = QtGui.QPixmap(self.path).scaled(QSize(800, 500), aspectMode=Qt.KeepAspectRatio)
        # self.viewer.setPixmap(self.image)
        # self.viewer.setAlignment(Qt.AlignCenter)
        self.current_im=self.path
        self.update()
     
        
    def on_clicked_1(self):
        self.path = self.current_im
        self.image = QtGui.QPixmap(self.path).scaled(QSize(800, 500), aspectMode=Qt.KeepAspectRatio)
        # self.viewer.setPixmap(self.image)
        # self.viewer.setAlignment(Qt.AlignCenter)
        self.update()

        
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
            self.image = QtGui.QPixmap(self.path).scaled(QSize(640, 360), aspectMode=Qt.KeepAspectRatio)
            # self.viewer.setPixmap(self.image)
            # self.viewer.setAlignment(Qt.AlignCenter)
    
