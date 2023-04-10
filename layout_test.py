import sys
import os

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout

from PySide6.QtWidgets import *

from PySide6.QtCore import Qt, QPoint, QRect, QSize

from PySide6.QtGui import QPixmap


class MyApp(QWidget):

    def __init__(self, Dialog):
        super().__init__()
        self.filename=[]
        self.path=[]
        self.current_im=[]
        
        self.initUI(Dialog)

    def initUI(self, Dialog, ui_w=1200, ui_h=800):
        Dialog.setObjectName("Dialog")
        Dialog.resize(ui_w, ui_h)
        
        self.setMinimumSize(ui_w, ui_h)
        # self.setWindowTitle('Box Layout')
        # self.setGeometry(1000, 600, 300, 200)
        
        #########################################
        #########################################
        
        image_viewer = QVBoxLayout()
        self.viewer= QtWidgets.QLabel()
        
        pix = QPixmap()
        pix.fill(Qt.gray)
        self.viewer.setPixmap(pix)
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
        
        main_layout = QHBoxLayout(Dialog)
        main_layout.addLayout(image_viewer, 70)
        main_layout.addLayout(ann_tool, 30)
        
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

        
    def open_dialog_box(self):
        self.filename=QFileDialog.getExistingDirectory()

        self.fileModel.setRootPath(self.filename)
        self.file_list.setModel(self.fileModel)
        self.file_list.setRootIndex(self.fileModel.index(self.filename))
        self.file_list.clicked.connect(self.on_clicked)
        
        
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))

        
        self.loadButton.clicked.connect(self.open_dialog_box)
        self.nextButton.clicked.connect(self.next_im)
        self.prevButton.clicked.connect(self.previous_im)
    
    
    def on_clicked(self, index):
        self.path = self.fileModel.fileInfo(index).absoluteFilePath()
        self.viewer.setPixmap(QtGui.QPixmap(self.path).scaled(QSize(640, 360), aspectMode=Qt.KeepAspectRatio))
        self.viewer.setAlignment(Qt.AlignCenter)
        self.current_im=self.path
     
        
    def on_clicked_1(self):
        self.path = self.current_im
        self.viewer.setPixmap(QtGui.QPixmap(self.path).scaled(QSize(640, 360), aspectMode=Qt.KeepAspectRatio))
        self.viewer.setAlignment(Qt.AlignCenter)

        
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
            self.viewer.setPixmap(QtGui.QPixmap(self.current_im).scaled(QSize(640, 360), aspectMode=Qt.KeepAspectRatio))
            self.viewer.setAlignment(Qt.AlignCenter)
    


if __name__ == '__main__':
    app = QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ex = MyApp(Dialog)
    Dialog.show()
    sys.exit(app.exec())