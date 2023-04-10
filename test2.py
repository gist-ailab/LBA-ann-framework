from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import QFileDialog
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import QColor,QTextCharFormat
import os


class Ui_Dialog(object):
    def setupUi(self, Dialog, ui_w=1200, ui_h=600, ui_margin=20):
        self.filename=[]
        self.path=[]
        self.current_im=[]
        
        #################################
        # set main window
        
        Dialog.setObjectName("Dialog")
        Dialog.resize(ui_w, ui_h)
        # self.setMinimumSize(1024, 512)
        #################################
        
        
        #################################
        # floder finder
        
        finder_x, finder_y = ui_margin, 70
        self.toolButton = QtWidgets.QToolButton(Dialog)
        self.toolButton.setGeometry(QtCore.QRect(finder_x + 90, finder_y, 30, 20))
        self.toolButton.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
        self.toolButton.setObjectName("toolButton") # floder finder

        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(finder_x, finder_y, 80, 20))
        self.label.setObjectName("label") # floder finder label
        #################################


        #################################
        # image list in dataset floder
        
        vl_x, vl_y = ui_margin, 140
        vl_w, vl_h = 200, 400
        
        self.listview = QListView(Dialog)
        self.fileModel = QFileSystemModel(Dialog)
        self.listview.setGeometry(QtCore.QRect(vl_x, vl_y, vl_w, vl_h))
        #################################


        #################################
        # image viewer area
        
        image_x, image_y = 220+20, 140
        image_w, image_h = 640, 360
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(image_x, image_y, image_w, image_h))
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.label_3.setStyleSheet("color: red;"
                      "border-style: solid;"
                      "border-width: 2px;"
                      "border-color: #403c3c;"
                      "border-radius: 3px")
        #################################
        
        
        #################################
        #previous & next button
        
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(290, 510, 81, 41))
        self.pushButton.setObjectName("pushButton")

        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(380, 510, 81, 41))
        self.pushButton_2.setObjectName("pushButton_2")
        #################################
        
    
        self.comboBox = QtWidgets.QComboBox(Dialog)
        self.comboBox.setGeometry(QtCore.QRect(1000, 140, 69, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")

        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(1000, 120, 51, 16))
        self.label_4.setObjectName("label_4")


        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def onChanged(self, text):

        DB=text+'.csv'
        print()

            
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))

        self.toolButton.setText(_translate("Dialog", "..."))
        self.label.setText(_translate("Dialog", "Images folder"))

        self.comboBox.setItemText(0, _translate("Dialog", "1A"))
        self.comboBox.setItemText(1, _translate("Dialog", "2A"))
        self.comboBox.setItemText(2, _translate("Dialog", "3A"))
        self.comboBox.setItemText(3, _translate("Dialog", "5A"))
        self.comboBox.setItemText(4, _translate("Dialog", "12A"))
        self.comboBox.setItemText(5, _translate("Dialog", "17A"))
        self.comboBox.setItemText(6, _translate("Dialog", "19A"))
        self.comboBox.setItemText(7, _translate("Dialog", "20A"))
        self.comboBox.setItemText(8, _translate("Dialog", "22A"))
        self.comboBox.setItemText(9, _translate("Dialog", "Unsorted"))

        self.label_4.setText(_translate("Dialog", "Class:"))


        self.pushButton.setText(_translate("Dialog", "Previous"))
        self.pushButton_2.setText(_translate("Dialog", "Next"))
        
        self.toolButton.clicked.connect(self.open_dialog_box)
        
        self.pushButton_2.clicked.connect(self.next_im)
        self.pushButton.clicked.connect(self.previous_im)


    def open_dialog_box(self):
        self.filename=QFileDialog.getExistingDirectory()

        self.fileModel.setRootPath(self.filename)
        self.listview.setModel(self.fileModel)
        self.listview.setRootIndex(self.fileModel.index(self.filename))
        self.listview.clicked.connect(self.on_clicked)
        

    def on_clicked(self, index):
        self.path = self.fileModel.fileInfo(index).absoluteFilePath()
        self.label_3.setPixmap(QtGui.QPixmap(self.path).scaled(QSize(640, 360), aspectMode=Qt.KeepAspectRatio))
        self.label_3.setAlignment(Qt.AlignCenter)
        self.current_im=self.path

    def on_clicked_1(self):
        self.path = self.current_im
        self.label_3.setPixmap(QtGui.QPixmap(self.path).scaled(QSize(640, 360), aspectMode=Qt.KeepAspectRatio))
        self.label_3.setAlignment(Qt.AlignCenter)



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
            self.label_3.setPixmap(QtGui.QPixmap(self.current_im).scaled(QSize(640, 360), aspectMode=Qt.KeepAspectRatio))
            self.label_3.setAlignment(Qt.AlignCenter)
            # self.label_3.setScaledContents(True)





if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()

    sys.exit(app.exec())