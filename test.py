import sys
from PySide6.QtCore import QSize, QRect, Qt

from PySide6.QtGui import QPixmap

from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton



class MovableButton(QPushButton):

    def mousePressEvent(self, event):
        super().mousePressEvent(event)



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "Image Viewer"
        self.setWindowTitle(self.title)
        # self.setMinimumSize(1024, 512)
        self.setFixedSize(1024, 512)

        self.button = {}
        self.width = 1024
        self.height = 512
        self.Label = QLabel(self)
        self.Label.resize(self.width, self.height)
        
        self.load_image()
        
        MainWindow.select_button(self)
        MainWindow.load_button(self)
        
        
        self.show()
        
    def load_image(self):
        labelImage = QLabel(self)
        labelImage.resize(self.width, self.height)
        labelImage.move(50, 0)
        pixmap = QPixmap('cat.jpg')
        pixmap = pixmap.scaled(500, 500, Qt.KeepAspectRatio)
        
        labelImage.setPixmap(pixmap)
        
        

        self.resize(pixmap.width(), pixmap.height())
        

    def select_button(self):
        i=190
        
        button_name = ["Previous", "Next"]
        
        for j in range(len(button_name)):
            self.button[j] = MovableButton(self.Label)
            self.button[j].setGeometry(QRect(i, 450, 80, 30))
            self.button[j].setText(button_name[j])
            self.button[j].setObjectName(button_name[j])
            self.button[j].show()
            i = i + 100
            
    def load_button(self):
        i=700
        
        button_name = "Load"
        
        j = len(self.button)

        self.button[j] = MovableButton(self.Label)
        self.button[j].setGeometry(QRect(i, 450, 80, 30))
        self.button[j].setText(button_name)
        self.button[j].setObjectName(button_name)
        self.button[j].show()

    def draw_bbox(self):
        
        return


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()