from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *


class BboxDrawer(QWidget):
	def __init__(self):
		super().__init__()
		self.begin, self.destination = QPoint(), QPoint()	


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
			painter = QPainter(self.image)
			painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
			painter.drawRect(rect.normalized())
			
			self.record_bbox_ann() # record ann bbox information
			print(self.image_ann)
			self.begin, self.destination = QPoint(), QPoint()
			self.update()


	def on_clicked(self, index):
		self.path = self.fileModel.fileInfo(index).absoluteFilePath()
		self.image = QPixmap(self.path).scaled(QSize(800, 500), aspectMode=Qt.KeepAspectRatio)
		self.current_im=self.path
		self.update()
		self.image_ann = []
		
  
	def on_clicked_1(self):
		self.path = self.current_im
		self.image = QPixmap(self.path).scaled(QSize(800, 500), aspectMode=Qt.KeepAspectRatio)
		self.update()
		self.image_ann = []


