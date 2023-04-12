from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *


CURSOR_DEFAULT = Qt.ArrowCursor
CURSOR_POINT = Qt.PointingHandCursor
CURSOR_DRAW = Qt.CrossCursor
CURSOR_MOVE = Qt.ClosedHandCursor
CURSOR_GRAB = Qt.OpenHandCursor


class AnnDrawer(QLabel):
	def __init__(self):
		super().__init__()


	def paintEvent(self, event):
		super().paintEvent(event)
		qp = QPainter(self)
		br = QBrush(QColor(100, 10, 10, 40))
		qp.setBrush(br)
		qp.drawRect(QRect(self.begin, self.end))


	def paint_bbox(self, begin, destination, color):

		painter = QPainter()
		painter.begin(self.image)

		rect = QRect(begin - QPoint(self.xh, self.yh), destination - QPoint(self.xh, self.yh))
		print(rect)
		painter.setPen(QPen(color, 2, Qt.SolidLine))
		painter.drawRect(rect.normalized())

		painter.device()
		painter.end()
		del painter


class BboxDrawer(QWidget):
	def __init__(self):
		super().__init__()
		self.begin, self.destination = QPoint(), QPoint()
		

	def paintEvent(self, event):
		painter = QPainter(self)
		painter.drawPixmap(QPoint(self.xh, self.yh), self.image)
		
		if not self.begin.isNull() and not self.destination.isNull():
		
			rect = QRect(self.begin, self.destination)
			color = QColor(255, 0, 0)
			painter.setPen(QPen(color, 2, Qt.SolidLine))
			brush = QBrush(Qt.BDiagPattern)
			painter.setBrush(brush)
			painter.drawRect(rect.normalized())


	def mousePressEvent(self, event):
		if Qt.LeftButton:
			self.begin = event.position().toPoint()
			self.destination = self.begin
			self.update()
			
            
	def mouseMoveEvent(self, event):
		if  Qt.LeftButton:
			
			self.destination = event.position().toPoint()
			self.update()


	def mouseReleaseEvent(self, event):
		if  Qt.LeftButton:
			color = QColor(0, 0, 0)
			self.paint_bbox(self.begin, self.destination, color)
   
			self.record_bbox_ann() # record ann bbox information
			self.add_obj()
			self.begin, self.destination = QPoint(), QPoint()
			self.update()


	def paint_bbox(self, begin, destination, color):
		painter = QPainter()
		painter.begin(self.image)
  

		rect = QRect(begin - QPoint(self.xh, self.yh), destination - QPoint(self.xh, self.yh))
		print(rect)
		painter.setPen(QPen(color, 2, Qt.SolidLine))
		painter.drawRect(rect.normalized())
  
		painter.device()
		painter.end()
		del painter
 
 

	def on_clicked(self, index):
     
		self.ann_init()
  
		self.path = self.fileModel.fileInfo(index).absoluteFilePath()
		self.image = QPixmap(self.path).scaled(QSize(self.imsp_w + 20, self.imsp_h - 20), aspectMode=Qt.KeepAspectRatio)
  
		img_w, img_h = self.image.width(), self.image.height()
		self.xh, self.yh = abs(self.imsp_w - img_w)//2 , abs(self.imsp_h - img_h)//2
  
		self.current_im=self.path
		self.update()
		
  
	def on_clicked_1(self):
     
		self.ann_init()
  
		self.path = self.current_im
		self.image = QPixmap(self.path).scaled(QSize(self.imsp_w + 20, self.imsp_h - 20), aspectMode=Qt.KeepAspectRatio)

		img_w, img_h = self.image.width(), self.image.height()
		self.xh, self.yh = abs(self.imsp_w - img_w)//2 , abs(self.imsp_h - img_h)//2
  
		self.update()



