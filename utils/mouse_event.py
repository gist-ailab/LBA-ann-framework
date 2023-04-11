from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *


CURSOR_DEFAULT = Qt.ArrowCursor
CURSOR_POINT = Qt.PointingHandCursor
CURSOR_DRAW = Qt.CrossCursor
CURSOR_MOVE = Qt.ClosedHandCursor
CURSOR_GRAB = Qt.OpenHandCursor


class BboxDrawer(QWidget):
	def __init__(self):
		super().__init__()
		self.begin, self.destination = QPoint(), QPoint()
		

	def paintEvent(self, event):
		
		painter = QPainter(self)
		painter.drawPixmap(QPoint(self.xh, self.yh), self.image)
		
		if not self.begin.isNull() and not self.destination.isNull():
		
			rect = QRect(self.begin, self.destination)
			painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
			brush = QBrush(Qt.BDiagPattern)
			painter.setBrush(brush)
			painter.drawRect(rect.normalized())

			painter.end()
			del painter


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
			self.paint_bbox(self.begin, self.destination)
			# self.viewer.setPixmap(self.image)
   
			self.record_bbox_ann() # record ann bbox information
			self.add_obj()
			self.begin, self.destination = QPoint(), QPoint()
			self.update()


	def paint_bbox(self, begin, destination):
		painter = QPainter()
		painter.begin(self.image)
		print(begin - QPoint(self.xh, self.yh), destination - QPoint(self.xh, self.yh))
  
		rect = QRect(begin - QPoint(self.xh, self.yh), 
					 destination - QPoint(self.xh, self.yh))
		
		painter.setPen(QPen((QColor(255, 0, 0)), 2, Qt.SolidLine))
		painter.drawRect(rect.normalized())
  
		painter.device()
		painter.end()
		del painter
 
 

	def on_clicked(self, index):
		self.path = self.fileModel.fileInfo(index).absoluteFilePath()
		self.image = QPixmap(self.path).scaled(QSize(self.imsp_w + 20, self.imsp_h - 20), aspectMode=Qt.KeepAspectRatio)
  
		img_w, img_h = self.image.width(), self.image.height()
		self.xh, self.yh = abs(self.imsp_w - img_w)//2 , abs(self.imsp_h - img_h)//2
  
		# self.viewer.setPixmap(self.image)
		self.current_im=self.path
		self.update()
		self.ann_init()
		# self.scroll(self.path)
  
		
	def on_clicked_1(self):

		self.path = self.current_im
		self.image = QPixmap(self.path).scaled(QSize(self.imsp_w + 20, self.imsp_h - 20), aspectMode=Qt.KeepAspectRatio)

		img_w, img_h = self.image.width(), self.image.height()
		self.xh, self.yh = abs(self.imsp_w - img_w)//2 , abs(self.imsp_h - img_h)//2
  
  		# self.viewer.setPixmap(self.image)
		self.update()
		self.ann_init()
		print(self.fileModel.index())
		# self.scroll(self.path)
  
	# def scroll(self, name):
	# 	item = self.file_list.findItems(name, Qt.MatchRegExp)[0]
	# 	item.setSelected(True)
	# 	self.file_list.scrollToItem(item, QAbstractItemView.PositionAtTop)


