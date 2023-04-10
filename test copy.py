import sys, os
from PySide6.QtWidgets import QApplication, QWidget, QTableWidget, QAbstractItemView, QHeaderView, QTableWidgetItem, QVBoxLayout
from PySide6.QtGui import QIcon

class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        path_dir = 'D:/' #디렉토리
        file_list = os.listdir(path_dir)
        file_list_count = file_list.__len__() #list 갯수

        self.setWindowTitle('TEST') #App Title
        # self.setWindowIcon(QIcon('cloud.png')) #favicon 이미지 파일 필요

        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(file_list_count)
        self.tableWidget.setColumnCount(1)
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableWidget.setHorizontalHeaderLabels(["Type","Size","Value"])
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        for i in range(file_list_count):
            self.tableWidget.setItem(i, 0, QTableWidgetItem(file_list[i].format()))

        layout = QVBoxLayout()
        layout.addWidget(self.tableWidget)
        self.setLayout(layout)

        self.setGeometry(300, 100, 600, 400) #App size
        self.show()


if __name__ == '__main__':
   app = QApplication(sys.argv)
   ex = MyApp()
   sys.exit(app.exec_())