import sys
from PySide6 import QtCore, QtGui, QtWidgets


from utils import main_ui

def main():
    app = QtWidgets.QApplication(sys.argv)
    # Dialog = QtWidgets.QDialog()
    
    ex = main_ui.Ui_Dialog()

    # Dialog.show()
    try:
        sys.exit(app.exec())
    except SystemExit:
        print('Closing Window...')


if __name__ == "__main__":
    main()
