import sys
from PySide6 import QtWidgets


from utils import main_ui

def main():
    app = QtWidgets.QApplication(sys.argv)
    ex = main_ui.Ui_Dialog()
    try:
        sys.exit(app.exec())
    except SystemExit:
        print('Closing Window...')


if __name__ == "__main__":
    main()
