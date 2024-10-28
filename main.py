import sys
from PySide6 import QtWidgets


from windows import main_ui

def main():
    app = QtWidgets.QApplication(sys.argv)
    ex = main_ui.MainWindow()
    try:
        sys.exit(app.exec())
    except SystemExit:
        print('Closing Window...')


if __name__ == "__main__":
    main()
