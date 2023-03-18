# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'screen_page.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(752, 557)
        MainWindow.showMaximized()
        MainWindow.setStyleSheet("background-color: rgb(54, 54, 54);\n"
"color:rgb(225,225,225);font-size:10pt")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.camera_label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.camera_label.sizePolicy().hasHeightForWidth())
        self.camera_label.setSizePolicy(sizePolicy)
        self.camera_label.setMinimumSize(QtCore.QSize(0, 0))
        self.camera_label.setStyleSheet("background-color: rgb(40,40,40);\n"
"color:rgb(225,225,225);font-size:10pt")
        self.camera_label.setText("")
        self.camera_label.setObjectName("camera_label")
        self.verticalLayout.addWidget(self.camera_label)
        self.widget = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setMinimumSize(QtCore.QSize(0, 50))
        self.widget.setMaximumSize(QtCore.QSize(16777215, 55))
        self.widget.setStyleSheet("background-color: rgb(54, 54, 54);\n"
"color:rgb(225,225,225);font-size:10pt")
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.start_button = QtWidgets.QPushButton(self.widget)
        self.start_button.setObjectName("Load a video")
        self.horizontalLayout.addWidget(self.start_button)
        self.verticalLayout.addWidget(self.widget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
    def setTitle(self, MainWindow, title):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", title))
        

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Facial Recognition with Emotion/Eye Detection"))
        MainWindow.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        MainWindow.setWindowIcon(QtGui.QIcon('icon.png'))
        self.start_button.setText(_translate("MainWindow", "Load a video"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

