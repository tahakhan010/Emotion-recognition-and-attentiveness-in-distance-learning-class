# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dashboard.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtWidgets, QtGui

class Ui_GraphsWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1073, 701)
        MainWindow.showMaximized()
        MainWindow.setStyleSheet("background-color: rgb(54, 54, 54);\n"
"color:rgb(225,225,225);font-size:10pt")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setMinimumSize(QtCore.QSize(100, 50))
        self.groupBox.setMaximumSize(QtCore.QSize(16777215, 100))
        self.groupBox.setStyleSheet("border-style: solid;\n"
"border-width: 2px;\n"
"border-color:  rgb(40,40,40);;")
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(10, 30, 131, 28))
        self.pushButton.setObjectName("pushButton")
        self.comboBox = QtWidgets.QComboBox(self.groupBox)
        self.comboBox.setGeometry(QtCore.QRect(170, 30, 131, 31))
        self.comboBox.setObjectName("comboBox")
        self.verticalLayout.addWidget(self.groupBox)
        self.mlp_general = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mlp_general.sizePolicy().hasHeightForWidth())
        self.mlp_general.setSizePolicy(sizePolicy)
        self.mlp_general.setStyleSheet("border-style: solid;\n"
"border-width: 2px;\n"
"border-color:  rgb(40,40,40);;")
        self.mlp_general.setObjectName("mlp_general")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.mlp_general)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout.addWidget(self.mlp_general)
        self.mlp_student = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mlp_student.sizePolicy().hasHeightForWidth())
        self.mlp_student.setSizePolicy(sizePolicy)
        self.mlp_student.setStyleSheet("")
        self.mlp_student.setObjectName("mlp_student")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.mlp_student)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.widget_2 = QtWidgets.QWidget(self.mlp_student)
        self.widget_2.setStyleSheet("border-style: solid;\n"
"border-width: 2px;\n"
"border-color:  rgb(40,40,40);")
        self.widget_2.setObjectName("widget_2")
        self.mlp_general.setSizePolicy(sizePolicy)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.widget_2)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.horizontalLayout_2.addWidget(self.widget_2)
        self.label_conclusion = QtWidgets.QLabel(self.mlp_student)
        self.label_conclusion.setStyleSheet("border-style: solid;\n"
"border-width: 2px;\n"
"border-color:  rgb(40,40,40);")
        self.label_conclusion.setText("")
        self.label_conclusion.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_conclusion.setObjectName("label_conclusion")
        self.horizontalLayout_2.addWidget(self.label_conclusion)
        self.verticalLayout.addWidget(self.mlp_student)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1073, 29))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        MainWindow.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        MainWindow.setWindowIcon(QtGui.QIcon('icon.png'))
        self.pushButton.setText(_translate("MainWindow", "Show statstics"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_GraphsWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

