# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 23:33:16 2022

@author: laudi
"""

from datetime import datetime

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

# ###segmentation
import imutils, sys
import scipy.ndimage as ndi
import numpy as np

from skimage import measure
import matplotlib.patches as mpatches


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import QColor, QIcon, QMovie
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QWidget, QToolTip
from PyQt5.uic import loadUi
from PyQt5.QtCore import (QCoreApplication, QThread,
                          QThreadPool, pyqtSignal)

from screen_page import Ui_MainWindow
from dashboard import Ui_GraphsWindow


import csv
import pandas as pd

# filename = 'PersonEmotions_2022-01-10_05-15.csv'
# datafile = pd.read_csv(filename)

# with open(filename, 'r') as f:
#     d_reader = csv.DictReader(f)
#     headers = d_reader.fieldnames

# # print(headers)
# nameslen = len(headers)

# data=datafile[headers[0]]

# # datafile['count'] = datafile[headers[0]].str.count('Neutral')
# # print (datafile['count'])

# datacount = data.value_counts()
# count_row = datafile.shape[0] #total of emotions
# list_datacount = datacount.values ##list of times emotions

# # print(datacount.index.values) ##name of emotion

# #print(datacount)

# percent1 = [i*100/nameslen for i in list_datacount]

# # ax = df.plot.bar(x='lab', y='val', rot=0)

b_Canvas = False

class GraphsWindow(QMainWindow, Ui_GraphsWindow):
    global csv_file
    
    switch_graphs = QtCore.pyqtSignal(str)
    back_to_login = QtCore.pyqtSignal()
    
    
    
    def __init__(self, parent=None):
        super(GraphsWindow, self).__init__(parent)
        QMainWindow.__init__(self, parent)
        self.ui = Ui_GraphsWindow()
        self.ui.setupUi(self)
        
        self.ui.pushButton.clicked.connect(self.show_plot)
        
        global b_Canvas, datafile, headers
        
        
        csv_file = 'PersonEmotions_2022-01-10_05-15.csv'
        datafile = pd.read_csv(csv_file)
        with open(csv_file, 'r') as f:
            d_reader = csv.DictReader(f)
            headers = d_reader.fieldnames
            
        for i in range(len(headers)):
            self.ui.comboBox.addItem(headers[i])
        # print(headers) ##names of people
        #nameslen = len(headers) 
        
        
        
    def rm_mpl(self):
        global b_Canvas
        self.ui.verticalLayout.removeWidget(self.canvas)
        self.canvas.close()

        b_Canvas = False          
   
    def show_plot(self):
        global b_Canvas, datafile, number, var, headers
        

        var = self.ui.comboBox.currentText()
        print(var)
        number = headers.index(var)
        
        if b_Canvas == True:
            self.rm_mpl()
            
       
        fig = Figure()
        self.canvas = FigureCanvas(fig)
        self.ui.verticalLayout.addWidget(self.canvas)        
        ax1f2 = fig.add_subplot(122)
        
        data=datafile[headers[number]]
        
        datacount = data.value_counts()
        count_row = datafile.shape[0] #total of emotions
         
        name_emotions= datacount.index.values ##name of all emotions
        print(name_emotions)
        list_datacount = datacount.values ## number of times each emotions
        percentemotion = [round(i*100/count_row) for i in list_datacount] ## percentage of emotion states
        
        print("number of emotions", len(list_datacount))        
        
        
        width = 0.8
        ind = np.arange(0,len(list_datacount))
    
        # ax1f1.plot(value_series)
        for i in range(len(percentemotion)):
            ax1f2.bar(i,percentemotion[i],width)
    
    
    
        ax1f2.set_title("Representation of emotion of Students in Percentage")
        ax1f2.set_xlabel("Emotions of " + str(var))
        ax1f2.set_ylabel("Percentage %")
        ax1f2.set_xticks(ind,minor=False)
        ax1f2.set_xticklabels(name_emotions, fontdict=None, minor=False)
    
        rects = ax1f2.patches
    
        for rect, name_emotions in zip(rects, percentemotion):
            height = rect.get_height()
            ax1f2.text(
                rect.get_x() + rect.get_width() / 2, height + 1, name_emotions, ha="center", va="bottom"
            )            
            
    
        self.canvas.show()
        b_Canvas = True
            


if __name__ == '__main__':
    app=QApplication(sys.argv)
    # app.aboutToQuit.connect(exitthread)
    main = GraphsWindow()
    main.show()
    sys.exit(app.exec_())