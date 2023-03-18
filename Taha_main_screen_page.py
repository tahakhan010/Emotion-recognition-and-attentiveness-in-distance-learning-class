# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 13:09:19 2021

@author: M Taha khan
"""
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2, sys, os
import numpy as np
from mss import mss
from PIL import Image
from time import time
import pyautogui
import math, dlib

### face and emotion recognition
from imutils import face_utils
import face_recognition
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import QColor, QIcon, QMovie
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QWidget, QToolTip
from PyQt5.uic import loadUi
from PyQt5.QtCore import (QCoreApplication, QThread,
                          QThreadPool, pyqtSignal)




from screen_page import Ui_MainWindow




#global variable
btncamera = False
checkthreadcamera = '' # fill with object camera thread to find out that is running or not
emotion_list=[]
filename = ""


class MainWindow(QMainWindow, Ui_MainWindow):
    
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.ui.start_button.clicked.connect(self.load_file)

    
    def load_file(self):
        global filename, btncamera, checkthreadcamera,emotion_list
        
        try:
            if btncamera: #when the button is in stop mode
    
                self.ui.start_button.setText('Load Video')
                checkthreadcamera.stop()
                print("stopped")
                print(emotion_list)
                btncamera=False
                # print(btncamera)
                
            else:
                btncamera = True

                filename = QFileDialog.getOpenFileName(self, 'Open', "", "Video(*.mp4)")[0]
                if filename == '':
                    #ERROR - No file selected
                    pass  
                            
                self.ui.start_button.setText('Stop video')
                
                self.cameramode = camera_mode_thread()
                print("created camera_mode_thread")
                
                self.cameramode.set_video_camera.connect(self.set_video_func)
                print('connect thread to set_video_func')
                
                self.cameramode.start()
                print("start the thread")         
                
                checkthreadcamera = self.cameramode
                print('running')
                
                #### add stats when button clicked here
                
        except:
            print("error in start/stop button")
    
       
        
        return filename
            
        
            #### set the frame of camera coming from thread of camera to the Qlabel in Gui                                 
    def set_video_func(self,pixmap4):
        global w_camera_label, h_camera_label
        
        ## dynamically resizable ##
        w_camera_label = self.ui.camera_label.width()
        h_camera_label = self.ui.camera_label.height()
        pixmap4 = pixmap4.scaled(w_camera_label, h_camera_label, QtCore.Qt.KeepAspectRatio) 
        
        self.ui.camera_label.setPixmap(pixmap4) # add frames of camera to Qlabel
    
    

class camera_mode_thread(QThread):
    
    set_video_camera = pyqtSignal(object)
    print('signal set_video cam')
    
    def stop(self):
        #self.video_capture.release() #when thread stop, so is camera
        print("stop self working")
        self.terminate()
    

    def run(self):
        global btncamera, checkthreadcamera, emotion_list, w_camera_label, h_camera_label,filename
        global eye_cascade, distract_model
        global known_face_encodings, known_face_names
        
        get_faces()
       
        self.video_capture = cv2.VideoCapture(filename)
        
        print("def run starting")

        #put your model path etc
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')                         #keras eye
        distract_model = load_model('distraction_model.hdf5', compile=False)
        classifier =load_model('model.h5')
        
        while True:
                    # Capture frame-by-frame
            ret, frame = self.video_capture.read()

           
            frame = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)
            print("reading frame from webcam")
        
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
######## FUNCTION EMOTION RECOGNTION ######################        
            emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']        
            emotion_list=[]
            face_names = []
            
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE)

            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
                roi_gray = gray[y:y+h,x:x+w]               
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        
                trackEyeMovements(x,y,w,h, roi_gray, frame) # Eye tracking function
                
                single_face_encoding = face_recognition.face_encodings(frame, known_face_locations=[(x, w, h, y)])[0]
                print("face location and encoding done")
        
                # See if the face is a match for the known face
                matches = face_recognition.compare_faces(known_face_encodings, single_face_encoding)
                name = "Unknown"
                
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, single_face_encoding)
                print("distance calculated")
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    print('good match')
        
                face_names.append(name)
                print("face found")
                print(name)
                
                if np.sum([roi_gray])!=0:
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi,axis=0)
        
                    prediction = classifier.predict(roi)[0]
                    label=emotion_labels[prediction.argmax()]
                    label_position = (x,y)
                    
                    # emotion_face = label + " " + name
                    cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

                else:
                    cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                    emotion_list.append("No face")
        
        
            # Display the results
            for (x,y,w,h), name in zip(faces, face_names):
                # cv2.putText(frame, name, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 2)
                # emotion_face = label + " " + name
                cv2.putText(frame,name,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)             
                
########### create ratio for image lable in GUI  ############################  
            imageOut = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
            pixmap4 = QtGui.QPixmap(imageOut)
            self.set_video_camera.emit(pixmap4)
            print("emitting the frame")
            

            if not btncamera: # break thread when cliked on btn stop
                
                break
            
        self.video_capture.release()
        
def get_faces():
    global known_face_encodings, known_face_names
    
    known_face_encodings=[]
    known_face_names = []
    dirName="images"
    jpegs=os.listdir(dirName)
    for imface in jpegs:
        #print(imface[:-4])
        path="{}\\{}".format(dirName,imface)
        imageface=face_recognition.load_image_file(path)
        im_encoding=face_recognition.face_encodings(imageface)[0]
        known_face_encodings.append(im_encoding)
        known_face_names.append(imface[:-5])
    print("Faces loaded: ", len(known_face_encodings))

####################   EYE-MOVEMENT ###############################
def trackEyeMovements(x,y,w,h, roi_gray, frame):
    global eye_cascade, distract_model
    roi_color = frame[y:y+h, x:x+w]
    
    eyes = eye_cascade.detectMultiScale(
        roi_gray, 
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60,60)
        )

    # init probability list for each eye prediction
    probs = list()

    # loop through detected eyes
    for (ex,ey,ew,eh) in eyes:
        # draw eye rectangles
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        # get colour eye for distraction detection
        roi = roi_color[ey+2:ey+eh-2, ex+2:ex+ew-2]
        # match CNN input shape
        roi = cv2.resize(roi, (64, 64))
        # normalize (as done in model training)
        roi = roi.astype("float") / 255.0
        # change to array
        roi = img_to_array(roi)
        # correct shape
        roi = np.expand_dims(roi, axis=0)

        # distraction classification/detection
        prediction = distract_model.predict(roi)
        # save eye result
        probs.append(prediction[0])

    # get average score for all eyes
    probs_mean = np.mean(probs)

    # get label
    if probs_mean < 0.5:
        label = 'distracted'
    else:
        label = 'focused'
    
    cv2.putText(frame,label,(x-6,y-25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

        
def exitthread():
    global btncamera, table
    if btncamera:
        checkthreadcamera.stop()
        print("exit checkthreadcamera")

        
if __name__ == '__main__':
    app=QApplication(sys.argv)
    app.aboutToQuit.connect(exitthread)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())