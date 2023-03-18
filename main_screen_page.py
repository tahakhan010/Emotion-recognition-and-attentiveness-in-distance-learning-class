#from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2, sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from datetime import datetime

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

# ###segmentation
import imutils
import scipy.ndimage as ndi

from skimage import measure
import matplotlib.patches as mpatches

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
from dashboard import Ui_GraphsWindow


#global variable
btncamera = False
checkthreadcamera = '' # fill with object camera thread to find out that is running or not
emotion_list=[]
filename = ""
vis_dict = {'D': 0, 'F': 0}
state_list = []
csv_file = ""
b_Canvas = False
dist_Canvas = FigureCanvas(Figure())

class MainWindow(QMainWindow, Ui_MainWindow):
    switch_graphs = QtCore.pyqtSignal()
    back_to_login = QtCore.pyqtSignal(str)
    
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.ui.start_button.clicked.connect(self.load_file)

    
    def load_file(self):
        global filename, btncamera, checkthreadcamera,emotion_list, csv_file
        
        try:
            if btncamera: #when the button is in stop mode
    
                self.ui.start_button.setText('Load Video')
                checkthreadcamera.stop()
                print("stopped")
                print(emotion_list)
                #### add stats when button clicked here
                try:
                    # self.ui.start_button.connect(self.change_graphs)
                    self.change_graphs()
                    self.hide()
                    # self.change_graphs
                except:
                    print('error in windows')
                                                
                btncamera=False
                self.ui.setTitle(self,"Facial Recognition with Emotion/Eye Detection")
                # print(btncamera)
                
            else:
                btncamera = True

                filename = QFileDialog.getOpenFileName(self, 'Open', "", "Video(*.mp4)")[0]
                if filename == '':
                    #ERROR - No file selected
                    pass
                
                time_now = datetime.now().strftime('%Y-%m-%d_%H-%M')
                csv_file = "PersonEmotions_" + time_now +".csv"
                
                self.ui.start_button.setText('Stop video')
                self.ui.setTitle(self,"Playing File: " + filename)
                
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
            
    def change_graphs(self):
        self.switch_graphs.emit()        
        
            #### set the frame of camera coming from thread of camera to the Qlabel in Gui                                 
    def set_video_func(self,pixmap4):
        global w_camera_label, h_camera_label
        
        ## dynamically resizable ##
        w_camera_label = self.ui.camera_label.width()
        h_camera_label = self.ui.camera_label.height()
        pixmap4 = pixmap4.scaled(w_camera_label, h_camera_label, QtCore.Qt.KeepAspectRatio) 
        
        self.ui.camera_label.setPixmap(pixmap4) # add frames of camera to Qlabel
    
class GraphsWindow(QMainWindow, Ui_GraphsWindow):
    global vis_dict
    global value_series, csv_file
    
    switch_graphs = QtCore.pyqtSignal(str)
    back_to_login = QtCore.pyqtSignal()
    
    def __init__(self, parent=None):
        super(GraphsWindow, self).__init__(parent)
        QMainWindow.__init__(self, parent)
        self.ui = Ui_GraphsWindow()
        self.ui.setupUi(self)
        
        self.ui.pushButton.clicked.connect(self.show_plot)
        self.ui.pushButton.clicked.connect(self.show_plot_distraction)
        self.ui.pushButton.clicked.connect(self.show_conclusion)
        
        global b_Canvas, datafile, headers
        
        
        ########csv_file = 'PersonEmotions_2022-01-10_05-15.csv'
        datafile = pd.read_csv(csv_file)
        with open(csv_file, 'r') as f:
            d_reader = csv.DictReader(f)
            headers = d_reader.fieldnames
            
        for i in range(len(headers)):
            self.ui.comboBox.addItem(headers[i])
        # print(headers) ##names of people
        #nameslen = len(headers)
        
        # self.show_plot_distraction()
        
    def show_conclusion(self):
        global value_series, var, gen_emotion
        
        if value_series[1]>70:
            sentence_class = "Successful class, students were focused " + str(value_series[1])
        else:
            sentence_class = "Be careful, the students were not focused on the class " + str(value_series[1])
            
        sentence_student = "\n The general emotion of " + str(var) + " is " + str(gen_emotion)
        
        sentences = sentence_class + "\n" + sentence_student
            
        
        self.ui.label_conclusion.setText(sentences)
        
    def rm_mpl(self):
        global b_Canvas
        self.ui.horizontalLayout.removeWidget(self.canvas)
        self.canvas.close()

        b_Canvas = False 
        
    def show_plot(self):
        global b_Canvas, b_Canvas2, datafile, number, var, headers, number_emotions, gen_emotion
        

        var = self.ui.comboBox.currentText()
        # print(var) #name of student selected
        number = headers.index(var)
        
        if b_Canvas == True :
            self.rm_mpl()
            
       
        fig = Figure()
        self.canvas = FigureCanvas(fig)
        self.ui.horizontalLayout.addWidget(self.canvas)        
        ax1f2 = fig.add_subplot(122)
        
    
        ##########emotions for one student
        data=datafile[headers[number]]
        
        datacount = data.value_counts()
        
        gen_emotion  = datacount.idxmax() ###general emotion

        count_row = datafile.shape[0] #total of emotions
         
        name_emotions= datacount.index.values ##name of all emotions
        # print(name_emotions)
        list_datacount = datacount.values ## number of times each emotions
        percentemotion = [round(i*100/count_row) for i in list_datacount] ## percentage of emotion states
        
        # print("number of emotions", len(list_datacount))        
        
        
        width = 0.8
        ind = np.arange(0,len(list_datacount))
    
        # ax1f1.plot(value_series)
        for i in range(len(percentemotion)):
            ax1f2.bar(i,percentemotion[i],width)
    
    
    
        ax1f2.set_title("Representation of emotion of "+ str(var) + " in Percentage")
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
    
       ###general emotions of all student    #######################
        ax1f1 = fig.add_subplot(121)
       
        filtered_column_names = datafile.any(axis=0)
        subset = datafile[datafile.columns[filtered_column_names]]
        number_emotions = subset.stack().value_counts()
        # print("count per emotion", number_emotions)
        name_number_emotions = number_emotions.index.values
        # print(name_number_emotions)
        list_number_emotions = number_emotions.values ###only the counts of emotions in a list for the %
        emotionNum = datafile.shape[0]*datafile.shape[1]
        # print("Total number of emotions", emotionNum)
        percentemotionall = [round(i*100/emotionNum) for i in list_number_emotions] ## percentage of emotion states

        ind2 = np.arange(0,len(list_number_emotions))
    
        for i in range(len(percentemotionall)):
            ax1f1.bar(i,percentemotionall[i],width)

        ax1f1.set_title("Representation of emotion of Students in Percentage")
        ax1f1.set_xlabel("Emotions")
        ax1f1.set_ylabel("Percentage %")
        ax1f1.set_xticks(ind2,minor=False)
        ax1f1.set_xticklabels(name_number_emotions, fontdict=None, minor=False)
    
        rects = ax1f1.patches
    
        for rect, name_number_emotions in zip(rects, percentemotionall):
            height = rect.get_height()
            ax1f1.text(
                rect.get_x() + rect.get_width() / 2, height + 1, name_number_emotions, ha="center", va="bottom"
            )                    
        
        self.canvas.draw()
        b_Canvas = True

    def show_plot_distraction(self):
        global vis_dict, value_series, dist_Canvas
        self.ui.horizontalLayout_4.removeWidget(dist_Canvas)
        total = sum(vis_dict.values()) if sum(vis_dict.values()) != 0 else 1
        
        for k, v in vis_dict.items():
            vis_dict[k] = round((v/total)*100, 2)
            
        keys = vis_dict.keys();
        values = vis_dict.values()
        
        value_series = pd.Series(values)
        
        # Plot the figure.
        ax1f1 = dist_Canvas.figure.add_subplot(121)
        
        # fig.subplots_adjust(top=0.96, bottom = 0.08, left=0.045, right=0.99, hspace = 0.4)

        # axvalue = value_series.plot(kind="bar")
        
        width = 0.8
        ind = np.arange(1,3,1)
        labels = ["Distracted","Focused"]
        # ax1f1.plot(value_series)
        ax1f1.bar(1,value_series[0],width)
        ax1f1.bar(2,value_series[1],width)

        ax1f1.set_title("Representation of State of Students in Percentage")
        ax1f1.set_xlabel("State of Student")
        ax1f1.set_ylabel("Percentage %")
        ax1f1.set_xticks(ind,minor=False)
        ax1f1.set_xticklabels(labels, fontdict=None, minor=False)

        rects = ax1f1.patches

        listpercent = [str(value_series[0]),str(value_series[1])]
        for rect, label in zip(rects, listpercent):
            height = rect.get_height()
            ax1f1.text(
                rect.get_x() + rect.get_width() / 2, height + 1, label, ha="center", va="bottom"
            )
        self.ui.horizontalLayout_4.addWidget(dist_Canvas)
        dist_Canvas.show()


class camera_mode_thread(QThread):
    
    set_video_camera = pyqtSignal(object)
    
    print('signal set_video cam')
    
    def stop(self):
        #self.video_capture.release() #when thread stop, so is camera
        print("stop self working")
        self.terminate()
        #plot_graph()
        saveToCSV()
    

    def run(self):
        global btncamera, checkthreadcamera, emotion_list, w_camera_label, h_camera_label,filename
        global eye_cascade, distract_model, vis_dict
        global known_face_encodings, known_face_names
        
        get_faces()
       
        self.video_capture = cv2.VideoCapture(filename)
        
        print("def run starting")

        #put your model path etc
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml') #keras eye
        distract_model = load_model('distraction_model.hdf5', compile=False)
        classifier = load_model('model_v6.hdf5', compile=False)
        emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}
        
        while True:
                    # Capture frame-by-frame
            ret, frame = self.video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            print("reading frame from webcam")
        
            
######## FUNCTION EMOTION RECOGNTION ######################        
            person_emotion_dict = dict.fromkeys(known_face_names)
            for key in person_emotion_dict:
                person_emotion_dict[key] = 'Neutral'
                
            fast_frame = frame #cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)
            rgb_frame = fast_frame[:, :, ::-1] # convert to RBG from BGR
            
            # face_locations = ROI
            face_locations = face_recognition.face_locations(rgb_frame)
            
            # face_locations = ROI
            for face_location in face_locations:
                top, right, bottom, left = face_location # four points of ROI
                
                 # Modified ROI with exact face only
                face_image = rgb_frame[top:bottom, left:right]
                
                try:                
                    face_encoding = face_recognition.face_encodings(face_image)[0]
                except IndexError: # only when no face is found
                    pass
                    continue
            
                # Or instead, use the known face with the smallest distance to the new face
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                # Get best matching face
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    print('good match found: ', name)
                
                # Used for Emotion detection (convert to 48x48, to Grayscale, etc)
                face_image = cv2.resize(face_image, (48,48))
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
                
                # Emotion detection
                label_map = dict((v,k) for k,v in emotion_dict.items()) 
                predicted_class = np.argmax(classifier.predict(face_image))
                predicted_label = label_map[predicted_class] # get index matching the key.
                
                if name != "Unknown":
                    person_emotion_dict[name] = predicted_label
                
                label_position = (int(left)-10,int(top)-15)
                name_label_position = (int(left),int(bottom))
                
                cv2.rectangle(frame,label_position,(int(right)+10, int(bottom)+10),(0,255,0),2) # Rectangle
                cv2.putText(frame,predicted_label,label_position,cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2) # Emotion
                cv2.putText(frame,name,name_label_position,cv2.FONT_HERSHEY_SIMPLEX,0.95,(20,200,60),2) # Face
                
                roi_color = frame[top:bottom, left:right]
                roi_gray = gray[top:bottom, left:right]
                
                # Eye tracking function
                label = trackEyeMovements(roi_gray, roi_color)
                cv2.putText(frame,label,(int(left)-10,int(top)-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                

            write_person_emotion_to_csv(person_emotion_dict)
            
########### create ratio for image lable in GUI  ############################  
            imageOut = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
            pixmap4 = QtGui.QPixmap(imageOut)
            self.set_video_camera.emit(pixmap4)
            print("emitting the frame")

            if not btncamera: # break thread when cliked on btn stop
                
                break
            
        self.video_capture.release()
        
        
def saveToCSV():
    global state_list
    filename = "state.csv"
    df = pd.DataFrame(state_list, columns=['Distracted','Focused'])
    df.to_csv(filename)
        
def write_person_emotion_to_csv(dictionary):
    global csv_file
    file_exists = os.path.isfile(csv_file)
    try:
        with open(csv_file, 'a', newline='') as out_file:
            writer = csv.DictWriter(out_file, fieldnames=dictionary.keys(), dialect='excel')
            if not file_exists:
                writer.writeheader()
            writer.writerow(dictionary)
    except IOError:
           print("I/O error")

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
def trackEyeMovements(roi_gray, roi_color):
    global eye_cascade, distract_model
    
    eyes = eye_cascade.detectMultiScale(roi_gray)

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

    # get average score for all eyes
    probs_mean = np.mean(probs)
    
    # get label
    if probs_mean <= 0.6:
        label = 'Distracted'
        vis_dict['D'] += 1
        state_list.append([0,1])
        
        
    else:
        label = 'Focused'
        vis_dict['F'] += 1
        state_list.append([1,0])
        
    return label
    
        
def exitthread():
    global btncamera, table
    if btncamera:
        checkthreadcamera.stop()
        print("exit checkthreadcamera")


class Controller:

    def __init__(self):
        pass

    # Login Window
    def show_login(self):
        self.login = MainWindow()
        self.login.switch_graphs.connect(self.show_about)
        self.login.show()



    # About us Window
    def show_about(self):
        self.about = GraphsWindow()
        # self.about.back_to_login.connect(self.back_login)
        # self.login.close()
        self.about.show()

    # def back_login(self):
    #     self.about.close()
    #     self.login.show()

 
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.aboutToQuit.connect(exitthread)
    controller = Controller()
    controller.show_login()
    sys.exit(app.exec_())
       
if __name__ == '__main__':
    main()