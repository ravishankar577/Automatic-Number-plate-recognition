# Importing the necessary libraries for the program to run.
from ai.ai_model import load_yolov5_model
from ai.ai_model import detection

from helper.params import Parameters
from helper.general_utils import filter_text
from helper.general_utils import save_results

from ai.ocr_model import easyocr_model_load
from ai.ocr_model import easyocr_model_works
from utils.visual_utils import *
import moviepy.editor as moviepy

import cv2
from datetime import datetime

# Loading the parameters from the params.py file.
params = Parameters()

canvas = None
drawing = False # true if mouse is pressed

#Retrieve first frame
def initialize_camera(cap):
    _, frame = cap.read()
    return frame


# mouse callback function
def mouse_draw_rect(event,x,y,flags, params):
    global drawing, canvas

    if drawing:
        canvas = params[0].copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        params.append((x,y)) #Save first point

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.rectangle(canvas, params[1],(x,y),(0,255,0),2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        params.append((x,y)) #Save second point
        cv2.rectangle(canvas,params[1],params[2],(0,255,0),2)


def select_roi(frame):
    global canvas
    canvas = frame.copy()
    params = [frame]
    ROI_SELECTION_WINDOW = 'Select ROI'
    cv2.namedWindow(ROI_SELECTION_WINDOW)
    cv2.setMouseCallback(ROI_SELECTION_WINDOW, mouse_draw_rect, params)
    roi_selected = False
    while True:
        cv2.imshow(ROI_SELECTION_WINDOW, canvas)
        key = cv2.waitKey(10)

        #Press Enter to break the loop
        if key == 13:
            break


    cv2.destroyWindow(ROI_SELECTION_WINDOW)
    roi_selected = (3 == len(params))

    if roi_selected:
        p1 = params[1]
        p2 = params[2]
        if (p1[0] == p2[0]) and (p1[1] == p2[1]):
            roi_selected = False

    #Use whole frame if ROI has not been selected
    if not roi_selected:
        print('ROI Not Selected. Using Full Frame')
        p1 = (0,0)
        p2 = (frame.shape[1] - 1, frame.shape[0] -1)


    return roi_selected, p1, p2


if __name__ == "__main__":
    
    

    # Loading the model and labels from the ai_model.py file.
    model, labels = load_yolov5_model()
    # Capturing the video from the webcam.
    file = './112.AVI'
    if (file.endswith(".avi") or  file.endswith(".AVI")):
        clip = moviepy.VideoFileClip(file)
        clip.write_videofile("myvideo.mp4")
        file = './myvideo.mp4'
        
    

    cap = cv2.VideoCapture(file)
    
     #Grab first frame
    first_frame = initialize_camera(cap)

    #Select ROI for processing. Hit Enter after drawing the rectangle to finalize selection
    roi_selected, point1, point2 = select_roi(first_frame)    

    #Grab ROI of first frame
    first_frame_roi = first_frame[point1[1]:point2[1], point1[0]:point2[0], :]
    # Loading the model for the OCR.
    text_reader = easyocr_model_load()

    while 1:

        # Reading the video from the webcam.
        ret, frame = cap.read()
        

        if ret:
            
            try:
            
                #ROI of current frame
                roi = frame[point1[1]:point2[1], point1[0]:point2[0], :]
        
            except:
                roi = frame

            # Detecting the text from the image.
            # detected, _ = detection(frame, model, labels)
            try:
                detected, _ = detection(roi, model, labels)
            except:
                detected, _ = detection(frame, model, labels)
            # Reading the text from the image.
            resulteasyocr = text_reader.readtext(
                detected
            )  # text_read.recognize() , you can use cropped plate image or whole image
            # Filtering the text from the image.
            text = filter_text(params.rect_size, resulteasyocr, params.region_threshold)
            # Saving the results of the OCR in a csv file.
            # save_results(text[-1], "ocr_results.csv", "Detection_Images")
            print("TEXT",text)
            cv2.imshow("detected", detected)

        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            break
