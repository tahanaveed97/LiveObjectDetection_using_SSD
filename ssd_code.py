import numpy as np
import time
import cv2
from tkinter import filedialog
from tkinter import Tk
import matplotlib.pyplot as plt
import random
import os


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
"sofa", "train", "tvmonitor"]
print('Total Number of classes are: {} \n'.format(len(CLASSES)-1))

# Randomize with same pattern each time.
np.random.seed(50)

# Create the random list of colors of required size.
COLORS = np.random.uniform(0, 255, size=(len(CLASSES),3))

# Load the Model Weights.
weights ='MobileNetSSD_deploy.caffemodel'

# Load the googleNet Architecture.
arch= 'MobileNetSSD_deploy.prototxt.txt'

# Initialize the network.
net = cv2.dnn.readNetFromCaffe(arch, weights)


###########################################################################################

threshold = 0.3
cap=cv2.VideoCapture(0)
w  = cap.get(3)
h = cap.get(4)

while(True):
    start_time=time.time()
    ret,frame=cap.read()
    blob = cv2.dnn.blobFromImage(frame,  0.007843,(128, 128), (127.5,127.5,127.5),True)

    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter Out weak detections
        if confidence > threshold:
            # Extracting the index of the class.
            index = int(detections[0, 0, i, 1])

            # Extract the (x1,x2,y1,y2) coordinates of the bounding box for each obejct 
            x_1 = int(detections[0, 0, i, 3] * w)
            y_1 = int(detections[0, 0, i, 4] * h)
            x_2 = int(detections[0, 0, i, 5] * w)
            y_2 = int(detections[0, 0, i, 6] * h)

            if index==15:
                # Display the target class name and their confidence score.
                label = "{}: {:.2f}%".format(CLASSES[index], confidence * 100)
                # Draw the rectangle on object w.r.t their cordinates. 
                cv2.rectangle(frame, (x_1, y_1), (x_2, y_2), COLORS[index], 4)

                # Put the Label of the detected object 
                cv2.putText(frame, label, (x_1, y_1), cv2.FONT_ITALIC, 1.3, COLORS[index], 3)


    fps= (1.0 / (time.time() - start_time))
    cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (400, 20), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 20, 55), 2)
    cv2.imshow("Image",frame)
    
    k = cv2.waitKey(1)   
    if k == ord('q'):
        break

cap.release() 
cv2.destroyAllWindows() 

