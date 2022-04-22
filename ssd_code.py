import numpy as np
import time
import cv2

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
"sofa", "train", "tvmonitor"]
weights ='MobileNetSSD_deploy.caffemodel'

# Load the Model
arch= 'MobileNetSSD_deploy.prototxt.txt'
net = cv2.dnn.readNetFromCaffe(arch, weights)


###########################################################################################

threshold = 0.6
cap=cv2.VideoCapture(0)
w  = cap.get(3)
h = cap.get(4)


while(True):

    t1=time.time()
    ret,frame=cap.read()
    blob = cv2.dnn.blobFromImage(frame,  0.007843,(320,320), (127.5,127.5,127.5),True)

    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # print(detections)

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
                label = "{}: {:.2f}%".format(CLASSES[index], confidence * 100)
                cv2.rectangle(frame, (x_1, y_1), (x_2, y_2), (0,0,255), 4)
                cv2.putText(frame, label, (x_1, y_1), cv2.FONT_ITALIC, 2, (0,0,255), 4)
                print('person found')
                t2=time.time()
                # print('Time: ',t2-t1)
            else:
                print('Person Not Found')
                pass


    cv2.imshow("Image",frame)
    k = cv2.waitKey(1)   
    if k == ord('q'):
        break

cap.release() 
cv2.destroyAllWindows() 

