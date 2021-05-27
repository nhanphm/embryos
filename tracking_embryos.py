# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import libraries of python OpenCV  
import cv2
import numpy as np
import time
import os
import sys

# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations

video_path = 'D:\output18.mp4'


"""
Tracker type
"""
tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[2]

if tracker_type == 'BOOSTING':
    tracker_class = cv2.TrackerBoosting_create
elif tracker_type == 'MIL':
    tracker_class = cv2.TrackerMIL_create
elif tracker_type == 'KCF':
    tracker_class = cv2.TrackerKCF_create
elif tracker_type == 'TLD':
    tracker_class = cv2.TrackerTLD_create
elif tracker_type == 'MEDIANFLOW':
    tracker_class = cv2.TrackerMedianFlow_create
# elif tracker_type == 'GOTURN':
#     tracker = cv2.TrackerGOTURN_create()
elif tracker_type == 'MOSSE':
    tracker_class = cv2.TrackerMOSSE_create
elif tracker_type == "CSRT":
    tracker_class = cv2.TrackerCSRT_create


"""
Yolo config
"""
#Define weight path for Yolo
weight_path =   "D:\yolo_weight\yolov3-tiny_final.weights"
cfg_path =      "D:\yolo_weight\yolov3-tiny.cfg"
class_path =    "D:\yolo_weight\classes.names"

# Load Yolo
net = cv2.dnn.readNet(weight_path, cfg_path)
classes = []
with open(class_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))



"""
Laplacian setup
"""
#Define function check blur value
def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

# threshold Laplacian method
threshold = 1500
padding = 20
# initialize OpenCV's special multi-object tracker
def out_of_range(box, width, height):
    _center_x = int(box[0]) + int(box[2])/2
    _center_y = int(box[1]) + int(box[3])/2
    if _center_x < padding or _center_x > (width-padding):
        return False
    if _center_y < padding or _center_y > (height-padding):
        return False
    return True
                                       
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(int(boxA[0]), int(boxB[0]))
	yA = max(int(boxA[1]), int(boxB[1]))
	xB = min(int(boxA[0])+int(boxA[2]), int(boxB[0])+int(boxB[2]))
	yB = min(int(boxA[1])+int(boxA[3]), int(boxB[1])+int(boxB[3]))
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (int(boxA[2]) + 1) * (int(boxA[3]) + 1)
	boxBArea = (int(boxB[2]) + 1) * (int(boxB[3]) + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

trackers = cv2.MultiTracker_create()
tracking = False    # disable/ enable Yolo detect
label = ""          # label of embryos
fp = 0              # order of frames
emb_count = 0       # order of embryos
n = 0       # Số lượng phôi mỗi lần tracking
m = 0       # Số lượng frame ko chứa phôi liên tục
v = 0       # Số lượng frame tracking liên tục

max_blurry = threshold
"""
Init Video capture
"""
# if a video path was not supplied, grab the reference to the web cam
video = cv2.VideoCapture(video_path)
if not video.isOpened():
    print("Could not open video")
    sys.exit()

# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    ok, frame = video.read()
    fp +=1
    # check to see if we have reached the end of the stream
    if ok is None or frame is None:
        break

    # resize the frame (so we can process it faster)
    frame = cv2.resize(frame, (416, 416))
    
    height, width, channels = frame.shape
    # load the image, convert it to grayscale, and compute the
	# focus measure of the image using the Variance of Laplacian
	# method    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    if fm < threshold or fm > 2000:
        text = "Blurry"
    else:
        text = "Not Blurry"
        if tracking == False or fp%10==0:
            # Detecting objects with YOLO
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            font = cv2.FONT_HERSHEY_PLAIN
            
            if len(indexes) > 0:
                bol = True
                if tracking == False:
                    trackers = cv2.MultiTracker_create()
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        color = colors[i-1]
                        #cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        label = "{}: {:.4f}".format(label, confidences[i])
                        #cv2.putText(frame, label, (x, y - 5), font, 1, color, 2)
                        # create a new object tracker for the bounding box and add it
                        # to our multi-object tracker
                        box = (x,y, w, h)
                        
                        if tracking == True:
                            success, _boxes = trackers.update(frame)
                            if success:
                                for count, bbox in enumerate(_boxes):
                                    bb = bb_intersection_over_union(box, bbox)
                                    if bb> 0.3:
                                        bol = False
                        if bol == True:   
                            tracker = tracker_class()
                            trackers.add(tracker, frame, box)
                            #n+=1
                            tracking = True
                        
                        
                        
    #header text
    cv2.putText(frame,  "{}: {:.2f}".format(text, fm), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
    
    
    # grab the updated bounding box coordinates (if any) for each
    # object that is being tracked
    success, boxes = trackers.update(frame)

    # loop over the bounding boxes and draw then on the frame
    if success:
        for count, bbox in enumerate(boxes):
            # Tracking success
            if fm > max_blurry:
                max_blurry = fm
                img_name = "D:\embryos_{}.png".format(emb_count+count+1)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
            
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            cv2.putText(frame, "{}: {}".format(emb_count+count+1, label), (int(bbox[0]), int(bbox[1])-5), font, 1, color, 2)
            m = 0
            v+=1    # Dem so frame tracking lien tuc
            n = max(count+1, n)
            # Check blurry and get frame
           
    else:
        # Tracking failure
        m+=1    # Dem so frame khong chua phoi lien tuc
        if v>70: 
            emb_count +=n    
            n=0
            max_blurry = threshold
        if m>10:
            tracking = False
        v=0
        #trackers = cv2.MultiTracker_create()
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        box = cv2.selectROI("Frame", frame, fromCenter=False,
                            showCrosshair=True)

        # create a new object tracker for the bounding box and add it
        # to our multi-object tracker
        tracker = tracker_class()
        trackers.add(tracker, frame, box)

    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break

# if we are using a webcam, release the pointer
video.release()

# close all windows
cv2.destroyAllWindows()