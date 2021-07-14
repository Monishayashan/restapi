#!/usr/bin/env python
# coding: utf-8

# 
# In[137]:
def mask(): 
    import cv2
    import numpy as np 
    net_mask = cv2.dnn.readNet("yolov3_mask_last.weights", "yolov3_mask.cfg")
    classes_mask = []
    with open("coco -1.names", "r") as f:
        classes_mask = [line.strip() for line in f.readlines()]  
        layer_names_mask = net_mask.getLayerNames()
        output_layer_mask = [layer_names_mask[i[0] - 1] for i in net_mask.getUnconnectedOutLayers()]
        while True:
            re,img = video_capture.read()
            img = cv2.resize(img, None, fx=0.4, fy=0.4)
            height, width, channels = img.shape
            blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),  swapRB=True, crop=False)
            net_mask.setInput(blob)
            outs = net_mask.forward(output_layer_mask)
            class_ids_mask = []
            confidences_mask = []
            boxes_mask = []     
            for out in outs:
                for detection in out:
                    scores_mask = detection[5:]
                    class_id_mask = np.argmax(scores_mask)
                    confidence_mask = scores_mask[class_id_mask]
                    if confidence_mask > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes_mask.append([x, y, w, h])
                        confidences_mask.append(float(confidence_mask))
                        class_ids_mask.append(class_id_mask)                
            indexes_mask = cv2.dnn.NMSBoxes(boxes_mask, confidences_mask, 0.5, 0.4)
            font = cv2.FONT_HERSHEY_PLAIN
            colors = np.random.uniform(0, 255, size=(len(classes_mask), 3))
            for i in range(len(boxes_mask)):
                if i in indexes_mask:
                    x, y, w, h = boxes_mask[i]
                    label_mask = str(classes_mask[class_ids_mask[i]])
                    if(label_mask=='Mask weared partially'):
                        label_mask='No mask'
 
                    c=str(confidences_mask[i])
                    color = colors[class_ids_mask[i]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label_mask, (x, y -5),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            cv2.imshow("object detection",cv2.resize(img, (800,600)))
            if cv2.waitKey(500):
                break


# In[138]:
import cv2
import numpy as np 
import math
# In[139]:
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# In[140]:
classes = []
# In[141]:
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]  
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# In[142]:
class_ids = []
confidences = []
boxes = []
# In[143]:
video_capture = cv2.VideoCapture(0)
def make_1080p():
    video_capture.set(3,1920)
    video_capture.set(4,1080)
def make_720p():
    video_capture.set(3,1280)
    video_capture.set(4,720)
def make_480p():
    video_capture.set(3,640)
    video_capture.set(4,480)
def change_res(width,height):
    video_capture.set(3,width)
    video_capture.set(4,height)
make_720p()
change_res(1280,720)

# In[ ]:
while True:
    re,img = video_capture.read()
    print(re)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),  swapRB=True, crop=False)
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
                
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
               
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
             
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
   
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in range(len(boxes)):
        if i in indexes:
            x, y , w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label=='person':
                mask()
            if label=='traffic light':
                pass# traffic()
            c=str(confidences[i])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                      
            cv2.putText(img, label, (x, y -10),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.imshow("object detection",cv2.resize(img, (800,600)))
   
    if cv2.waitKey(1) == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
