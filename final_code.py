# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 19:55:16 2021

@author: Cem Kilic
"""

import cv2
import time

conf_scale = 0.6
nms_scale = 0.4
colors = [(0, 0, 255), (255, 255, 255), (0, 255, 0), (255, 0, 0)]
class_names = ["afiyet ilk","afiyet olsun","gorusuruz","ismin","nasilsin","sagol","sakin"]
first_move = 0
cap = cv2.VideoCapture(0)
net = cv2.dnn.readNet("F:\YOLO\own_half_tested\yolov4-obj_best.weights","F:\YOLO\own_half_tested\yolov4-obj.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)
while 1:
    (_, frame) = cap.read()
    impscore = 0
    impclassid = -1
    classes, scores, boxes = model.detect(frame, conf_scale, nms_scale)
    for (classid, score, box) in zip(classes, scores, boxes):
        if score>impscore:
            impscore = score
            impclassid = classid
        else:
            continue
    for (classid, score, box) in zip(classes, scores, boxes):
        if impclassid != classid:
            continue
        label = ""
        color_id = int(classid) % len(colors)
        color = colors[color_id]
        if classid[0] == 0:
            if score > conf_scale:
                first_move = time.time()
                print("Afiyet olsun ilk hareketi")
            continue
        elif classid[0] == 1:
            if (time.time()-first_move)<3:
                label = "%s -> %%%.2f" % (class_names[classid[0]], score * 100)
                print(label)
            else:
                continue
        else :
            label = "%s -> %%%.2f" % (class_names[classid[0]], score*100)
            print(label)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        break

    cv2.imshow("detections", frame)

    if cv2.waitKey(1) > 1:
        break