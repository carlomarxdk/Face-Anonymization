# -*- coding: utf-8 -*-

import cv2

def Detect(img, img_color, map_x_32, map_y_32):  
    cascade_alt = cv2.CascadeClassifier()
    cascade_alt.load("haarcascades/haarcascade_frontalface_alt.xml")
    rects = cascade_alt.detectMultiScale(img, 
                                         scaleFactor = 1.1, 
                                         minNeighbors=3,
                                         minSize = (3,3))
    
    
    cascade_prof = cv2.CascadeClassifier()
    cascade_prof.load("haarcascades/haarcascade_profileface.xml")
    rects_prof = cascade_prof.detectMultiScale(img, 
                                         scaleFactor = 1.1,
                                         minNeighbors = 2, 
                                         minSize = (3,3))
    
    img_r = cv2.remap(img, map_x_32, map_y_32, cv2.INTER_AREA) 
    cascade_prof2 = cv2.CascadeClassifier()
    cascade_prof2.load("haarcascades/haarcascade_profileface.xml")
    rects_prof2 = cascade_prof.detectMultiScale(img_r,  
                                                scaleFactor = 1.1,
                                                minNeighbors = 2, 
                                                minSize = (3,3))
    
    if len(rects_prof2) != 0:
        rects_prof2[:, 2:] += rects_prof2[:, :2]
    #box(rects_prof2, img_r)  
    img = cv2.remap(img_r, map_x_32, map_y_32, cv2.INTER_AREA) 


    
    if len(rects) != 0:
        rects[:, 2:] += rects[:, :2]
    #box(rects, img)    
    
    if len(rects_prof) != 0:
        rects_prof[:, 2:] += rects_prof[:, :2]
    #box(rects_prof, img)   
    
    
    for x1, y1, x2, y2 in rects_prof:
        roi = img_color[y1:y2, x1:x2, :]
        roi = cv2.blur(roi,(20,20))
        img_color[y1:y2, x1:x2, :] = roi
        
    for x1, y1, x2, y2 in rects_prof2:
        roi = img_color[y1:y2, x1:x2, :]
        roi = cv2.blur(roi,(20,20))
        img_color[y1:y2, x1:x2, :] = roi
    
    for x1, y1, x2, y2 in rects:
        roi = img_color[y1:y2, x1:x2, :]
        roi = cv2.blur(roi,(20,20))
        img_color[y1:y2, x1:x2, :] = roi
        
 
    return img_color

def box(rects, img):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)
        
    
    