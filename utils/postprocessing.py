import csv
from glob import glob
import numpy as np
import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt
import os
import medpy.metric.binary as bin
from math import pi

def draw_contour(args: np.ndarray):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

    pred= args

    pred = pred // 255

    # erase metric bar
    pred[1080:, 1400:] = 0

    o = np.uint8(pred)
    contours, hie= cv2.findContours(o, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    r = cv2.fillPoly(o, pts=contours, color=(255,255,255))

    o = cv2.morphologyEx(r, cv2.MORPH_OPEN, kernel, iterations=2)

    pp = o

    o = np.uint8(o//255)

    contours, hie= cv2.findContours(o, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_contour = cv2.drawContours(o, contours, -1, color=(255, 255, 255), thickness=5)

    return img_contour, contours, hie, pp



def analysis(img_contour, contours, hie):
    info = {}
    c = contours
    c_im = img_contour
    for i, x in enumerate(c):
        tmp = {}
        M = cv2.moments(x)

        area = M['m00']
        if area == 0.0:
            continue

        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])


        _,radius = cv2.minEnclosingCircle(x)
        _, (minorAxisLength, majorAxisLength), angle = cv2.fitEllipse(x)
        convex_hull = cv2.convexHull(x)
        hull_area = cv2.contourArea(convex_hull)
        radius = int(radius)
        diameter = radius * 2
    
        a = majorAxisLength / 2
        b = minorAxisLength / 2

        ## Here you can add any metric you want for further anaylsis
        Eccentricity = round(np.sqrt(pow(a, 2) - pow(b, 2))/a, 2)
        perimeter = np.round(cv2.arcLength(x, True),2)
        circularity = (4*pi*area)/(perimeter**2)
        Roundness = (4*area) / (pi * (majorAxisLength**2))
        solidity = float(area) / hull_area




        cv2.putText(c_im, text=str(i+1), org=(cX, cY), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255),
                thickness=1, lineType=cv2.LINE_AA)
        
        tmp["Area"] = area
        tmp["Diameter"] = diameter
        tmp["majorAxisLength"] = np.round(majorAxisLength, 2)
        tmp["minorAxisLength"] = np.round(minorAxisLength,2)
        tmp["Eccentricity"] = Eccentricity
        tmp["Perimeter"] = perimeter
        tmp["Circularity"] = circularity
        tmp["Roundness"] = Roundness
        tmp["Solidity"] = solidity
        
        info[i+1] = tmp

    
    return info, c_im


