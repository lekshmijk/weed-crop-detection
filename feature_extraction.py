import cv2
import glob
from skimage.filters import gaussian
from skimage import img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
from skimage.io import imread, imshow
from skimage.color import rgb2gray, rgb2hsv
from skimage.measure import label, regionprops, regionprops_table
from skimage.filters import threshold_otsu
from scipy.ndimage import median_filter
from matplotlib.patches import Rectangle
from skimage.io import imread, imshow
from skimage.color import rgb2gray, rgb2hsv
from skimage.measure import label, regionprops, regionprops_table
from skimage.filters import threshold_otsu
from scipy.ndimage import median_filter
from matplotlib.patches import Rectangle
import os
import math


current_path = os.getcwd()
path = "C:\\Users\\lekshmi\\Desktop\\code\\asphalt-crack-classifier\\process\\"
os.chdir(path)
img_number = 0

featuresList = []

def getContours(img, img_number, file_name):
        contours = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        BLOB_number = 0

        for cnt in contours: 
            area = cv2.contourArea(cnt)
            #print(area)
            if area > 2500:
                x,y,w,h = cv2.boundingRect(cnt)
                (b1, b2),(major_axis, minor_axis), angle = cv2.fitEllipse(cnt)
                perimeter = cv2.arcLength(cnt, True)
                form_factor = (4 * math.pi * area) / math.pow(perimeter, 2)
                hull = cv2.convexHull(cnt)
                hullArea = cv2.contourArea(hull)
                solidity = area / float(hullArea)
                hullPerimeter = cv2.arcLength(hull,True)
                convexity = hullPerimeter / perimeter
                eccentricity = minor_axis / major_axis
                # Calculate Moments
                moments = cv2.moments(img)
                # Calculate Hu Moments
                huMoments = cv2.HuMoments(moments)
                # Log scale hu moments
                for i in range(0,7):
                    huMoments[i] = -1* math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))
                BLOB = img[y:y+h, x:x+w]
                cv2.imwrite('C:\\Users\\lekshmi\\Desktop\\code\\asphalt-crack-classifier\\process_blobs\\BLOB'+str(img_number)+'_{}.png'.format(BLOB_number), BLOB)
                print('BLOB'+str(img_number)+'_{}.png'.format(BLOB_number))
                blob_file = 'BLOB'+str(img_number)+'_{}.png'.format(BLOB_number)
                cv2.rectangle(img,(x,y),(x+w,y+h),(36,255,12),2)
            
                BLOB_number += 1
            
                feature = [blob_file, form_factor, eccentricity, convexity, solidity, huMoments[1],huMoments[2],huMoments[3],huMoments[4]]
                featuresList.append(feature)

files = os.listdir()


for file in files:
    #print(file)
    img = cv2.imread(file)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (25, 25, 25), (70, 255,255))

    imask = mask>0
    green = np.zeros_like(img, np.uint8)
    green[imask] = img[imask]

    kernel = np.ones((6,6),np.uint8)
    kernel2 = np.ones((3,3),np.uint8)


    #dilation = cv2.dilate(mask,kernel,iterations = 1)

    closing2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    
    #imgContour = img.copy()             
    
    getContours(closing2, img_number, file)
    cv2.imwrite("C:\\Users\\lekshmi\\Desktop\\code\\asphalt-crack-classifier\\processed\\img_"+str(img_number)+".png", closing2)
    img_number +=1



blobs = label(closing2 > 0)
#imshow(blobs, cmap = 'tab10')
properties =['Blob_path','form_factor', 'eccentricity', 'convexity', 'solidity', 'HuMoments_1', 'HuMoments_2', 'HuMoments_3', 'HuMoments_4']

df = pd.DataFrame(featuresList)

os.chdir(current_path)
df.to_csv('feature1.csv', index=False, sep=",", header=properties)