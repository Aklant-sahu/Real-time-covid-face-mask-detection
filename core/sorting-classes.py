from PIL import Image
import os
import cv2 as cv
from os import listdir
from os.path import splitext
import shutil
location="C:/Users/asus/yolov4-custom-functions/detections/crop/190111_07_SkywalkMahanakhon_HD_05_SparkVideo/images"

lab="C:/Users/asus/yolov4-custom-functions/detections/crop/190111_07_SkywalkMahanakhon_HD_05_SparkVideo/"
#os.mkdir(directory)


while 1:
    
    for i in listdir(location):
        try:
            a=i.split('.')[0]
            a=a.split('_')[-1]
            #print(a)
            shutil.move(os.path.join(location,i),os.path.join(lab,a))

        except:
            print('error')
    else:
        break