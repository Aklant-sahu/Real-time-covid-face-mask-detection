import os
import cv2
import random
import numpy as np
import tensorflow as tf
import pytesseract
from core.utils import read_class_names
from core.config import cfg
###################################
import tensorflow
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

IMAGE_SIZE = (224, 224)


    # add preprocessing layer to the front of VGG

from tensorflow import keras


#####################################

# function to count objects, can return total classes or count per class
def count_objects(data, by_class = False, allowed_classes = list(read_class_names(cfg.YOLO.CLASSES).values())):
    boxes, scores, classes, num_objects = data

    #create dictionary to hold count of objects
    counts = dict()

    # if by_class = True then count objects per class
    if by_class:
        class_names = read_class_names(cfg.YOLO.CLASSES)

        # loop through total number of objects found
        for i in range(num_objects):
            # grab class index and convert into corresponding class name
            class_index = int(classes[i])
            class_name = class_names[class_index]
            if class_name in allowed_classes:
                counts[class_name] = counts.get(class_name, 0) + 1
            else:
                continue

    # else count total objects found
    else:
        counts['total object'] = num_objects
    
    return counts

# function for cropping each detection and saving as new image
def crop_objects(img, data, path, allowed_classes,frame_num):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    #create dictionary to hold count of objects for image name
    counts = dict()
    #################
    input_vgg=[]
    ##############
    for i in range(num_objects):

        # get count of class for part of image name
        class_index = int(classes[i])
        class_name = class_names[class_index]
        if class_name in allowed_classes:
            counts[class_name] = counts.get(class_name, 0) + 1
            # get box coords
            xmin, ymin, xmax, ymax = boxes[i]
            # crop detection from image (take an additional 5 pixels around all edges)
            cropped_img = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]

            try:
                cropped_img=cv2.resize(cropped_img,(224,224),interpolation=cv2.INTER_NEAREST)
                #print(np.array(cropped_img).shape)
                input_vgg.append(cropped_img/255)
            except:
                continue
            ##############################
            
            ##############################
            # construct image name and join it to path for saving crop properly

            '''
            uncomment this if you want to save the cropped images
            img_name =str(frame_num)+ '-'+ class_name + '_' + str(counts[class_name]) + '.png'
            img_path = os.path.join(path, img_name )
            # save image

            #cv2.imwrite(img_path, cropped_img)
            '''
            
        else:
            continue
    #print(np.array(input_vgg).shape)
    ###################
    return np.array(input_vgg)
    ##################
        
# function to run general Tesseract OCR on any detections 
def ocr(img, data):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    for i in range(num_objects):
        # get class name for detection
        class_index = int(classes[i])
        class_name = class_names[class_index]
        # separate coordinates from box
        xmin, ymin, xmax, ymax = boxes[i]
        # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
        box = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
        # grayscale region within bounding box
        gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
        # threshold the image using Otsus method to preprocess for tesseract
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # perform a median blur to smooth image slightly
        blur = cv2.medianBlur(thresh, 3)
        # resize image to double the original size as tesseract does better with certain text size
        blur = cv2.resize(blur, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
        # run tesseract and convert image text to string
        try:
            text = pytesseract.image_to_string(blur, config='--psm 11 --oem 3')
            print("Class: {}, Text Extracted: {}".format(class_name, text))
        except: 
            text = None
#######################################3
def vgg_model():
    IMAGE_SIZE = [224, 224]
    '''vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

    for layer in vgg.layers:
        layer.trainable = False
    # No of layers
    x = Flatten()(vgg.output)
    prediction = Dense(8, activation='softmax')(x)

    # create a model object
    model = Model(inputs=vgg.input, outputs=prediction)

    # view the structure of the model
    model.summary()


    # Compile the model
    model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
    )


    path=r'final_model-weights.hd5'
    model = model.load_weights(path)
    
    train_path = "C:/Users/asus/yolov4-custom-functions/detections/crop/190111_07_SkywalkMahanakhon_HD_05_SparkVideo/train"
    valid_path = "C:/Users/asus/yolov4-custom-functions/detections/crop/190111_07_SkywalkMahanakhon_HD_05_SparkVideo/test"
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)
    training_set = train_datagen.flow_from_directory("C:/Users/asus/yolov4-custom-functions/detections/crop/190111_07_SkywalkMahanakhon_HD_05_SparkVideo/train",
                                                 target_size = (224, 224),
                                                 batch_size = 16,
                                                 class_mode = 'categorical')

    test_set = test_datagen.flow_from_directory("C:/Users/asus/yolov4-custom-functions/detections/crop/190111_07_SkywalkMahanakhon_HD_05_SparkVideo/test/",
                                                target_size = (224, 224),
                                                batch_size = 16,
                                                class_mode = 'categorical')
    print(len(training_set))

    print(len(test_set))

    # fit the model
    r = model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=1,
    )'''
    path=r'final_model'
    model = tf.keras.models.load_model(path)

    return model
    ######################################################