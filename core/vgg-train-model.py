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

# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = "C:/Users/asus/yolov4-custom-functions/detections/crop/190111_07_SkywalkMahanakhon_HD_05_SparkVideo/train"
valid_path = "C:/Users/asus/yolov4-custom-functions/detections/crop/190111_07_SkywalkMahanakhon_HD_05_SparkVideo/test"

# add preprocessing layer to the front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in vgg.layers:
  layer.trainable = False



folders = glob('C:/Users/asus/yolov4-custom-functions/detections/crop/190111_07_SkywalkMahanakhon_HD_05_SparkVideo/train/*')
print(len(folders))

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
  epochs=17,
  
)
'''
# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')'''

import tensorflow

from tensorflow.keras.models import load_model

##Saving the model
model.save('final_model')
