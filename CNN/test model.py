from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model
import tensorflow_hub as hub
import tensorflow as tf
run_eagerly=False
tf.config.run_functions_eagerly(
    run_eagerly
)

import tensorflow.keras
from tensorflow.keras import backend as K
#from keras.layers.core import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dropout
import numpy as np
from IPython.display import Image
from tensorflow.keras.optimizers import Adam

#tf.logging.set_verbosity(tf.logging.ERROR)
#tf.enable_eager_execution()
import tensorflow_hub as hub
import os
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
#from keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =============================================================================
# Data Generator
# =============================================================================
size  =512
target_size=(size,size) #provided by network resizing
#bast5dm el swar 3shan a3mlha zoom in w out w rescale 3shan a5od mnha kol el positions 
#ll validation w el training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

val_datagen = ImageDataGenerator(rescale=1./255)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) # set validation split
# =============================================================================
# Generator
# =============================================================================
base_dir = 'data'
train_dir = os.path.join(base_dir ,'train2')
test_dir = os.path.join(base_dir,'test')
batch_size=4
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    test_dir, # same directory as training data
    target_size=target_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical') # set as validation data
# =============================================================================
# 
# =============================================================================
train_generator.image_shape 
train_generator.class_indices 
# =============================================================================
# Model
# =============================================================================
model = tf.keras.Sequential([
   tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=(size,size,3)),
   tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
   tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
   tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
   tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
   tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
   tf.keras.layers.Flatten(),
   tf.keras.layers.Dense(64, activation='relu'),
   tf.keras.layers.Dense(64,activation='relu'),
   tf.keras.layers.Dropout(rate=0.8),
   tf.keras.layers.Dense(6, activation='softmax')
])
# =============================================================================
# convert to array
# =============================================================================
img1 = image.load_img('data/train/5/5.0.jpg')
#plt.imshow(img1)
#preprocess image
img1 = image.load_img('data/train/5/5.0.jpg', target_size=(256, 256))
img = image.img_to_array(img1)
img = img/255
img = np.expand_dims(img, axis=0)
# =============================================================================
# Summary
# =============================================================================
EPOCHS = 20
#EPOCHS = 250
INIT_LR = 1e-3
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
#model.compile(loss = 'binary_crossentropy', metrics = ['acc'])

from tensorflow.keras.optimizers import SGD
#opt = SGD(lr=0.005)
model.compile(loss = "binary_crossentropy", optimizer = opt,metrics = ['acc'])

 
history = model.fit(train_generator, epochs = EPOCHS, validation_data = validation_generator)

# =============================================================================
# Plot Acc
# =============================================================================
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# =============================================================================
# Plot Loss
# =============================================================================
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# =============================================================================
# Save model
# =============================================================================
model.save('currency.h5')
# =============================================================================
# 
# =============================================================================
# Get classes of model trained on
classes = train_generator.class_indices 
Classes = ["Ten Pounds","100 Pound","Twenty Pounds","200 Pound","Five Pounds","50 Pound"]


