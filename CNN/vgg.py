from __future__ import absolute_import, division, print_function, unicode_literals

import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import tensorflow as tf
run_eagerly=False
tf.config.run_functions_eagerly(run_eagerly)
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =============================================================================
# Data Generator
# =============================================================================
size  =64
target_size=(size,size) #provided by network resizing
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
test_dir = os.path.join(base_dir,'test2')
batch_size=512
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
train_generator.image_shape # bcheck 3la el shape (256,256,3)
train_generator.class_indices #bcheck 3la el classes el training


# example of tending the vgg16 model


# Building Model
vggModel = VGG16(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(size, size, 3)))

outputs = vggModel.output
outputs = Flatten(name="flatten")(outputs)
outputs = Dropout(0.9)(outputs)
outputs = Dense(6, activation="softmax")(outputs)

model = Model(inputs=vggModel.input, outputs=outputs)

for layer in vggModel.layers:
    layer.trainable = False

model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['acc']
)


ACCURACY_THRESHOLD = 0.999
import tensorflow as tf
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_acc') >= ACCURACY_THRESHOLD):
            print("\nReached %2.2f%% accuracy, so stopping training!!" %(ACCURACY_THRESHOLD*100))
            self.model.stop_training = True

# Instantiate a callback object
callbacks = myCallback()


epochs = 120
history = model.fit(train_generator, epochs = epochs,batch_size=batch_size, 
                    callbacks=[callbacks],
                    validation_data = validation_generator)
model.save("VGGXRAY.h5")
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
