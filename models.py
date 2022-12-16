"""
RELU 
Rectified Linear Unit(ReLU) as it is the simplest non-Linear
function appropriate for resource-constrained microcontrollers.

CNN
CNN improves performance and reduced model sizewhen convolution is applied along the frequency axis.
Multiple convolutional layers could exacerbate number of multiplies across time, frequency and feature maps
which is not suitable for microcontrollers.

BATCH NORM 
Using batch normalization learning becomes efficient also it can be used as regularization to avoid overfitting of the model.
It is often placed just after defining the sequential model and after the convolution and pooling layers. 

DROPOUT 
Usually advised not to use after the convolution layers, they are mostly used after the dense layers of the network. 

"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers

def model_0(input_shape, nb_classes):
    model = tf.keras.models.Sequential()

    # Normalize data
    model.add(layers.Normalization())

    #model.add(layers.InputLayer(input_shape=(train_set.shape[1],train_set.shape[2],train_set.shape[3]), batch_size=(batchSize)))
    model.add(layers.Conv2D(filters=3, kernel_size=(3,3), padding="same", input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(filters=16, kernel_size=(3,3), strides=(2,2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPool2D((2,2)))

    model.add(layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPool2D((2,2)))

    #model.add(layers.Conv2D(filters=48, kernel_size=(3,3), strides=(2,2), padding='same'))
    model.add(layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Flatten())

    model.add(layers.Dense(8, kernel_regularizer=(regularizers.l1(0))))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.1))

    model.add(layers.Dense(nb_classes))  # Nb classes to recognize
    model.add(layers.Activation('softmax'))  # Softmax to get probas of each class

    return model

def model_1(input_shape, nb_classes):
    model = tf.keras.models.Sequential()

    # Normalize data
    model.add(layers.Normalization())

    model.add(layers.Conv2D(filters=16, kernel_size=(3,3), padding="same", input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPool2D((2,2)))

    model.add(layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPool2D((2,2)))

    model.add(layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Flatten())
    
    model.add(layers.Dense(32, kernel_regularizer=(regularizers.l1(0))))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.1))

    model.add(layers.Dense(nb_classes))  # Nb classes to recognize
    model.add(layers.Activation('softmax'))  # Softmax to get probas of each class

    return model

def get_model(input_shape, nb_classes, model_idx):
    if model_idx == 0:
        return model_0(input_shape, nb_classes)
    elif model_idx==1:
        return model_1(input_shape, nb_classes)
    else:
        raise "Model not found"
    