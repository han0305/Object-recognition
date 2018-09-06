from keras.datasets import cifar10
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

(X_train,Y_train),(X_test,Y_test) = cifar10.load_data()
print("Training Images {}".format(X_train.shape))
print("Testing Images {}".format(X_test.shape))
print(X_train[0].shape)
for i in range(0,9):
    plt.subplot(330+1+i)
    img=X_train[i]
    plt.imshow(img)
plt.show()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train/=255.0
X_test/=255.0
y_train = np_utils.to_categorical(Y_train)
y_test = np_utils.to_categorical(Y_test)

from keras.models import Sequential
from keras.layers import Dropout,Activation,Conv2D,GlobalAveragePooling2D
from keras.optimizers import SGD

def allcnn(weights=None):
    
    model = Sequential()

    
    model.add(Conv2D(96, (3, 3), padding = 'same', input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), padding = 'same', strides = (2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3), padding = 'same', strides = (2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (1, 1), padding = 'valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(10, (1, 1), padding = 'valid'))


    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    
    
    if weights:
        model.load_weights(weights)
    
    
    return model
    

learning_rate = 0.01
weight_decay = 1e-6
momentum = 0.9
 
model = allcnn()


sgd = SGD(lr=learning_rate, decay=weight_decay, momentum=momentum, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print (model.summary())


epochs = 350
batch_size = 32


model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose = 1)