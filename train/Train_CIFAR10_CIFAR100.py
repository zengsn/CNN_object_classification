
# coding: utf-8

# In[9]:


# Simple CNN model for CIFAR-10
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Create the model Large scales
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
epochs = 50
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
adam = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())


# In[10]:



from keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

# Fit the model
hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=128, callbacks=[earlyStopping])
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[26]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
print(hist.history.keys())
# summarize history for accuracy
plt.figure(figsize=(15,8))
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy', size=20)
plt.ylabel('accuracy', size=20)
plt.xlabel('epoch', size=20)
plt.legend(['train', 'test'], loc='upper left', prop={'size':15})
plt.show()
# summarize history for loss
plt.figure(figsize=(15,8))
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss', size=20)
plt.ylabel('loss', size=20)
plt.xlabel('epoch', size=20)
plt.legend(['train', 'test'], loc='upper left', prop={'size':15})
plt.show()


# In[13]:


# serialize model to JSON
model_json = model.to_json()
with open("cifar10/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("cifar10/model.h5")
print("Saved model to disk")


# In[1]:


from keras.datasets import cifar100
from matplotlib import pyplot
# load data
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
# create a grid of 3x3 images
for i in range(0, 9):
	pyplot.subplot(330 + 1 + i)
	pyplot.imshow((X_train[i]))
# show the plot
pyplot.show()


# In[2]:


# Simple CNN model for CIFAR-100
import numpy as np
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

def normalize(X_train,X_test):
    #this function normalize inputs for zero mean and unit variance
    # it is used when training a model.
    # Input: training set and test set
    # Output: normalized training set and test set according to the trianing set statistics.
    mean = np.mean(X_train,axis=(0,1,2,3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    print(mean)
    print(std)
    X_train = (X_train-mean)/(std+1e-7)
    X_test = (X_test-mean)/(std+1e-7)
    return X_train, X_test

(X_train, y_train), (X_test, y_test) = cifar100.load_data()
# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train, X_test = normalize(X_train, X_test)
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(X_train)


# In[12]:


from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
# Create the model
model = Sequential()
# model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
weight_decay = 0.0005

model.add(Conv2D(64, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same',kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu',kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu',kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu',kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu',kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same',kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same',kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same',kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
          
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, activation='relu',kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization())
          
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


# In[77]:


epochs = 150
lrate = 0.1
decay = lrate/epochs
#sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
adam = Adam(lr=0.001)
sgd = SGD(lr=0.1, decay=decay, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())


# In[13]:


#training parameters
batch_size = 128
maxepoches = 250
learning_rate = 0.1
lr_decay = 1e-6
lrf = learning_rate

#optimization details
sgd = SGD(lr=lrf, decay=lr_decay, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
print(model.summary())


# In[14]:


# training process in a for loop with learning rate drop every 25 epoches.
for epoch in range(1,maxepoches):

    if epoch%25==0 and epoch>0:
        lrf/=2
        sgd = SGD(lr=lrf, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    hist = model.fit_generator(datagen.flow(X_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        epochs=epoch, verbose=1,
                        validation_data=(X_test, y_test),initial_epoch=epoch-1)


# In[48]:


f = open('cifar10/output.txt', 'r')
hist.history = {}
hist.history['acc'] = []
hist.history['loss'] = []
hist.history['val_acc'] = []
hist.history['val_loss'] = []
lines = f.readlines()
for i in range(0, len(lines)): #
    line = lines[i]
    tmps = line.split(' ')
    if tmps[0][0] == '3':
        print(tmps[6], tmps[9], tmps[12], tmps[15], )
        hist.history['loss'].append(tmps[6])
        hist.history['acc'].append(tmps[9])
        hist.history['val_loss'].append(tmps[12])
        hist.history['val_acc'].append(tmps[15])
f.close()


# In[49]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15,8))
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy', size=20)
plt.ylabel('accuracy', size=20)
plt.xlabel('epoch', size=20)
plt.legend(['train', 'test'], loc='upper left', prop={'size':15})
plt.savefig('cifar10/100accuracy_70.png')
# summarize history for loss
plt.figure(figsize=(15,8))
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss', size=20)
plt.ylabel('loss', size=20)
plt.xlabel('epoch', size=20)
plt.legend(['train', 'test'], loc='upper left', prop={'size':15})
plt.savefig('cifar10/100loss_70.png')


# In[16]:


# serialize model to JSON
model_json = model.to_json()
with open("cifar10/100model_70.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("cifar10/100model_70.h5")
print("Saved model to disk")

