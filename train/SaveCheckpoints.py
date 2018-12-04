import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import initializers
from keras import backend as K
K.set_image_dim_ordering('th')

import os
import numpy as np
import scipy.io
import scipy.misc
import matplotlib.pyplot

model = Sequential()
saver = tf.train.Saver()
model.load_model(h5_filename)
sess  = K.get_session()
model_name = "vgg_caltech_101"
ckpt_filename = model_name + ".ckpt"
save_path = saver.save(sess, ckpt_filename)
print("Saved checkpoints to disk: " + ckpt_filename)