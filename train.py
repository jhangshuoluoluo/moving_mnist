from model import ConvLSTM2d_model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow_datasets as tfds
import os
import numpy as np

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
batch_size = 8

def preprocess(x):
    datas = x['image_sequence'] / 255
    x = datas[:-1, :, :, :]
    y = datas[1:, :, :, :]
    return x, y

## Prepare dataset
train_datas = tfds.load('moving_mnist', split='test[:80%]')
valid_datas = tfds.load('moving_mnist', split='test[-20%:]')
##
train_datas = train_datas.map(preprocess).cache().shuffle(1024).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
valid_datas = valid_datas.map(preprocess).batch(batch_size)

## Load model
model = ConvLSTM2d_model(batch_size=batch_size, training=True)
model.compile(optimizer='adam', loss='mae')
#model.compile(optimizer='adam', loss='mae')

### callbacks
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=10)
checkpoint = ModelCheckpoint('./models/best_ConvLstm2D.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True)
reduce_lr = ReduceLROnPlateau(factor=0.5, min_lr=1e-5)
callback_list = [early, checkpoint, reduce_lr]

history = model.fit(train_datas, epochs=10, validation_data=valid_datas, callbacks=callback_list)


