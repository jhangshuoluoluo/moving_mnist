from model import ConvLSTM2d_model2
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow_datasets as tfds
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from train2 import loss_function, valid_step, preprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def valid_step(x_input, y_input):
    loss = 0

    ## Encoder
    pred, states_c, states_h = model(x_input)

    collect = []
    ## Decoder
    dec_input = tf.expand_dims(x_input[:, -1], 1)
    for t in range(y_input.shape[1]):
        dec_pred, states_c, states_h = model(dec_input, states_c, states_h)
        loss += loss_function(tf.expand_dims(y_input[:, t], 1), dec_pred)
        
        collect.append(dec_input)
        dec_input = dec_pred

    batch_loss = (loss / int(y_input.shape[1]))

    return batch_loss, collect


## Prepare dataset
train_datas = tfds.load('moving_mnist', split='test[:80%]')
valid_datas = tfds.load('moving_mnist', split='test[-20%:]')
##
train_datas = train_datas.map(preprocess).shuffle(2000).batch(1)
valid_datas = valid_datas.map(preprocess).shuffle(2000).batch(1)

## Load model
model = ConvLSTM2d_model2(training=False)
model.build(input_shape=(None, None, 64, 64, 1))

optimizer = tf.keras.optimizers.Adam()
print(tf.train.latest_checkpoint('./training_checkpoints'))
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(tf.train.latest_checkpoint('./training_checkpoints'))
#status = checkpoint.restore('./training_checkpoints/ckpt-3').assert_consumed()

valid_loss = 0
lowest = 100
for batch_step, (xx, yy) in enumerate(train_datas.take(1)):
    batch_loss, collect = valid_step(xx, yy)
    valid_loss += batch_loss

print(valid_loss.numpy()/(batch_step+1))

fig = plt.figure(1)
ims = []
#collect = np.squeeze(yy.numpy())
#collect = preserved
for ii in collect:
    ii = np.squeeze(ii)
    im = plt.imshow(ii, cmap='gray')
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=1000)
plt.show()
    
