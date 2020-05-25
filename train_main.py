from model import ConvLSTM2d_model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow_datasets as tfds
import os, time
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def preprocess(x):
    datas = x['image_sequence'] / 255
    x = datas[:10, :, :, :]
    y = datas[10:, :, :, :]

    return x, y

def loss_function(real, pred):
    real = tf.reshape(real, [real.shape[0], -1])
    pred = tf.reshape(pred, [pred.shape[0], -1])
    loss_ = tf.keras.losses.binary_crossentropy(real, pred)

    return tf.reduce_mean(loss_)

@tf.function
def train_step(x_input, y_input, epoch):
    loss = 0

    with tf.GradientTape() as tape:
        ## Encoder
        pred, states_c, states_h = model(x_input)

        ## Decoder
        dec_input = tf.expand_dims(x_input[:, -1], 1)  ## Feed to decoder
        for t in range(y_input.shape[1]):
            dec_pred, states_c, states_h = model(dec_input, states_c, states_h)    
            loss += loss_function(tf.expand_dims(y_input[:, t], 1), dec_pred)

            ## Schedule sampling
            if np.random.random() <= 1 - 0.95**epoch:
                dec_input = dec_pred
            else:
                dec_input = tf.expand_dims(y_input[:, t], 1)  ## teacher forcing

    batch_loss = (loss / int(y_input.shape[1]))

    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

@tf.function
def valid_step(x_input, y_input):
    loss = 0
    
    ## Encoder
    pred, states_c, states_h = model(x_input)

    ## Decoder
    dec_input = tf.expand_dims(x_input[:, -1], 1)
    for t in range(y_input.shape[1]):
        dec_pred, states_c, states_h = model(dec_input, states_c, states_h)
        loss += loss_function(tf.expand_dims(y_input[:, t], 1), dec_pred)

        dec_input = dec_pred

    batch_loss = (loss / int(y_input.shape[1]))

    return batch_loss

if __name__ == '__main__':
    EPOCHS = 30
    batch_size = 8
    
    ## Prepare dataset
    train_datas = tfds.load('moving_mnist', split='test[:80%]')
    valid_datas = tfds.load('moving_mnist', split='test[-20%:]')
    ##
    train_datas = train_datas.map(preprocess).shuffle(8000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    valid_datas = valid_datas.map(preprocess).batch(batch_size)
    
    ## Load model
    model = ConvLSTM2d_model2(training=True)
    model.build(input_shape=(None, None, 64, 64, 1))
    ## optimizer
    optimizer = tf.keras.optimizers.Adam()
    #model.compile(loss='mse', optimizer=optimizer)
    #model.fit(tf.random.uniform((1, 1, 64, 64, 1)), tf.random.uniform((1, 1, 64, 64, 1)), epochs=1)

    ## training process
    ## checkpoint
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    
    ## Restore models
    #status = checkpoint.restore(tf.train.latest_checkpoint('./training_checkpoints')).assert_consumed()
    
    for epoch in range(EPOCHS):
        start = time.time()
    
        total_loss = 0
        for batch_step, (xx, yy) in enumerate(train_datas):
            batch_loss = train_step(xx, yy, epoch)
            total_loss += batch_loss
    
            if batch_step % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch+1, batch_step, batch_loss.numpy()))
    
        valid_loss = 0
        print('Validation time...', end='')
        for valid_batch_step, (xx, yy) in enumerate(valid_datas):
            batch_loss = valid_step(xx, yy)
            valid_loss += batch_loss
        print('Done')
    
        if (epoch + 1) % 1 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
    
        print('Epoch {} Loss {:.4f} Valid Loss {:.4f}'.format(epoch+1, total_loss/(batch_step+1), valid_loss/(valid_batch_step+1)))
    
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
