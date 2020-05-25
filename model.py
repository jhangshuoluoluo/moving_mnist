import tensorflow as tf
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv3D, Input
from tensorflow.keras.models import Sequential

class ConvLSTM2d_model(tf.keras.Model):
    def __init__(self, training=True):
        super(ConvLSTM2d_model2, self).__init__()
        self.convlstm1 = ConvLSTM2D(64, (5, 5), input_shape=(None, 64, 64, 1), padding='same', return_sequences=True, return_state=True)
        self.bn1 = BatchNormalization()
        self.convlstm2 = ConvLSTM2D(64, (3, 3), padding='same', return_sequences=True, return_state=True)
        self.bn2 = BatchNormalization()
        self.out = Conv3D(1, kernel_size=(1, 3, 3), activation='sigmoid', padding='same')

    def call(self, image_sequences, states_c=None, states_h=None):
        ## image_sequences shape => [batch_size, 1, rows, cols, channel]
        if states_c is None:
            x1, state0_c, state0_h = self.convlstm1(image_sequences)
            x1 = self.bn1(x1)
            x2, state1_c, state1_h = self.convlstm2(x1)
            x2 = self.bn2(x2)
            x3 = tf.concat([image_sequences, x1, x2], axis=-1)
            output = self.out(x3)
        else:
            x1, state0_c, state0_h = self.convlstm1(image_sequences, initial_state=[states_c[0], states_h[0]])
            x1 = self.bn1(x1)
            x2, state1_c, state1_h = self.convlstm2(x1, initial_state=[states_c[1], states_h[1]])
            x2 = self.bn2(x2)
            x3 = tf.concat([image_sequences, x1, x2], axis=-1)
            output = self.out(x3)


        return output, [state0_c, state1_c], [state0_h, state1_h] 

if __name__ == '__main__':
    model = ConvLSTM2d_model2()
