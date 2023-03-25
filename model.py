import tensorflow as tf
from tensorflow_datasets.core import dataset_info
from util import encoder

class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],), initializer='zeros', trainable=True)

    def call(self, inputs):
        q = tf.nn.tanh(tf.linalg.matmul(inputs, self.W) + self.b)
        a = tf.nn.softmax(q, axis=1)
        return tf.reduce_sum(inputs * a, axis=1)

def create_sequential_model(embed_dim=128, lstm_units=64):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(None,)),
        tf.keras.layers.Embedding(encoder.vocab_size, embed_dim),
        tf.keras.layers.LSTM(lstm_units, return_sequences=True),
        Attention(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return model
