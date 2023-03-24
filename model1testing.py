from __future__ import print_function

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops

# Load IMDB dataset
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

BATCH_SIZE = 64
BUFFER_SIZE = 10000

encoder = info.features['text'].encoder

padded_shapes = ([None], ())

# Split the train_data into train_data and val_data
train_data = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=padded_shapes)
test_data = test_dataset.padded_batch(BATCH_SIZE, padded_shapes=padded_shapes)

def pad_to_size(vec, size):
    zeros = [0]*(size-len(vec))
    vec.extend(zeros)
    return vec

def sentiment_predict(text, pad):
    encoded_text = encoder.encode(text)
    if pad:
        encoded_text = pad_to_size(text, 64)
    encoded_text = tf.cast(encoded_text, tf.float32)
    predicts = model.predict(tf.expand_dims(encoded_text), 0)
    return predicts

model = tf.keras.Sequential([tf.keras.layers.Embedding(encoder.vocab_size,64), 
                             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                             tf.keras.layers.Dense(64, activation='relu'),
                             tf.keras.layers.Dense(1, activation='sigmoid')])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

model.summary()

history = model.fit(train_data, epochs=5, validation_data=test_data, validation_steps=30)