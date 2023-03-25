from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tqdm import tqdm

_, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)

# Define 10-fold cross-validation splits
val_splits = [f'train[{k}%:{k+10}%]' for k in range(0, 100, 10)]
train_splits = [f'train[:{k}%]+train[{k+10}%:]' for k in range(0, 100, 10)]

# Load CIFAR-100 dataset for first split
train_datasets = [tfds.load('imdb_reviews/subwords8k', split=s, as_supervised=True) for s in train_splits]
val_datasets = [tfds.load('imdb_reviews/subwords8k', split=s, as_supervised=True) for s in val_splits]
test_dataset = tfds.load('imdb_reviews/subwords8k', split='test', as_supervised=True)

BATCH_SIZE = 64
BUFFER_SIZE = 10000


encoder = info.features['text'].encoder

padded_shapes = ([None], ())

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

sequential_model = create_sequential_model()
sequential_model.summary()

def prepare_datasets(train_datasets, val_datasets):
    padded_train_datasets = [ds.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes) for ds in train_datasets]
    padded_val_datasets = [ds.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes) for ds in val_datasets]
    test_dataset_padded = test_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes)
    
    return padded_train_datasets, padded_val_datasets, test_dataset_padded

train_datasets, val_datasets, test_dataset = prepare_datasets(train_datasets, val_datasets)

accs = []

for i in range(len(train_datasets)):
    # Compile the model
    sequential_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Create early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)

    train_ds, val_ds, test_ds = train_datasets[i], val_datasets[i], test_dataset
    
        # Train the model with early stopping
    epochs = 500
    history = sequential_model.fit(train_ds, epochs=epochs, validation_data=test_ds, callbacks=[early_stopping])

    # Evaluate the model
    _, accuracy = sequential_model.evaluate(test_ds)
    print(f'Test accuracy: {accuracy:.4f}')
    accs.append(accuracy)

print(f"Average validation accuracy across {len(train_datasets)} folds (model1): {np.mean(accs)}")

