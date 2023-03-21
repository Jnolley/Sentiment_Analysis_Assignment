import tensorflow as tf
import tensorflow_datasets as tfds

def load_data():
    # Load IMDB dataset
    (train_data, test_data), info = tfds.load('imdb_reviews', split=['train', 'test'], with_info=True, as_supervised=True)

    # Preprocess the data
    # Implement your preprocessing steps here

    # Split the train_data into train_data and val_data
    train_data = train_data.shuffle(25000)
    val_data = train_data.take(5000)
    train_data = train_data.skip(5000)

    return train_data, val_data, test_data

def early_stopping_callback():
    return tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

