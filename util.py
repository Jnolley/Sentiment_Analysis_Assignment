import tensorflow as tf
import tensorflow_datasets as tfds

_, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
encoder = info.features['text'].encoder

BATCH_SIZE = 64
BUFFER_SIZE = 10000
padded_shapes = ([None], ())

def prepare_datasets():
    val_splits = [f'train[{k}%:{k+10}%]' for k in range(0, 100, 10)]
    train_splits = [f'train[:{k}%]+train[{k+10}%:]' for k in range(0, 100, 10)]

    train_datasets = [tfds.load('imdb_reviews/subwords8k', split=s, as_supervised=True) for s in train_splits]
    val_datasets = [tfds.load('imdb_reviews/subwords8k', split=s, as_supervised=True) for s in val_splits]
    test_dataset = tfds.load('imdb_reviews/subwords8k', split='test', as_supervised=True)

    padded_train_datasets = [ds.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes) for ds in train_datasets]
    padded_val_datasets = [ds.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes) for ds in val_datasets]
    test_dataset_padded = test_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes)
    
    return padded_train_datasets, padded_val_datasets, test_dataset_padded