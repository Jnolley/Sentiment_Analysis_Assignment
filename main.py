import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from model import create_sequential_model
from util import prepare_datasets

# Load dataset and create the train and validation splits
train_datasets, val_datasets, test_dataset = prepare_datasets()

# Create the model
sequential_model = create_sequential_model()
sequential_model.summary()

# Train and evaluate the model
accs = []

for i in range(len(train_datasets)):
    # Compile the model
    sequential_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model with early stopping
    epochs = 500
    history = sequential_model.fit(train_datasets[i], epochs=epochs, validation_data=val_datasets[i], callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)])

    # Evaluate the model
    _, accuracy = sequential_model.evaluate(test_dataset)
    print(f'Test accuracy: {accuracy:.4f}')
    accs.append(accuracy)

print(f"Average validation accuracy across {len(train_datasets)} folds (model1): {np.mean(accs)}")
