import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from model import create_lstm_model, create_drop_model
from util import prepare_datasets

# Load dataset and create the train and validation splits
train_datasets, val_datasets, test_dataset = prepare_datasets()

# Train and evaluate the model
accs = []
accs2 = []
for i in range(len(train_datasets)):
    # Create the models
    lstm_model = create_lstm_model()
    drop_model = create_drop_model()
    if i == 0:
        lstm_model.summary()
        drop_model.summary()
    
    # Compile the model
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    drop_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model with early stopping
    epochs = 500
    history = lstm_model.fit(train_datasets[i], epochs=epochs, validation_data=val_datasets[i], callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)])
    history = drop_model.fit(train_datasets[i], epochs=epochs, validation_data=val_datasets[i], callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)])

    # Evaluate the model
    _, accuracy = lstm_model.evaluate(test_dataset)
    print(f'Test accuracy: {accuracy:.4f}')
    accs.append(accuracy)
    
    _, accuracy2 = drop_model.evaluate(test_dataset)
    print(f'Test accuracy: {accuracy:.4f}')
    accs2.append(accuracy2)

print(f"Average validation accuracy across {len(train_datasets)} folds (LSTM-based): {np.mean(accs)}")
print(f"Average validation accuracy across {len(train_datasets)} folds (GRU-based): {np.mean(accs)}")
