import tensorflow as tf
import model
import util

def main():
    # Load the data
    train_data, val_data, test_data = util.load_data()

    # Create the model
    sentiment_model = model.create_model()

    # Compile the model
    sentiment_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = sentiment_model.fit(train_data, epochs=10, validation_data=val_data, callbacks=[util.early_stopping_callback()])

    # Evaluate the model
    test_loss, test_acc = sentiment_model.evaluate(test_data)

    print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')

if __name__ == "__main__":
    main()

