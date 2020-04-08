import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from pprint import pprint
import matplotlib.pyplot as plt


EXTRACTED_FILE = "extracted.json"


def load_extract():
    with open(EXTRACTED_FILE, 'r') as fp:
        data = json.load(fp)

    x = np.array(data['mfcc'])
    y = np.array(data['labels'])
    return x, y


if __name__ == '__main__':
    x, y = load_extract()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    print('Train label counts:')
    bins = np.amax(y_train) + 1
    hist_train, _ = np.histogram(y_train, bins=bins)
    pprint(hist_train)

    print('Test label counts:')
    hist_test, _ = np.histogram(y_test, bins=bins)
    pprint(hist_test)
    print('Total label counts:')
    hist_total = np.sum([hist_train, hist_test], axis=0)
    pprint(hist_total)

    # NN topology
    print('shape', x.shape[1], x.shape[2])

    # initial model
    # model = keras.Sequential([
    #     keras.layers.Flatten(input_shape=(x.shape[1], x.shape[2])),     # input layer
    #     keras.layers.Dense(512, activation='relu'),                     # 1st dense layer
    #     keras.layers.Dense(256, activation='relu'),                     # 2nd dense layer
    #     keras.layers.Dense(64, activation='relu'),                      # 3rd dense layer
    #     keras.layers.Dense(10, activation='softmax')                    # output layer
    # ])

    # model solving for overfitting
    # model = keras.Sequential([
    #     keras.layers.Flatten(input_shape=(x.shape[1], x.shape[2])),     # input layer
    #     keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)), # 1st dense layer
    #     keras.layers.Dropout(0.3),
    #     keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)), # 2nd dense layer
    #     keras.layers.Dropout(0.3),
    #     keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)), # 3rd dense layer
    #     keras.layers.Dropout(0.3),
    #     keras.layers.Dense(10, activation='softmax') # output layer
    # ])

    # CNN model
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    model = keras.Sequential([
        # 1st conv layer
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(x_train.shape[1], x_train.shape[2], 1)),
        keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),
        # 2nd conv layer
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),
        # 3rd conv layer
        keras.layers.Conv2D(32, (2, 2), activation='relu'),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),
        # flatten output and feed it into dense layer
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        # output layer
        keras.layers.Dense(bins, activation='softmax')
    ])

    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    print("Fitting model...")
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=100)

    print('history', history)
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
