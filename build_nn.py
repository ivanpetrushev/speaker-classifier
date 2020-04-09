import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow.keras as keras
from pprint import pprint
import matplotlib.pyplot as plt
from datetime import datetime


EXTRACTED_FILE = "extracted.json"


def load_extract():
    with open(EXTRACTED_FILE, 'r') as fp:
        data = json.load(fp)

    x = np.array(data['mfcc'])
    y = np.array(data['labels'])
    return x, y, data['mapping']


# source: https://gist.github.com/zachguo/10296432#gistcomment-2638135
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    # Begin CHANGES
    fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES

    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")

    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


if __name__ == '__main__':
    x, y, mapping = load_extract()

    # divide train, test and predict sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    print('Train set count:')
    bins = np.amax(y_train) + 1
    hist_train, _ = np.histogram(y_train, bins=bins)
    pprint(hist_train)

    x_test, x_predict, y_test, y_predict = train_test_split(x_test, y_test, test_size=0.5)

    print('Test set count:')
    hist_test, _ = np.histogram(y_test, bins=bins)
    pprint(hist_test)

    print('Predict set count:')
    hist_predict, _ = np.histogram(y_predict, bins=bins)
    pprint(hist_predict)

    print('Total set count:')
    hist_total = np.sum([hist_train, hist_test, hist_predict], axis=0)
    pprint(hist_total)

    # NN topology - trying different models

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
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=50)

    # Plot training & validation accuracy values
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    #
    # # Plot training & validation loss values
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()

    # generate confusion matrix
    print("Generating confusion matrix")
    y_confusion = []
    y_predict = y_predict.tolist()
    for i, idx in enumerate(y_predict):
        y_predict[i] = mapping[idx]
    for x_item in x_predict:
        x_item = x_item[..., np.newaxis]
        x_item = x_item[np.newaxis, ...]
        prediction = model.predict(x_item)
        prediction = prediction[0]
        selected_idx = np.argmax(prediction)
        selected_tag = mapping[selected_idx]
        y_confusion.append(selected_tag)
    matrix = confusion_matrix(y_predict, y_confusion, labels=mapping)
    pprint(matrix)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(matrix)
    # plt.title('Confusion matrix of the classifier')
    # fig.colorbar(cax)
    # ax.set_xticklabels([''] + mapping)
    # ax.set_yticklabels([''] + mapping)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.show()

    print_cm(matrix, mapping)

    # save model
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_filename = 'data/model-' + now + '.h5'
    model.save(model_filename)
    print("Model saved to: ", model_filename)
