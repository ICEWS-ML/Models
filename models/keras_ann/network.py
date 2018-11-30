import keras
import numpy as np


def iterator(data):
    return data() if callable(data) else iter(data)


def filter_split(data, split='train'):
    for x, y, label in data:
        if label == split:
            yield x, y


def create_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=100, activation='relu', input_dim=23))
    model.add(keras.layers.Dense(units=50, activation='relu'))
    model.add(keras.layers.Dense(units=4, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
        metrics=['accuracy']
    )

    return model


def train_keras(data, trainparameters):
    x, y = [np.array(i) for i in list(zip(*filter_split(iterator(data), split='train')))]
    model = create_model()
    model.fit(x, y, **trainparameters)
    model.save('./models/keras_ann/model.h5')


def test_keras(data):
    x, y = [np.array(i) for i in list(zip(*filter_split(iterator(data), split='test')))]
    model = keras.models.load_model('./models/keras_ann/model.h5')
    print(model.evaluate(x, y))
