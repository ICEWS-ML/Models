from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


def test_train_split(data):
    x_train, y_train = [], []
    x_test, y_test = [], []

    for x, y, label in data:
        if label == 'train':
            x_train.append(x)
            y_train.append(y)
        else:
            x_test.append(x)
            y_test.append(y)
    return x_train, y_train, x_test, y_test


def train_random_forest(data, hyperparameters=None, trainparameters=None):
    model = RandomForestClassifier(**hyperparameters)
    x_train, y_train, x_test, y_test = test_train_split(data)
    model.fit(x_train, y_train)
    joblib.dump(model, './models/random_forest/weights')


def test_random_forest(data):
    model = joblib.load('./models/random_forest/weights')
    x_train, y_train, x_test, y_test = test_train_split(data)
    return model.predict(x_test)
