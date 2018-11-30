from models.torch_lstm.network import LSTMClassifier
import numpy as np

import torch


# I want to evaluate generator functions if passed
def iterator(data):
    return data() if callable(data) else iter(data)


def filter_split(data, split='train'):
    for x, y, label in data:
        if label == split:
            yield x, y


def train_lstm(data, hyperparameters, trainparameters=None):
    torch.manual_seed(0)

    trainparameters = {
        **{'epochs': 10, 'learning_rate': 0.1},
        **(trainparameters or {})
    }

    model = LSTMClassifier(**hyperparameters)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=trainparameters['learning_rate'])

    for epoch in range(trainparameters['epochs']):
        for i, (x, y) in enumerate(filter_split(iterator(data), split='train')):
            print(f'Step: {i}')
            model.lstm_state_init(1)
            model.zero_grad()

            predicted = model(torch.from_numpy(x)[None])
            loss = criterion(predicted, torch.tensor(y).long())

            # all function calls since zero_grad() were recorded. Compute gradients from call history and update weights
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), './models/torch_lstm/weights')


def test_lstm(data, networkparameters):
    model = LSTMClassifier(**networkparameters)

    model.load_state_dict(torch.load('./models/torch_lstm/weights'))
    model.eval()  # turn on evaluation mode

    expected, actual = [], []

    with torch.no_grad():
        for x, y in filter_split(iterator(data), split='test'):
            actual.append(int(np.argmax(np.squeeze(model(torch.from_numpy(x)[None])))))
            expected.append(int(np.squeeze(np.array(y))))

    return expected, actual
