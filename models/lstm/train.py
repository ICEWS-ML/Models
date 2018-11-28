from models.lstm.network import LSTMClassifier
import numpy as np

import torch


# I want to evaluate generator functions if passed
def iterator(data):
    return data() if callable(data) else iter(data)


def filter_split(data):
    return [(x, y) for x, y, label in data if label == 'train']


def train(data, hyperparameters=None, trainparameters=None):
    torch.manual_seed(0)

    trainparameters = {
        **{'epochs': 1, 'learning_rate': 0.1},
        **(trainparameters or {})
    }

    model = LSTMClassifier(input_size=23, output_size=4, **(hyperparameters or {}))

    loss_function = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=trainparameters['learning_rate'])

    for epoch in range(trainparameters['epochs']):
        for i, time_step in enumerate(iterator(data)):
            print(f'Step: {i}')

            time_step = filter_split(time_step)
            if not time_step:
                continue

            # at each iteration, reset the gradients and lstm_state
            model.zero_grad()
            model.lstm_state_init(len(time_step))

            # format (X, Y) observations into torch tensors
            stimulus = torch.stack([torch.tensor(observation[0]) for observation in time_step], dim=0)[None].float()
            expected = torch.stack([torch.tensor(observation[1]) for observation in time_step], dim=0)[:, 0].long()

            predicted = model(stimulus)
            loss = loss_function(predicted, expected)

            # all function calls since zero_grad() were recorded. Compute gradients from call history and update weights
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), './models/lstm/weights')


def test_lstm(data, hyperparameters=None):
    model = LSTMClassifier(input_size=23, output_size=4, **(hyperparameters or {}))

    model.load_state_dict(torch.load('./models/lstm/weights'))
    model.eval()

    for i, time_step in enumerate(iterator(data)):
        print(f'Step: {i}')

        labels = np.array([val == 'train' for val in time_step])

        # format (X, Y) observations into torch tensors
        stimulus = torch.stack([torch.tensor(observation[0]) for observation in time_step], dim=0)[None].float()
        expected = torch.stack([torch.tensor(observation[1]) for observation in time_step], dim=0)[:, 0].long()

        predicted = model(stimulus)

        print(labels)
        print(expected)
        print(predicted)
