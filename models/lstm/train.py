from models.lstm.network import LSTMClassifier

import torch


def train(data, hyperparameters=None):
    torch.manual_seed(0)

    first_x, first_y = next(iter(data))

    model = LSTMClassifier(input_size=len(first_x), output_size=len(first_y), **(hyperparameters or {}))

    loss_function = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with torch.no_grad():
        print(model(torch.from_numpy(first_x[None, None]).float()))
