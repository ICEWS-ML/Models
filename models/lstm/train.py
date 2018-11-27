from models.lstm.network import LSTMClassifier

import torch


# I want to evaluate generator functions if passed
def iterator(data):
    return data() if callable(data) else iter(data)


def train(data, hyperparameters=None, trainparameters=None):
    torch.manual_seed(0)

    trainparameters = {
        **{'epochs': 5, 'learning_rate': 0.1},
        **(trainparameters or {})
    }

    first_x, first_y = next(iterator(data))[0]

    model = LSTMClassifier(input_size=len(first_x), output_size=4, **(hyperparameters or {}))

    loss_function = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=trainparameters['learning_rate'])

    # with torch.no_grad():
    #     model.lstm_state_init(1)
    #     print(model(torch.from_numpy(first_x[None, None]).float()))

    for epoch in range(trainparameters['epochs']):
        for i, time_step in enumerate(iterator(data)):
            print(f'Step: {i}')

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

    # with torch.no_grad():
    #     model.lstm_state_init(1)
    #     print(model(torch.from_numpy(first_x[None, None]).float()))

    torch.save(model.state_dict(), './models/lstm/weights')
