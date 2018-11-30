from models.torch_ann.network import ANNClassifier
import numpy as np
import torch


# I want to evaluate generator functions if passed
def iterator(data):
    return data() if callable(data) else iter(data)


def filter_split(data, split='train'):
    for x, y, label in data:
        if label == split:
            yield x, y


def train_ann(data, networkparameters, trainparameters=None):
    torch.manual_seed(0)

    trainparameters = {
        **{'epochs': 100, 'learning_rate': 0.5},
        **(trainparameters or {})
    }

    model = ANNClassifier(**networkparameters)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=trainparameters['learning_rate'])

    for epoch in range(trainparameters['epochs']):
        for i, (x, y) in enumerate(filter_split(iterator(data), split='train')):
            # print(f'Step: {i}')

            predicted = model(torch.from_numpy(x)[None])
            loss = criterion(predicted, torch.tensor(y).long())

            # all function calls since zero_grad() were recorded. Compute gradients from call history and update weights
            loss.backward()
            optimizer.step()
            print(loss.item())

            # at each iteration, reset the gradients
            model.zero_grad()

    torch.save(model.state_dict(), './models/torch_ann/weights')


def test_ann(data, networkparameters):
    model = ANNClassifier(**networkparameters)

    model.load_state_dict(torch.load('./models/torch_ann/weights'))
    model.eval()  # turn on evaluation mode, which disables dropout

    expected, actual = [], []

    with torch.no_grad():
        for x, y in filter_split(iterator(data), split='test'):
            actual.append(int(np.argmax(np.squeeze(model(torch.from_numpy(x)[None])))))
            expected.append(int(np.squeeze(np.array(y))))

    print(actual)
    return expected, actual
