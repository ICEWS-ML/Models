from models.torch_simple.network import SimpleClassifier
import numpy as np
import torch


# I want to evaluate generator functions if passed
def iterator(data):
    return data() if callable(data) else iter(data)


def filter_split(data, split='train'):
    for x, y, label in data:
        if label == split:
            yield x, y


def train_simple(data, networkparameters):

    model = SimpleClassifier(**networkparameters)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(100):
        for i, (x, y) in enumerate(filter_split(iterator(data), split='train')):

            predicted = model(torch.from_numpy(x)[None])
            loss = criterion(predicted, torch.tensor(y).long())

            # all function calls since zero_grad() were recorded. Compute gradients from call history and update weights
            loss.backward()
            optimizer.step()
            print(loss.item())

            # at each iteration, reset the gradients
            model.zero_grad()

    torch.save(model.state_dict(), './models/torch_simple/weights')


def test_simple(data, networkparameters):
    model = SimpleClassifier(**networkparameters)

    model.load_state_dict(torch.load('./models/torch_simple/weights'))
    model.eval()

    expected, actual = [], []

    with torch.no_grad():
        for x, y in filter_split(iterator(data), split='test'):
            actual.append(int(np.argmax(np.squeeze(model(torch.from_numpy(x)[None])))))
            expected.append(int(np.squeeze(np.array(y))))

    return expected, actual
