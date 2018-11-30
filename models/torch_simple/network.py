import torch
import torch.nn.functional as f


class SimpleClassifier(torch.nn.Module):
    def __init__(self, input_size, output_size):
        """Network initialization

        Args:
            input_size: number of features in an observation
            output_size: number of classes in a prediction
        """

        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.layer1 = torch.nn.Linear(23, 100)
        self.layer2 = torch.nn.Linear(100, 20)
        self.layer3 = torch.nn.Linear(20, 4)

    def forward(self, data):
        """Given an observation, return the network prediction

        Args:
            data: a matrix of dimension [batch_size, input_features]
                batch_size: number of observations
                input_features: number of features for each observation
        Return:
            torch.tensor: the output of the network
        """

        # make sure data type is correct
        data = data.float()
        data = f.relu(self.layer1(data))
        data = f.relu(self.layer2(data))
        data = f.sigmoid(self.layer3(data))
        return data
