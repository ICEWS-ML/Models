import torch
import torch.nn.functional as f


class ANNClassifier(torch.nn.Module):
    def __init__(self, input_size, output_size, layer_sizes=(), activation=None, dropout=0.0):
        """Network initialization

        Args:
            input_size: number of features in an observation
            output_size: number of classes in a prediction
        """

        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layer_sizes = layer_sizes
        self.activation = activation or torch.nn.Sequential()  # empty sequential is a no-op

        self.dropout = dropout

        sizes = [input_size, *layer_sizes, output_size]
        lower, upper = iter(sizes), iter(sizes)
        next(upper)

        # torch can't find parameters inside lists
        self.layers = torch.nn.ParameterList(torch.nn.Linear(*window) for window in zip(lower, upper))

        # activation transforms features to probability vector
        self.activation_final = torch.nn.LogSoftmax(dim=1)

    def forward(self, data):
        """Given an observation, update lstm state and return the network prediction

        Args:
            data: a matrix of dimension [batch_size, input_features]
                batch_size: number of observations
                input_features: number of features for each observation
        Return:
            torch.tensor: the output of the network
        """
        for layer in self.layers[:-1]:
            data = f.dropout(self.activation(layer(data)), self.dropout)
        return self.activation_final(self.layers[-1])
