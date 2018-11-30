import torch


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
        self.activation = activation or torch.nn.Sigmoid()

        sizes = [input_size, *layer_sizes, output_size]
        lower, upper = iter(sizes), iter(sizes)
        next(upper)

        # torch can't find parameters inside lists
        self.layers = torch.nn.ModuleList(torch.nn.Linear(*window) for window in zip(lower, upper))
        self.dropout = torch.nn.Dropout(dropout)

        # activation transforms features to probability vector
        self.activation_final = torch.nn.Sigmoid()

    def forward(self, data):
        """Given an observation, update lstm state and return the network prediction

        Args:
            data: a matrix of dimension [batch_size, input_features]
                batch_size: number of observations
                input_features: number of features for each observation
        Return:
            torch.tensor: the output of the network
        """

        # make sure data type is correct
        data = data.float()

        for layer in self.layers[:-1]:
            data = self.dropout(self.activation(layer(data)))

        # print(self.layers[1].weight)

        return self.activation_final(self.activation(self.layers[-1](data)))
