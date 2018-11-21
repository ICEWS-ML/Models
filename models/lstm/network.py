import torch


class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=20, lstm_layers=1, batch_size=1):
        """Network initialization

        Args:
            input_size: number of features in an observation
            output_size: number of classes in a prediction
            hidden_dim: number of features in each lstm hidden layer
            lstm_layers: number of layers in the lstm
            batch_size: number of observations at each time step
        """

        super().__init__()
        # number of output nodes from LSTM, and input nodes to typical linear layer
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.batch_size = batch_size

        # outputs an intermediate feature vector
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=lstm_layers)

        # linear layer maps from intermediate feature space to class label
        self.linear = torch.nn.Linear(hidden_dim, output_size)
        # activation transforms features to probability vector
        self.activation_final = torch.nn.LogSoftmax(dim=1)

        self.lstm_state = (
            torch.randn(self.lstm_layers, self.batch_size, self.hidden_dim),
            torch.randn(self.lstm_layers, self.batch_size, self.hidden_dim)
        )

    def forward(self, observation):
        """Given an observation, update lstm state and return the network prediction

        Args:
            observation: a matrix of dimension [sequence_length, batch_size, input_features]
                sequence length: number of time steps represented in the tensor
                batch_size: number of observations at each time step
                input_features: number of features at each observation
        Return:
            torch.tensor: the output of the network
        """

        # after making an observation, save the output and update the internal state
        lstm_out, self.lstm_state = self.lstm(observation, self.lstm_state)

        return self.activation_final(self.linear(lstm_out))
