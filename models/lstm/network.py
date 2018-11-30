import torch


class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_size, output_size, lstm_hidden_dim=20, lstm_layers=1, batch_size=1):
        """Network initialization

        Args:
            input_size: number of features in an observation
            output_size: number of classes in a prediction
            lstm_hidden_dim: number of features in each lstm hidden layer
            lstm_layers: number of layers in the lstm
        """

        super().__init__()
        # number of output nodes from LSTM, and input nodes to typical linear layer
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.batch_size = batch_size

        # outputs an intermediate feature vector
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=lstm_hidden_dim, num_layers=lstm_layers)

        # linear layer maps from intermediate feature space to class label
        self.linear = torch.nn.Linear(lstm_hidden_dim, output_size)
        # activation transforms features to probability vector
        self.activation_final = torch.nn.LogSoftmax(dim=1)

        self.lstm_state_init(batch_size)

    def lstm_state_init(self, batch_size):
        """sets the state of the lstm gates"""
        self.lstm_state = (
            torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim),
            torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim)
        )

    def forward(self, data):
        """Given an observation, update lstm state and return the network prediction

        Args:
            data: a matrix of dimension [batch_size, input_features]
                batch_size: number of observations at each time step
                input_features: number of features at each observation
        Return:
            torch.tensor: the output of the network
        """
        # [None] adds an axis for sequence length: number of time steps represented in the tensor, which is one
        # after making an observation, save the output and update the internal state
        lstm_out, self.lstm_state = self.lstm(data[None].float(), self.lstm_state)
        return self.activation_final(self.linear(lstm_out[0]))
