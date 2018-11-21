from preprocessing import onehot_sampler
from models.lstm.train import train as train_lstm

import json


with open('hyperparameters.json', 'r') as param_file:
    params = json.load(param_file)

train_lstm(onehot_sampler(), params['LSTM'])
