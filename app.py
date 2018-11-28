from preprocessing import day_sampler, preprocess_sampler

import json

with open('hyperparameters.json', 'r') as param_file:
    params = json.load(param_file)


def run_lstm():
    from models.lstm.train import train as train_lstm, test_lstm
    # train_lstm(day_sampler, params['LSTM'])
    test_lstm(day_sampler, params['LSTM'])


def run_random_forest():
    from models.random_forest.model import train_random_forest, test_random_forest
    sampler = preprocess_sampler(x_format='OneHot', y_format='Ordinal')
    train_random_forest(sampler, params['RandomForest'])
    test_random_forest(sampler)

run_random_forest()
