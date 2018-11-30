from preprocess import dataset_sampler, test_train_split

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import numpy as np

import warnings
from collections import Counter

import json

from model_specifications import model_specifications
from sklearn.model_selection import GridSearchCV


print_results = True


def evaluate(expected, actual):
    """Collect diagnostics about model performance

    Args:
        expected: array of expected values
        actual: array of actual values
    """

    if print_results:
        print("\nDetailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print(classification_report(expected, actual))

        print("\nDetailed confusion matrix:")
        print(confusion_matrix(expected, actual))

        print("Accuracy Score:")
        print(accuracy_score(expected, actual))

    return [
        precision_score(expected, actual, average='macro'),
        recall_score(expected, actual, average='macro'),
        f1_score(expected, actual, average='macro'),
        accuracy_score(expected, actual),
    ]


def run_lstm():
    from models.torch_lstm.train import train_lstm, test_lstm
    sampler = lambda: dataset_sampler(x_format='OneHot', y_format='Ordinal')

    params = {
        'input_size': 23,
        'output_size': 4,

        'lstm_hidden_dim': 20,
        'lstm_layers': 2,
        'batch_size': 1
    }
    train_lstm(sampler, params)
    evaluate(*test_lstm(sampler, params))


def run_ann():
    from models.torch_ann.train import train_ann, test_ann
    sampler = lambda: dataset_sampler(x_format='OneHot', y_format='Ordinal')

    params = {
        'input_size': 23,
        'output_size': 4,
        'layer_sizes': (100, 20)
    }
    train_ann(sampler, params)
    evaluate(*test_ann(sampler, params))


def run_simple():
    from models.torch_simple.train import train_simple, test_simple
    sampler = lambda: dataset_sampler(x_format='OneHot', y_format='Ordinal')

    params = {
        'input_size': 23,
        'output_size': 4
    }
    train_simple(sampler, params)
    evaluate(*test_simple(sampler, params))


def run_keras():
    from models.keras_ann.network import train_keras, test_keras
    sampler = lambda: dataset_sampler(x_format='OneHot', y_format='OneHot')
    trainparams = {
        'epochs': 200,
        'batch_size': 10
    }
    train_keras(sampler, trainparams)
    test_keras(sampler)


# this code only runs if you run this file explicitly
if __name__ == '__main__':

    all_scores = []
    all_parameters = []

    for model_spec in model_specifications:

        split = test_train_split(model_spec['datasource']())
        x_train, y_train, x_test, y_test = [np.array(val) for val in split]

        # remove the singleton axis from y if ordinal, and ensure long datatype
        y_train, y_test = [np.squeeze(val.astype(np.int64)) for val in (y_train, y_test)]

        # catch warnings in bulk, show frequencies for each after grid search
        with warnings.catch_warnings(record=True) as warns:

            print(f'{model_spec["name"]}: Tuning hyper-parameters')

            # create an instance of the model
            model = model_spec['class'](**(model_spec.get('kwargs', {})))

            search = GridSearchCV(model, param_grid=model_spec['hyperparameters'], cv=5, scoring='accuracy')
            search.fit(x_train, y_train)

            y_true, y_pred = y_test, search.predict(x_test)

            best_params = search.best_params_

            if print_results:
                print("Grid scores on development set:")
                means = search.cv_results_['mean_test_score']
                stds = search.cv_results_['std_test_score']
                params = search.cv_results_['params']

                for mean, std, params in zip(means, stds, params):
                    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

            warning_counts = dict(Counter([str(warn.category) for warn in warns]))
            if warning_counts:
                print('Warnings during grid search:')
                print(json.dumps(warning_counts, indent=4))

            scores = evaluate(y_true, y_pred)
            scores = [round(score, 4) for score in scores]

            all_scores.append([model_spec['name'], *scores])
            all_parameters.append([
                model_spec['name'],
                *[f'{key}={value}' for key, value in best_params.items()]
            ])

    with open('./scores.csv', 'w') as file:
        file.write(', '.join([
            'Algorithm',
            'Avg Precision',
            'Avg Recall',
            'Avg F1',
            'Accuracy'
        ]) + '\n')
        file.writelines([', '.join([str(j) for j in i]) + '\n' for i in all_scores])

    with open('./parameters.csv', 'w') as file:
        file.writelines([', '.join(i) + '\n' for i in all_parameters])
