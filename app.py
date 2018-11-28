from preprocessing import day_sampler, preprocess_sampler, test_train_split

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import numpy as np

import warnings
from collections import Counter

import json

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

sklearn_models = {
    'Decision Tree': DecisionTreeClassifier,
    'Neural Network': MLPClassifier,
    'SVM': SVC,
    'Gaussian Naive Bayes': GaussianNB,
    'Logistic Regression': LogisticRegression,
    'K Nearest Neighbors': KNeighborsClassifier,
    'Bagging Classifier': BaggingClassifier,
    'Random Forest': RandomForestClassifier,
    'AdaBoost': AdaBoostClassifier,
    'Gradient Boosting Classifier': GradientBoostingClassifier,
    'XGBoost': XGBClassifier
}


def evaluate(expected, actual, print_results=True):
    """Collect diagnostics about model performance

    Args:
        expected: array of expected values
        actual: array of actual values
    """

    if print_results:
        print("\nDetailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print(expected, actual)
        print(classification_report(expected, actual))

        print("\nDetailed confusion matrix:")
        print(confusion_matrix(expected, actual))

        print("Accuracy Score:")
        print(accuracy_score(expected, actual))

    return [
        precision_score(expected, actual, average='micro'),
        recall_score(expected, actual, average='micro'),
        f1_score(expected, actual, average='micro'),
        accuracy_score(expected, actual),
    ]


print_results = True


def run_lstm():
    from models.lstm.train import train as train_lstm, test_lstm
    # train_lstm(day_sampler, params['LSTM'])
    test_lstm(day_sampler, params['LSTM'])


with open('model_grids.json', 'r') as param_file:
    model_grids = json.load(param_file)


all_scores = []
all_parameters = []

split = test_train_split(preprocess_sampler(x_format='OneHot', y_format='Ordinal'))
x_train, y_train, x_test, y_test = [np.array(val) for val in split]


for model in model_grids:

    y_true, y_pred, best_params = None, None, None

    # catch warnings in bulk, show frequencies for each after grid search
    with warnings.catch_warnings(record=True) as warns:
        print(f'{model["name"]}: Tuning hyper-parameters')

        if model['library'] == 'sklearn':
            clf = GridSearchCV(sklearn_models[model['name']](), model['parameters'], cv=5, scoring='accuracy')
            clf.fit(x_train, y_train)
            y_true, y_pred = np.array(y_test)[:, 0], clf.predict(x_test)

            best_params = clf.best_params_

            if print_results:
                print("Grid scores on development set:")
                means = clf.cv_results_['mean_test_score']
                stds = clf.cv_results_['std_test_score']
                params = clf.cv_results_['params']

                for mean, std, params in zip(means, stds, params):
                    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

        warning_counts = dict(Counter([str(warn.category) for warn in warns]))
        if warning_counts:
            print('Warnings during grid search:')
            print(json.dumps(warning_counts, indent=4))

        scores = evaluate(y_true, y_pred, print_results=print_results)
        scores = [round(score, 4) for score in scores]

        all_scores.append([model['name'], *scores])
        all_parameters.append([
            model['name'],
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
