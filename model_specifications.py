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

from models.lstm.network import LSTMClassifier
from skorch import NeuralNetClassifier
import torch

model_specifications = [
    {
        "name": "Decision Tree",
        "class": DecisionTreeClassifier,
        "hyperparameters": {
            "criterion": [
                "gini",
                "entropy"
            ],
            "splitter": [
                "best",
                "random"
            ],
            "min_samples_split": [2, 3],
            "min_samples_leaf": [1, 5]
        }
    },
    {
        "name": "Neural Network",
        "class": MLPClassifier,
        "hyperparameters": {
            "hidden_layer_sizes": [
                [50, 10],
                [100, 20]
            ],
            "activation": [
                "relu",
                "tanh"
            ],
            "learning_rate": [
                "constant",
                "adaptive"
            ],
            "alpha": [0.0, 0.0001],
            "max_iter": [40]
        }
    },
    {
        "name": "SVM",
        "class": SVC,
        "hyperparameters": {
            "kernel": [
                "rbf",
                "linear"
            ],
            "gamma": [1e-3, 1e-4],
            "degree": [2, 3, 5],
            "C": [1, 10, 100, 1000],
            "max_iter": [40]
        }
    },
    {
        "name": "Logistic Regression",
        "class": LogisticRegression,
        "hyperparameters": {
            "penalty": [
                "l1",
                "l2"
            ],
            "C": [0.1, 1, 10],
            "fit_intercept": [True, False],
            "class_weight": [None, "balanced"],
            "max_iter": [40]
        }
    },
    {
        "name": "K Nearest Neighbors",
        "class": KNeighborsClassifier,
        "hyperparameters": {
            "n_neighbors": [1, 5, 10],
            "weights": [
                "uniform",
                "distance"
            ],
            "algorithm": [
                "ball_tree",
                "kd_tree",
                "brute"
            ],
            "p": [1, 2]
        }
    },
    {
        "name": "Bagging Classifier",
        "class": BaggingClassifier,
        "hyperparameters": {
            "n_estimators": [5, 10, 20],
            "max_samples": [0.5, 1, 2],
            "max_features": [1, 2, 3],
            "random_state": [0, 1, 2]
        }
    },
    {
        'name': 'Gaussian Naive Bayes',
        'class': GaussianNB,
        'hyperparameters': {}
    },
    {
        "name": "Random Forest",
        "class": RandomForestClassifier,
        "hyperparameters": {
            "n_estimators": [5, 10, 20],
            "criterion": [
                "gini",
                "entropy"
            ],
            "min_samples_split": [2, 3],
            "min_samples_leaf": [1, 5]
        }
    },
    {
        "name": "AdaBoost",
        "class": AdaBoostClassifier,
        "hyperparameters": {
            "n_estimators": [10, 40],
            "learning_rate": [0.5, 0.75, 1.0],
            "algorithm": [
                "SAMME",
                "SAMME.R"
            ],
            "random_state": [0, 1, 2]
        }
    },
    {
        "name": "Gradient Boosting Classifier",
        "class": GradientBoostingClassifier,
        "hyperparameters": {
            "learning_rate": [0.02, 0.1, 0.5],
            "n_estimators": [20, 50],
            "min_samples_split": [2, 3],
            "min_samples_leaf": [1, 5]
        }
    },
    {
        "name": "XGBoost",
        "class": XGBClassifier,
        "hyperparameters": {
            "learning_rate": [0.02, 0.1, 0.5],
            "n_estimators": [20, 50],
            "min_child_weight": [1, 3],
            "booster": [
                "gbtree",
                "gblinear",
                "dart"
            ]
        }
    },
    {
        "name": "LSTM",
        "class": NeuralNetClassifier,
        "kwargs": {
            "module": LSTMClassifier,
            "criterion": torch.nn.NLLLoss,
            "optimizer": torch.optim.SGD,
            "batch_size": 1,
            "max_epochs": 5
        },
        "hyperparameters": {
            "module__input_size": [23],
            "module__output_size": [4],

            "module__hidden_dim": [5, 20, 50],  # dimensionality of the hidden LSTM layers
            "module__lstm_layers": [1, 4],  # number of LSTM layers
            "module__batch_size": [1],

            "optimizer__lr": [0.001, 0.1, 0.5],
        }
    }
]
