from preprocess import dataset_sampler

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

model_specifications = [
    {
        "name": "Decision Tree",
        "class": DecisionTreeClassifier,
        "datasource": lambda: dataset_sampler(x_format='OneHot', y_format='Ordinal'),
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
        "datasource": lambda: dataset_sampler(x_format='OneHot', y_format='Ordinal'),
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
            "max_iter": [5000]
        }
    },
    {
        "name": "SVM",
        "class": SVC,
        "datasource": lambda: dataset_sampler(x_format='OneHot', y_format='Ordinal'),
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
        "datasource": lambda: dataset_sampler(x_format='OneHot', y_format='Ordinal'),
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
        "datasource": lambda: dataset_sampler(x_format='OneHot', y_format='Ordinal'),
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
        "datasource": lambda: dataset_sampler(x_format='OneHot', y_format='Ordinal'),
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
        "datasource": lambda: dataset_sampler(x_format='OneHot', y_format='Ordinal'),
        'hyperparameters': {}
    },
    {
        "name": "Random Forest",
        "class": RandomForestClassifier,
        "datasource": lambda: dataset_sampler(x_format='OneHot', y_format='Ordinal'),
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
        "datasource": lambda: dataset_sampler(x_format='OneHot', y_format='Ordinal'),
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
        "datasource": lambda: dataset_sampler(x_format='OneHot', y_format='Ordinal'),
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
        "datasource": lambda: dataset_sampler(x_format='OneHot', y_format='Ordinal'),
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
    }
]


try:
    from models.torch_simple.network import SimpleClassifier
    from models.torch_ann.network import ANNClassifier
    from models.torch_lstm.network import LSTMClassifier

    from skorch import NeuralNetClassifier
    import torch

    model_specifications.extend([
        {
            "name": "TorchSimple",
            "class": NeuralNetClassifier,
            "datasource": lambda: dataset_sampler(x_format='OneHot', y_format='Ordinal'),
            "kwargs": {
                "module": SimpleClassifier,
                "criterion": torch.nn.CrossEntropyLoss,
                "optimizer": torch.optim.SGD,
                "batch_size": 1,
                "max_epochs": 100
            },
            "hyperparameters": {
                "module__input_size": [23],
                "module__output_size": [4],

                "optimizer__lr": [0.001, 0.1, 0.5],
            }
        },
        {
            "name": "TorchLSTM",
            "class": NeuralNetClassifier,
            "datasource": lambda: dataset_sampler(x_format='OneHot', y_format='Ordinal'),
            "kwargs": {
                "module": LSTMClassifier,
                "criterion": torch.nn.CrossEntropyLoss,
                "optimizer": torch.optim.SGD,
                "batch_size": 1,
                "max_epochs": 100
            },
            "hyperparameters": {
                "module__input_size": [23],
                "module__output_size": [4],

                "module__lstm_hidden_dim": [5, 20, 50],  # dimensionality of the hidden LSTM layers
                "module__lstm_layers": [1, 4],  # number of LSTM layers
                "module__batch_size": [1],

                "optimizer__lr": [0.001, 0.1, 0.5],
            }
        },
        {
            "name": "TorchANN",
            "class": NeuralNetClassifier,
            "datasource": lambda: dataset_sampler(x_format='OneHot', y_format='Ordinal'),
            "kwargs": {
                "module": ANNClassifier,
                "criterion": torch.nn.CrossEntropyLoss,
                "optimizer": torch.optim.SGD,
                "batch_size": 10,
                "max_epochs": 100
            },
            "hyperparameters": {
                "module__input_size": [23],
                "module__output_size": [4],

                "module__layer_sizes": [[200, 50], [100]],
                "module__dropout": [0.0, 0.5],

                "optimizer__lr": [0.5],
            }
        }
    ])

except ImportError:
    print('PyTorch models were not loaded because PyTorch is not installed.')


try:
    from keras.wrappers.scikit_learn import KerasClassifier
    from models.keras_ann.network import create_model

    model_specifications.extend([
        {
            "name": "KerasANN",
            "class": KerasClassifier,
            "datasource": lambda: dataset_sampler(x_format='OneHot', y_format='Ordinal'),
            "kwargs": {
                "build_fn": create_model
            },
            "hyperparameters": {
                'epochs': [100, 200]
            }
        }
    ])
except ImportError:
    print('Keras models were not loaded because Keras is not installed.')
