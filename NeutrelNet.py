import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('datafile_ordinal.csv', names=['PLOVER', 'EventDate', 'Latitude', 'Longitude', 'SourceBaseSector', 'TargetBaseSector'])
#df[['PLOVER']] = df[['PLOVER']].astype(float)

# d = pd.DataFrame(df)
# d.dtypes
X = df.drop(columns=['PLOVER'])
Y = df.drop(columns=['EventDate', 'Latitude', 'Longitude', 'SourceBaseSector', 'TargetBaseSector'])
#print(type(Y))
#print(type(X))
#print(Y)

# Split the dataset in two equal parts into 80:20 ratio for train:test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

tuned_parameters_NeuralNet = [{'hidden_layer_sizes': [(100,), (50,), (150,)],
                               'activation': ['identity', 'logistic', 'tanh', 'relu'], 'alpha': [0.0001, 0.001],
                               'learning_rate': ['constant', 'adaptive', 'invscaling']}]


scores = ['accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(MLPClassifier(), tuned_parameters_NeuralNet, cv=5, scoring='%s' % score)
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print("Detailed confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Accuracy Score: \n")
    print(accuracy_score(y_true, y_pred))

    print()