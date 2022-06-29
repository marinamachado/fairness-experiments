import numpy as np

# algoritmos de classificação
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

hidden_layer = [5, 8, 15, (10, 5), (8, 5), (15, 5), (7, 6, 5)]

classifiers_settings_test = {
    'Random Forest': [RandomForestClassifier,
          {'n_estimators' : np.arange(120, 220, 20)}],
    'Naive Bayes': [GaussianNB, {}],
    'Decision Tree' : [DecisionTreeClassifier,
          {'criterion' : ['gini', 'entropy'], 'splitter' : ['best', 'random']}]
}

classifiers_basic = {
    'Random Forest': [RandomForestClassifier,
          {}],
    'Naive Bayes': [GaussianNB, {}],
    'Decision Tree' : [DecisionTreeClassifier,{}]
}

classifiers_settings_test_weights = {
    'Random Forest': [RandomForestClassifier,
          {'n_estimators' : np.arange(120, 220, 20)}],
    'Naive Bayes': [GaussianNB, {}],
    'Decision Tree' : [DecisionTreeClassifier,
          {'criterion' : ['gini', 'entropy'], 'splitter' : ['best', 'random']}]
}

final_classifiers_settings = {
    
    'Decision Tree' : [DecisionTreeClassifier,
                      {'criterion' : ['gini', 'entropy'], 'splitter' : ['best', 'random'],'max_depth': np.arange(3, 16,4)}],
    'KNN' : [KNeighborsClassifier,
            {'n_neighbors' : [1, 3, 5, 7],'weights':['uniform', 'distance'] }],
    'Logistic Regression' : [LogisticRegression,
                            {'C': [0.75, 1, 1.25], 'class_weight': [None, 'balanced'] , 'n_jobs' : [-1]}],
    'MLP' : [MLPClassifier,
            {'hidden_layer_sizes' : hidden_layer, 'max_iter': [5000], 'alpha': 10.0 ** -np.arange(1, 10,5)}], 
    'Naive Bayes1' : [GaussianNB, {}],
    'Naive Bayes2' : [BernoulliNB,{'alpha' : [ 0.5, 0.75, 1.0, 1.25,1.5]}],
    
    'Random Forest' : [RandomForestClassifier,
                      {'n_estimators' : np.arange(50, 600, 50),'max_features': ['auto', 'sqrt'], 'n_jobs' : [-1]}],
    
    'SVM1' : [SVC,
            {'kernel' : ['poly'], 'degree': [2, 3, 4]}],
    'SVM2' : [SVC, {'kernel' : ['rbf','linear'],'gamma': [1,0.1,0.01,0.001]}]
}

final_classifiers_settings_weights = {
    'Decision Tree' : [DecisionTreeClassifier,
                      {'criterion' : ['gini', 'entropy'], 'splitter' : ['best', 'random'],'max_depth': np.arange(3, 16,4)}],
    'Logistic Regression' : [LogisticRegression,
                            {'C': [0.75, 1, 1.25], 'class_weight': [None, 'balanced'] , 'n_jobs' : [-1]}],
    'Naive Bayes1' : [GaussianNB, {}],
    'Naive Bayes2' : [BernoulliNB,{'alpha' : [ 0.5, 0.75, 1.0, 1.25,1.5]}],
    'Random Forest' : [RandomForestClassifier,
                      {'n_estimators' : np.arange(50, 600, 50),'max_features': ['auto', 'sqrt'], 'n_jobs' : [-1]}],
    'SVM1' : [SVC,
            {'kernel' : ['poly'], 'degree': [2, 3, 4]}],
    'SVM2' : [SVC, {'kernel' : ['rbf','linear'],'gamma': [1,0.1,0.01,0.001]}]
}

lfr_settings = {'German':{'k':10, 'Ax':0.1, 'Ay':1.0, 'Az':2.0,'seed':42,'maxiter':5000,'maxfun':5000},
                'Titanic':{'k':10, 'Ax':0.1, 'Ay':1.0, 'Az':2.0,'seed':42,'maxiter':5000,'maxfun':5000},
                'Adult':{'k':10, 'Ax':0.1, 'Ay':1.0, 'Az':2.0,'seed':42,'maxiter':5000,'maxfun':10000},
                'Compas':{'k':10, 'Ax':0.1, 'Ay':1.0, 'Az':2.0,'seed':42,'maxiter':5000,'maxfun':50000},
                'Arrhythmia':{'k':5, 'Ax':0.01, 'Ay':1.0, 'Az':50.0,'seed':42,'maxiter':5000,'maxfun':100000},
                'Contraceptive':{'k':10, 'Ax':0.1, 'Ay':1.0, 'Az':2.0,'seed':42,'maxiter':5000,'maxfun':5000},
                'Alcohol':{'k':7, 'Ax':0.1, 'Ay':1.0, 'Az':50.0,'seed':42,'maxiter':5000,'maxfun':5000},
                'Canabis':{'k':5, 'Ax':0.01, 'Ay':1.0, 'Az':50.0,'seed':42,'maxiter':5000,'maxfun':5000},
                'Heroin':{'k':10, 'Ax':0.01, 'Ay':1.0, 'Az':50.0,'seed':42,'maxiter':5000,'maxfun':5000},'Bank':{}}



