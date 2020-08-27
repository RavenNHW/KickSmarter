import os
import pickle
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import SGDClassifier, RidgeClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC
    
import matplotlib.pyplot as plt
import seaborn as sns


def plot_model_results(results, model_names, filepath, figure_title, figsize = (10, 8)):
    
    """Plots and saves an image of the plot.
    Input:
    results: the results of the models, gained from test_models()
    model_names: the names of models used, gained from test_models()
    filepath: the filepath for the graph image to be saved to
    """
    
    plt.figure(figsize = figsize)
    plt.boxplot(results, labels = model_names, showmeans = True)
    plt.title(f'{figure_title}')
    plt.ylabel('Accuracy'); plt.xlabel('Model')
    plt.savefig(filepath)
    plt.show()

def test_models(x_train, y_train, models, n_jobs = 2):
    """
    Test all models given.
    
    This will test each model on its own using RepeatedStratifiedKFold then it will test a stacking classifier with every single model in the dictionary.  
    
    returns: vanilla_dict (contains results and model names)"""
    results = []
    model_names = []
    pbar = tqdm(models.items())
    
    for model, m in pbar: 
        pbar.set_description(f'Evaluating {model.upper()}')
        cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 10)
        scores = cross_val_score(m, x_train, y_train, scoring = 'precision', cv = cv, n_jobs = n_jobs, 
                                 error_score = 'raise')
        results.append(scores)
        model_names.append(model)
    vanilla_dict = {i:y for i,y in zip(model_names, results)}
   
    return vanilla_dict


def stacked_model(models):
    """Creates a stacked model given a dictionary of SciKitLearn models
    -----------------------------------------
    Input: 
        models: Dictionary containing the model name and function.
    
    Output: 
        stack_model: A new dictionary containing a SciKitLearn StackingClassifier object
    -----------------------------------------"""

    stack_m = [] 
    for model, m in models.items(): 
        stack_m.append((model, m))
    stack_model = StackingClassifier(estimators = stack_m, final_estimator = LogisticRegression(), cv = 3)
    models['stacked'] = stack_model
    
    return models


def save_cv_results(model_dict, filename):
    """
    Pickles the model's results
    
    Input: 
    
    model_names: list of model names 
    results: list of results 
    filename: str, path for the file to be saved
    
    """
    pickle.dump(model_dict, open(filename, 'wb'))
    return 'Done'

def run_gridsearch(classifier, X_train, y_train, X_test, y_test, params, n_jobs = 2, verbose = 0):
    
    """A function for performing a grid search using a random forest model.
    Uses the training data and outputs the scores for the train and test data.
    
    Input: 
    
    classifier: classifier object
    X_train: The training features of the dataset
    y_train: The training class for the dataset
    X_test: The test features of the dataset
    y_test: The test classes of the dataset
   
    params: The parameters for classifier grid search
    n_jobs: n_jobs
    verbose: verbose
    
    Output:
    
    forest_clf: The Grid Search object
    """
    
    clf = GridSearchCV(
        estimator = classifier,
        param_grid = params,
        scoring = 'precision'
        n_jobs = n_jobs,
        verbose = verbose, error_score = 0.0
    )
    
    clf.fit(X_train, y_train)
    
    print(f"""       Results
~~~~~~~~~~~~~~~~~~~~~
Train Score: {clf.score(X_train, y_train):.2f}
---
Test Score: {clf.score(X_test, y_test):.2f}
Best Parameters:
{clf.best_params_}
""")
    
    return clf
