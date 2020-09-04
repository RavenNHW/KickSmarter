import os
import pickle
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import SGDClassifier, RidgeClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt

def load_train_test_data(ex = False):
    
    """Loads the traning and test data to be used alongside modeling. Separates it into X(features) and y(targets) for both train and test. 
-----------------------------------------
Output: 
    X_train: pandas DataFrame
    y_train: pandas Series
    X_test: pandas DataFrame
    y_test: pandas Series
"""
    
    if ex == True: 
        train_df = pd.read_pickle('data_processing/data/KS_train_data.pkl')
        test_df = pd.read_pickle('data_processing/data/KS_test_data.pkl')

        X_train = train_df.drop(columns = ['state'])
        y_train = train_df.state

        X_test = test_df.drop(columns = ['state'])
        y_test = test_df.state
        
    else:
        train_df = pd.read_pickle('data/KS_train_data.pkl')
        test_df = pd.read_pickle('data/KS_test_data.pkl')

        X_train = train_df.drop(columns = ['state'])
        y_train = train_df.state

        X_test = test_df.drop(columns = ['state'])
        y_test = test_df.state
    
    return X_train, y_train, X_test, y_test

def plot_violin(fig, y, label, line_color = 'black', line_width = 1, fillcolor = 'red', opacity = .6):
    
    """Adds a single violin plot trace to an already plotly figure.
-----------------------------------------
Input:
    fig: plotly.graph_objs._figure.Figure
    The target figure for the violin plot to be added to
    
    y: pandas Series object
    Sets the y sample data or coordinates. See plotly documentation for more information
    
    label: str
    Sets the trace name
    
    line_color: str, default = 'black'
    The color for the trace lines. Can be a valid CSS color name, hex, RGB(A), HSL(A)
    
    line_width: int, default = 1
    The width for the trace lines
    
    fillcolor: str, default = 'red'
    The color for the trace lines. Can be a valid CSS color name, hex, RGB(A), HSL(A)
    
    opacity: str or float, default = .6
    The opacity of the trace, between 1 and 0
    
-----------------------------------------
Output: 
    returns plotly.graph_objs._figure.Figure object
    """
    
    return fig.add_trace(go.Violin(
        y=y, 
        name = label,
        box_visible=True,
        meanline_visible=True,
        line_color= line_color,
        line_width = line_width,
        fillcolor=fillcolor,
        opacity=opacity))

def plot_model_results(results, model_names, filepath, figure_title, figsize = (10, 8)):
    
    """Plots and saves an image of the plot.
-----------------------------------------
Input:
    results: list
    the results of the models, gained from test_models()
    
    model_names: list
    the names of models used, gained from test_models()
    
    filepath: str
    the filepath for the graph image to be saved to

-----------------------------------------
Output:
    plt.show()
    """
    
    plt.figure(figsize = figsize)
    plt.boxplot(results, labels = model_names, showmeans = True)
    plt.title(f'{figure_title}')
    plt.ylabel('Accuracy'); plt.xlabel('Model')
    plt.savefig(filepath)
    return plt.show()

def test_models(x_train, y_train, models, n_jobs = 2):
    """Test all models given using RepeatedStratifiedKFold. 
-----------------------------------------
Input: 
    x_train: pandas DataFrame
    Training data features
    
    y_train: pandas Series
    Training data targets
    
    models: Dict 
    Dictionary of models to be tested
    
    n_jobs: Int 
    The number of jobs to run in parallel
-----------------------------------------
Output:
    results_dict: Dict 
    Contains results and model names. """
    
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
   
    return results_dict


def stacked_model(models):
    """Creates a stacked model given a dictionary of SciKitLearn models
-----------------------------------------
Input: 
    models: Dict
    Contains model name and function.
-----------------------------------------
Output: 
    stack_model: Dict
    A new dictionary containing a SciKitLearn StackingClassifier object"""

    stack_m = [] 
    for model, m in models.items(): 
        stack_m.append((model, m))
    stack_model = StackingClassifier(estimators = stack_m, final_estimator = LogisticRegression(), cv = 3)
    models['stacked'] = stack_model
    
    return models


def save_cv_results(model_dict, filename):
    """Pickles the model's results
-----------------------------------------
Input: 
    model_dict: Dict
    The results from test_models()
    
    filename: str
    path to the target directory for the file to be saved

-----------------------------------------
Output:
    Pickles the given dictionary object"""
    
    pickle.dump(model_dict, open(filename, 'wb'))
    return 'Pickling Complete'

def run_gridsearch(classifier, X_train, y_train, X_test, y_test, params, n_jobs = 2, verbose = 0):
    
    """A function for performing a grid search on a given model.
Uses the training data and outputs the scores for the train and test data.
-----------------------------------------
Input: 
    classifier: Function 
    SKLearn classifier object
        
    X_train: pandas DataFrame
    The training features of the dataset
        
    y_train: pandas Series
    The training class for the dataset
        
    X_test: pandas DataFrame 
    The test features of the dataset
        
    y_test: pandas Series
    The test classes of the dataset
   
    params: Dict
    The parameters for classifier grid search

    n_jobs: Int, default = 2
    The number of jobs to run in parallel

    verbose: int, default = 0
    Controls the verbosity when fitting and predicting.
-----------------------------------------
Output:
    
forest_clf: The Grid Search object"""
    
    clf = GridSearchCV(
        estimator = classifier,
        param_grid = params,
        scoring = 'precision',
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
