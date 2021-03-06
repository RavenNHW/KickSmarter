import os
import pickle
import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go
from datetime import datetime

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

def run_cv(model_names, X_train, y_train, save_results = True, filename = None):
    
    """A function that runs the get_models, test_models, and plot_cv_results functions. "All in one" function for cleaner code.
While saving the results is optional, it is on by default. If you do not select a filename then the default is set to datemonthyear_hour.minute
-----------------------------------------
Input:
    model_names: list or set
    A list or set of strings matching keys from the 'models' dictionary. Used to select the models to be tested
    
    X_train: pandas DataFrame
    The features from the dataset used for training 
    
    y_train: pandas Series
    The targets from the dataset used for training
    
    save_results: bool, default = True
    Whether or not you would like to save the results to the data_processing/models/CV_Results folder. 
    
    filename: str, default = None
    If 'None', is set to current date and time. The name of the model results files. 
"""
    
    models = get_models(model_names)

    results_dict = test_models(X_train, y_train, models)

    fig = plot_cv_results(results_dict)
    
    if save_results == True:
        if filename == None:
            filename = datetime.now().strftime("%d%m%Y_%H.%M")
        
        save_cv_results(results_dict, fig, filename)
    
    return fig.show() 
    
def load_cv(model_names, save_results = True, filename = None):
    
    """A function that pulls the information from run_cv, and runs the plot_cv_results and save_cv_results functions. "All in one" function for cleaner code.
While saving the results is optional, it is on by default. If you do not select a filename then the default is set to datemonthyear_hour.minute
-----------------------------------------
Input:
    model_names: list or set
    A list or set of strings matching keys from the 'models' dictionary. Used to select the models to be tested
    
    save_results: bool, default = True
    Whether or not you would like to save the results to the data_processing/models/CV_Results folder. 
    
    filename: str, default = None
    If 'None', is set to current date and time. The name of the model results files. 
"""
    
    if type(model_names) == list:
        model_names = set(model_names)
        
    elif type(model_names) == set:
        pass
    
    else:
        raise TypeError("'model_names' must be a list or set")
        
    og_results_dict = pickle.load(open('data_processing/models/CV_Results/VanillaResults_1.pkl', 'rb'))
    results_dict = {key:og_results_dict[key] for key in set(model_names) & set(og_results_dict)}
    
    fig = plot_cv_results(results_dict)
    
    if save_results == True:
        if filename == None:
            filename = datetime.now().strftime("%d%m%Y_%H.%M")
        
        save_cv_results(results_dict, fig, filename)
        
    return fig.show()
    
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

def save_cv_results(results, fig, filename):
    """Saves the results from cross-validation model testing
-----------------------------------------
Input:
    results: dict
    Dictionary of model results obtained from test_models()
    
    fig: plotly.graph_objs._figure.Figure
    Figure plotting the results of the cross-validation testing

    filename: str
    Name of the files, preferably in the format of [TESTTYPE]Results_[test_num]
-----------------------------------------
Output:
    Saves a PNG of the plotly fig, and pickles the dict object 
"""
    
    ext = ['pkl', 'png']

    pickle.dump(results,open(f'data_processing/models/CV_Results/{filename}.{ext[0]}', 'wb'))
    fig.write_image(f'data_processing/models/CV_Results/{filename}.{ext[1]}')

def plot_cv_results(data):
    """Plots the model cross-validation results.
-----------------------------------------
Input:
    data: pandas Dataframe or dict object
-----------------------------------------
Output:
    fig: plotly.graph_objs._figure.Figure
    Returns the figure object
    """
    
    if type(data) == pd.core.frame.DataFrame:
        dataframe = data
        
    elif type(data) == dict:
        dataframe = pd.DataFrame.from_dict(data)
        
    else:
        raise ValueError("Data must be in dataframe or dict format")
        
    
    fig = go.Figure()
    model_count = 0

    for i in dataframe: 
        plot_violin(fig,dataframe[i], i)
        model_count += 1

    fig.update_layout(
        title = dict(
            text = f"Cross validation precision Scores for {model_count} models",
            xanchor = 'center', x = .5),

        yaxis_zeroline = False, 
        height = 500, 
        showlegend = False, 
        yaxis_title = "Precision Score",
        xaxis_title = "Models"
        
    )
    
    return fig

def get_models(model_names):
    """Retrieve the models desired from the dictionary of models
-----------------------------------------
Input: 
    model_names: list or set
    The names of the models to be retrieved. Must be a key within the models dict object
-----------------------------------------
Output:
    model_dict: dict
    Dictionary containing the models to be used
    """
    if type(model_names) == list:
        model_names = set(model_names)
        
    elif type(model_names) == set:
        pass
    
    else:
        raise TypeError("'model_names' must be a list or set")
        
    model_dict = {key:models[key] for key in set(model_names) & set(models)}
    
    return model_dict

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
    results_dict = {i:y for i,y in zip(model_names, results)}
   
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

seed = 6

models = {'LogReg': LogisticRegression(),
          'KNN': KNeighborsClassifier(),
          'DT': DecisionTreeClassifier(random_state = seed), 
          'Gaussian': GaussianNB(),
          'Multinomial': MultinomialNB(),
          'LDA': LinearDiscriminantAnalysis(),
          'LinearSVC': LinearSVC(max_iter = 1250, random_state = seed),
          'SGD': SGDClassifier(random_state = seed),  
          'ADA': AdaBoostClassifier(random_state = seed),
          'Bagging': BaggingClassifier(random_state = seed), 
          'Ridge': RidgeClassifier(random_state = seed),
          'RF': RandomForestClassifier(random_state = seed),
          'GradientBoost' : GradientBoostingClassifier(random_state = seed)}