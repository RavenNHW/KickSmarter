from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import json
import pickle

pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None


def split_and_encode(df):
    """Splits the given dataframe into a train and test group, and encodes them separately using SKLearn's LabelEncoder()
---------------------
Input:
    df: pandas DataFrame
---------------------
Output: 
    encoded_train_df: pandas DataFrame
    The encoded training data for the dataset
    
    encoded_test_df: pandas DataFrame
    The encoding test data for the dataset
    
    LE_dict: Dict
    The label encoder dictionary used to decode results
    """
    df.set_index('id', inplace = True)
    train_df, test_df = train_test_split(df, test_size = .2, random_state = 6)
    
    col_list = ['category_parent_name', 'currency', 'country']

    encoded_train_df = train_df
    encoded_test_df = test_df
    le = LabelEncoder()
    LE_dict = {}

    for i in col_list:
        le.fit(encoded_train_df[i])
        encoded_train_df[i] = le.transform(encoded_train_df[i])
        encoded_test_df[i] = le.transform(encoded_test_df[i])
        LE_dict[i] = le.classes_

    return encoded_train_df, encoded_test_df, LE_dict

def extract_dates(df):
    """Extracts the start and end month from the dataframe, as well as the project length. Removes the original date columns afterwards.
---------------------
Input:
    df: pandas DataFrame
    Dataframe containing the 'launched_at' and 'deadline' columns, given that their dtypes are datetime64[s]
---------------------
Output:
    df: pandas DataFrame
    Dataframe containing the 'start_month', 'end_month', and 'project_length' columns"""
    df['start_month'] = pd.DatetimeIndex(df['launched_at']).month
    df['end_month'] = pd.DatetimeIndex(df['deadline']).month
    df['project_length'] = abs(df['launched_at'] - df['deadline'])
    df['project_length'] = [i.days for i in df['project_length']]
    
    df.drop(columns = ["launched_at", "deadline"], inplace = True)
    
    return df

def create_dicts(df):
    """Creates dictionaries for the 'currency_symbol', 'state', and 'country_displayable_name' columns.
---------------------
Input:
    df: pandas DataFrame
---------------------
Output:
    dict_list: list of dictionaries
    Each dictionary contains a key matching items from the dataframe, and a value matching items from dropped columns in the dataframe
    
    new_df: pandas DataFrame
    DataFrame with 'country_displayable_name' and 'currency_symbol' removed"""
    
    currency = df[['currency', 'currency_symbol']].groupby('currency').first()

    currency_dict = {k:v for (k,v) in zip(currency.index, currency.currency_symbol)}
    country_dict = {k:v for (k,v) in zip(df.country.unique(), df.country_displayable_name.unique())}
    state_dict = {1 : 'successful' , 0 : 'failed'}
    
    dict_list = [currency_dict, country_dict, state_dict]
    
    new_df = df.drop(columns = ['currency_symbol', 'country_displayable_name'])
    
    return dict_list, new_df

def split_and_pickle_df(df):
    """Splits the dataframe into two, one with the blurb feature and one without
---------------------
Input:
    df: pandas DataFrame
---------------------
Output:
    blurb_df: pandas DataFrame
    Contains "blurb" and "state" columns
    Pickled as "KS_blurb_data.pkl"
    
    df: pandas DataFrame
    contains all columns except for "blurb"
    Pickled as "KS_data.pkl" """
    
    blurb_df = df[['blurb', 'state']]
    blurb_df.to_pickle('data/KS_blurb_data.pkl')
    
    df['blurb_len'] = [len(i) 
                       if type(i) == str 
                       else 0 for i in df.blurb]

    df = df.drop(columns = ['blurb', 'name'])
    df.to_pickle('data/KS_data.pkl')
    
    return blurb_df, df
   
    
def drop_cols(df):
    """Drops columns not needed for the project
---------------------
Input:
    df: pandas DataFrame
---------------------
Output:
    df: pandas DataFrame"""
    
    df =df.drop(columns = [
   'source_url', 'slug', 'photo', 'pledged',
   'currency_trailing_code', 'fx_rate',
   'usd_pledged', 'usd_type', 'static_usd_rate',
   'created_at', 'state_changed_at', 'fx_rate',
   'creator', 'location', 'profile', 'urls',
   'current_currency', 'disable_communication',
    'is_starrable', 'backers_count', 'converted_pledged_amount', 'spotlight'
        
])
    return df

def drop_rows(df):
    """Drops duplicate projects, as well as any projects who's current currency is not USD
---------------------
Input:
    df: pandas DataFrame
---------------------
Output:
    df: pandas DataFrame
"""
    df.sort_values(by = 'state_changed_at', 
                   ascending = False,
                   inplace = True)
    
    df = df.drop_duplicates('id')
    
    df.dropna(axis = 1,
              how = 'all',
              inplace = True, )
    
    df = df[df['current_currency'] == 'USD']
  
    return df

def correct_dtypes(df):
    """Corrects datatypes for all columns that contain a date, or are a catagorical feature
---------------------
Input:
    df: pandas DataFrame
    The dataframe that contains the date and catagorical features
---------------------
Output: 
    df: pandas DataFrame
    The resulting dataframe, with the results sorted by the descending date. 
"""

    dates = ['created_at', 'deadline', 'launched_at', 'state_changed_at']
    
    categories = [
        'country', 'currency', 'currency_symbol',
        'category_parent_name'
    ]
    
    state_dict = {'successful' : 1, 'failed' : 0}
    
    for i in dates:
        df[i] = df[i].astype('datetime64[s]', errors = 'ignore')
    
    for i in categories:
        df[i] = df[i].astype('category', errors = 'ignore')
        
    df.state = df.state.map(state_dict)
    df.sort_values(by = 'state_changed_at', ascending = False, inplace = True)
    
    return df


    
def expand_category(df):
    
    """Expands the category column within the raw Kickstarter DataFrame. Returns the same dataframe with "category_parent_name" appended to the end.
---------------------
Input:
    df: A pandas.DataFrame object
---------------------    
Output:
    new_df: A pandas.DataFrame object
    
"""
    
    cat_df = expand_dict(df, "category")
    
    cat_df_cols = list(cat_df.columns)
    cat_df.columns = ["category_" + i for i in cat_df_cols]
    
    new_df = df.join(cat_df).drop(columns = [
        'category', 'category_slug', 'category_urls',
        'category_color', 'category_id', 'category_parent_id',
        'category_position'
    ])
    
    new_df.category_parent_name.fillna(value = new_df.category_name, inplace = True)
    new_df.drop(columns = 'category_name', inplace = True)
    
    return new_df

def expand_dict(df, col):
    """ A simple function for expanding a dictionary located within a DataFrame.
---------------------
Input:
    df: A pandas.DataFrame object
    col: str, a column within the DataFrame

---------------------
Output: 
    A pandas.DataFrame object
    """
    xp_dict_list = ([json.loads(i) for i in df[col]])
    return pd.DataFrame.from_dict(xp_dict_list)


def import_ks_data():
    """Imports the raw Kickstarter data for the project from
all CSVs and combines it into one file. Creates a pickle for quicker future loads as well.

---------------------
Output:
    KS_df: A pandas.DataFrame object
    
    Additionally saves 'raw_ks_data.p' in the data folder.
"""

    lst = list(range(1, 57))
    file_endings = [str(i).zfill(3) for i in lst]
    filepaths = ['data/Kickstarter_CSVs/Kickstarter'+ i + '.csv' for i in file_endings]
    
    KS_df = pd.read_csv('data/Kickstarter_CSVs/Kickstarter.csv')
    KS_df = KS_df[~KS_df['state'].isin(['canceled', 'live'])]
    
    for i in filepaths:
        n_df = pd.read_csv(i)
        n_df = n_df[~n_df.state.isin(['canceled', 'live'])]
        
        KS_df = KS_df.append(n_df, ignore_index = True)
    
    KS_df
    KS_df.to_pickle('data/raw_ks_data.p')
    return KS_df

def load_raw_ks_data():
    """Loads the kickstarter data if you have already run import_ks_data().
---------------------

Output:
    df: A pandas.DataFrame object"""
    
    df = pd.read_pickle('data/raw_ks_data.p')
    
    return df