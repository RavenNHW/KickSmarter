import pandas as pd
import numpy as np
import json

# def import_functs():
    
#     "Imports the functions required for the data cleaning notebook"
    
#     import pandas as pd
#     pd.set_option('display.max_columns', None)

seed = 6

def encode_text(train_df, test_df):
    col_list = ['category_parent_name', 'currency', 'country']
    
    encoded_train_df = df_train
    encoded_test_df = df_test
    
    le = LabelEncoder()
    LE_dict = {}

    for i in col_list:
        le.fit(encoded_df[i])
        encoded_train_df[i] = le.transform(encoded_train_df[i])
        encoded_test_df[i] = le.transform(encoded_test_df[i])
        LE_dict[i] = le.classes_
        
    return encoded_train_df, encoded_test_df, LE_dict

def create_dicts(df):
    
    currency = df[['currency', 'currency_symbol']].groupby('currency').first()

    currency_dict = {k:v for (k,v) in zip(currency.index, currency.currency_symbol)}
    country_dict = {k:v for (k,v) in zip(df.country.unique(), df.country_displayable_name.unique())}
    state_dict = {1 : 'successful' , 0 : 'failed'}
    
    dict_list = [currency_dict, country_dict, state_dict]
    
    new_df = df.drop(columns = ['currency_symbol', 'country_displayable_name'])
    
    return dict_list, new_df

def split_and_pickle_df(df):
    blurb_df = df[['blurb', 'state']]
    name_df = df[['name', 'state']]
    blurb_df.to_pickle('data/KS_blurb_data.pkl')
    
    df['blurb_len'] = [len(i) 
                       if type(i) == str 
                       else 0 for i in df.blurb]

    df = df.drop(columns = ['blurb', 'name'])
    df.to_pickle('data/KS_data.pkl')
    
    return blurb_df, df
   
    
def drop_cols(df):
    df =df.drop(columns = [
   'source_url', 'slug', 'photo', 'pledged',
   'currency_trailing_code', 'fx_rate',
   'usd_pledged', 'usd_type', 'static_usd_rate',
   'created_at', 'state_changed_at', 'fx_rate',
   'creator', 'location', 'profile', 'urls',
   'current_currency','category_name'
])
    return df

def drop_rows(df):
    df.sort_values(by = 'state_changed_at', 
                   ascending = False,
                   inplace = True)
    
    df = df.drop_duplicates('id')
    
    df.dropna(axis = 1,
              how = 'all',
              inplace = True)
    
    df = df[df['current_currency'] == 'USD']
  
    return df

def correct_dtypes(df):
#Correcting date values, based off of my local timezone, EST

    dates = ['created_at', 'deadline', 'launched_at', 'state_changed_at']
    
    categories = [
        'country', 'currency', 'currency_symbol',
        'category_name', 'category_parent_name'
    ]
    
    state_dict = {'successful' : 1, 'failed' : 0}
    
    for i in dates:
        df[i] = df[i].astype('datetime64[s]', errors = 'ignore')
    
    for i in categories:
        df[i] = df[i].astype('category', errors = 'ignore')
        
    df.state = df.state.map(state_dict)

    return df

    

    df.state = df.state.map(state_dict)
    #To help reduce the amount of data, i'll be converting the above into category datatypes or bools

    df.sort_values(by = 'state_changed_at', ascending = False, inplace = True)
    
def expand_category(df):
    
    """Expands the category column within the raw Kickstarter DataFrame. Returns the same dataframe with "category_name" and "category_parent_name" appended to the end.
---------------------
Input:
    df: A pandas.DataFrame object
    
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
    
    return new_df

def expand_dict(df, col):
    """ A simple function for expanding a dictionary located within a DataFrame.
---------------------
Input:
    df: A pandas.DataFrame object
    col: str, a column within the DataFrame
    
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
    filepaths = ['data/Kickstarter'+ i + '.csv' for i in file_endings]
    
    KS_df = pd.read_csv('data/Kickstarter.csv')
    KS_df = KS_df[~KS_df['state'].isin(['canceled', 'live'])]
    
    for i in filepaths:
        n_df = pd.read_csv(i)
        n_df = n_df[~n_df.state.isin(['canceled', 'live'])]
        
        KS_df = KS_df.append(n_df, ignore_index = True)

    KS_df.to_pickle('data/raw_ks_data.p')
    return KS_df

def load_raw_ks_data():
    """Loads the kickstarter data if you have already run import_ks_data().
---------------------

Output:
    df: A pandas.DataFrame object"""
    
    df = pd.read_pickle('data/raw_ks_data.p')
    
    return df