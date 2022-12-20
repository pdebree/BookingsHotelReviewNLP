
# pandas and numpy are used for data format, in order to allow for easier manipulation.
import pandas as pd
import numpy as np

# seaborn and variables matplotlib packages are used for visualiations.
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Scipy's Stats gives us access to basic statistical analysis packages
from scipy import stats

# The Stats model api gives us access to regression analysis packages 
import statsmodels.api as sm


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from sklearn.utils import resample

from sklearn.preprocessing import OneHotEncoder


from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# library for stemming
import nltk
from nltk.corpus import stopwords 

import ds_utils

def hotel_missing_explo(df):
    
    # checking the number of missing values for each variable. 
    print('Missing values by column:', df.isna().sum(), '\n', sep='\n')
    
    # check if all missing values from lat are also missing from lon
    if (df[df['lat'].isna() == True].index == df[df['lng'].isna() == True].index).all():
        print("\tAll missing values from \'lat' are also missing \'lng' values.\n\n")
    else:
        print("\tMissing values come from different rows\n\n")
        
    # Finding the addresses of the locations that are missing data. 
    df_coord_missing = df[df['lat'].isna()].groupby('Hotel_Address')['Hotel_Address'].count().to_frame()
    print("Hotel Addresses with missing \'lat\' and \'lon\' values:", df_coord_missing, "\n", sep="\n")
    
    # Finding the intersection between hotels that are missing data and hotels that are not
    print('\t',len(np.intersect1d(df_coord_missing.index, 
                   np.array(df[df['lat'].isna() == False]['Hotel_Address'].unique()))), 
          ' hotels that are missing data have coordinate location elsewhere in the dataset', sep='')


def token_range_vis(vec_model, train_matrix):
    """
    Creates a visualisation of a the distribution of features in the passed in matrix created by a vectorisation model. 
    """
    
    token_counts = pd.DataFrame(
        {"counts": train_matrix.toarray().sum(axis=0)},
         index=vec_model.get_feature_names_out()).sort_values("counts", ascending=False)
    
    plt.figure()
    token_counts.plot(kind="bar", legend=False)
    plt.xlabel("Tokens")
    plt.ylabel("Counts")
    plt.title(f'Distribution of Tokens in Training Matrix with {train_matrix.shape[1]} features')
    if (train_matrix.shape[1] > 25):
        plt.xticks([])
    else:
        plt.xticks(range(0,len(token_counts.index)), token_counts.index)
    plt.show()


def base_stop_stem(X_train, X_test, y_train,  y_test, df, vector=ds_utils.bagofwords, tokenizer=ds_utils.rev_tokenizer):
    """
    Creates three logistic regression models based on three different vectorisations.
        1) A Base Model with the default parameters
        2) A Vectorisation that considers removing stop words
        3) A Vectorisation that employs a specific tokenisation (the default being the rev_tokenizer defined above).
        
    The default vectoriser is a CountVectorizer, initiated in the bagofwords function, but other vectorisers may be passed in.
    """
    # Initialising the base model 
    nmodel, ntrain_mat, ntest_mat = vector(X_train, X_test)
    nbase, nbase_train, nbase_test = ds_utils.sk_logreg(ntrain_mat, ntest_mat, y_train, y_test, c=0.1)

    # Creating a pandas dataframe to keep track of our feature engineering 
    df = pd.concat([df, pd.DataFrame(
                      
        {'model': [nbase], 
         'parameters': ['c=0.1'], 
         'train_score':[nbase_train], 
         'test_score':[nbase_test], 
         'vectoriser':[vector],
        'number of features': [ntrain_mat.shape[1]]})])
    
    print(df)

    # Removing stop words from our vectorisation
    nmodel_stop, ntrain_mat_stop, ntest_mat_stop = vector(X_train, X_test, stop_words_='english')
    nstop, nbase_train_stop, nbase_test_stop = ds_utils.sk_logreg(ntrain_mat_stop, ntest_mat_stop, y_train, y_test, c=0.1)

    # Adding stop-word model to tracking dataframe
    df = pd.concat([df,  pd.DataFrame(
        {'model': [nstop], 
         'parameters': ['c=0.1, stopwords=english'], 
         'train_score':[nbase_train_stop], 
         'test_score':[nbase_test_stop], 
         'vectoriser':[vector],
        'number of features': [ntrain_mat_stop.shape[1]]})])

    # Adding our tokeniser (which deals with stop words - so we do not need to pass this in)
    nmodel_stem, ntrain_mat_stem, ntest_mat_stem = vector(X_train, X_test,
                                            tokenizer_= ds_utils.rev_tokenizer)
    nstem_model, nstem_train, nstem_test = ds_utils.sk_logreg(ntrain_mat_stem, ntest_mat_stem, y_train, y_test, c=0.1)

    # adding tokenised vector to dataframe
    df = pd.concat([df,  pd.DataFrame(
        {'model': [nstem_model], 
         'parameters': ['c=0.1, nltk stemmer'], 
         'train_score':[nstem_train], 
         'test_score':[nstem_test], 
         'vectoriser':[vector],
        'number of features': [ntrain_mat_stem.shape[1]]})])
    
    return df 


def vectoriser_min_max_explore(X_train, X_test, y_train, y_test, df,
                               vector=ds_utils.bagofwords, 
                               model=LogisticRegression(),
                               stop_words_=None, 
                               tokenizer_=ds_utils.rev_tokenizer, 
                               min_df__=None, max_features__=None):
    """
    """
    
    train_acc = []
    test_acc = []
    dfs = [df]
    
    for m in min_df__:
        vec_model, vec_train, vec_test = vector(X_train, X_test, stop_words_=stop_words_, min_df_=m, 
                                                      tokenizer_=tokenizer_, max_features_=max_features__)

        model, train_score, test_score = ds_utils.sk_logreg(vec_train, vec_test, y_train, y_test, c=0.1)

        df_new = pd.DataFrame(
            {'model': [model], 
             'parameters': ['c=0.1, nltk stemmer'], 
             'train_score':[train_score], 
             'test_score':[test_score], 
             'vectoriser':[vector],
             'number of features': [vec_train.shape[1]]})
        
        dfs.append(df_new)
        
        token_range_vis(vec_model, vec_train)
        train_acc.append(train_score)
        test_acc.append(test_score)
        
    df = pd.concat(dfs, axis=0) 

    if (len(train_acc) > 1):
        plt.figure()
        plt.plot(train_acc, label="Training")
        plt.plot(test_acc, label="Testing")
        plt.xticks(ticks = range(0, len(min_df__)), labels=min_df__)
        plt.xlabel("Minimum Document Frequencies")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
    else:
        print(f'Train Score: {train_acc[0]}')
        print(f'Test Score: {test_acc[0]}')
        
    return df



def hotel_encoding(hr_samp):
    
    # binning
    hr_samp['days_since_review_bins'] = pd.cut(hr_samp['days_since_review'], bins=[-1, 175, 350, 525,np.inf], 
                                               labels=['0-175','175-350','350-525','525+'])
    
    # dummy variables
    hr_samp['frequent_reviewer'] = np.where(hr_samp['Total_Number_of_Reviews_Reviewer_Has_Given'] >= 3, 1, 0)
    
    # encoding
    hr_samp['Country'] = hr_samp['Hotel_Address'].str.split(' ').apply(lambda x: x[-1]) 
    
    hr_samp['uk_reviewer'] = hr_samp['Reviewer_Nationality'] == ' United Kingdom '
    hr_samp['uk_reviewer'] = hr_samp['uk_reviewer'].astype(int)

    hr_samp['usa_reviewer'] = hr_samp['Reviewer_Nationality'] == ' United States of America '
    hr_samp['usa_reviewer'] = hr_samp['usa_reviewer'].astype(int)
   
    
    hr_samp['Tags_list'] = hr_samp['Tags'].apply(lambda x: x.strip('"[\'  \']"').split(' \', \' '))
    hr_samp = ds_utils.add_list_matrix(hr_samp, 'Tags_list')
    
    # create list of non-binary binned variables that need encoding
    cols_to_encode =  ['Country', 'days_since_review_bins']
    
    # encode 
    # days since review bins
    # country
    
    # loop over list of binned variable
    for i in cols_to_encode:
        # encode binned variable and add columsns to original dataframe 
        hr_samp = pd.merge(hr_samp, pd.DataFrame(data=ds_utils.ohe_sparse(hr_samp[i])).set_index(hr_samp.index.values),
                        left_index=True, right_index=True).drop(columns=[i])
    
    hr_samp = hr_samp.drop(columns=['days_since_review', 'Total_Number_of_Reviews_Reviewer_Has_Given', 
                                    'Hotel_Address', 'Hotel_Name', 'Reviewer_Nationality', 
                                    'Review_Date', 'Reviewer_Score', 'Tags', 'Tags_list'])
    
    # make all columns have no white space (could come from the tags
    hr_samp = ds_utils.remove_attribute_whitespace(hr_samp)
    
    
    return hr_samp
    

def train_test_downsample(hr_samp, test_size_=0.22):
    
    # Creating X variable from all columns except our dependent variable
    X = hr_samp.drop(columns=['good_review'])
    
    # Creating y variable from our dependent variable
    y = hr_samp['good_review']
    
    # Here we performing our train test spliot
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_)
    
    
    print('Before Downsampling:')
    print('Number of negative \'good_review\' values in training set:', X_train[y_train == 0].shape[0])
    print('Number of positive \'good_review\' values in training set:', X_train[y_train == 1].shape[0])
    
    
    print('\nAfter Downsampling:')
    
    X_train_bal, y_train_bal = train_downsample(X_train, y_train)

    print('\nTest Size:', len(y_test))
    print('Train Size:', len(y_train_bal))
    print('Total Sample Size: ', len(y_test) + len(y_train_bal), ' (proportionally: ',round(len(y)/511944, 2),')', sep='')
    print('New Train/Test Split Proportion:', round(len(y_test)/(len(y_test) + len(y_train_bal)), 3))
    
    
    X_train_bal.reset_index(drop=True, inplace=True)
    y_train_bal.reset_index(drop=True, inplace=True)

    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    
    return X_train_bal, y_train_bal, X_test, y_test
    
    
def train_downsample(X_train, y_train):
    
    
    # Creating a sa
    X_good_down, y_good_down = resample(X_train[y_train == 1], 
                                            y_train[y_train == 1], 
                                            replace=False,
                                            n_samples=X_train[y_train == 0].shape[0], 
                                            random_state=11)


    # Creating balances training sets by adding the equal sized dataframes together. 
    X_train_bal = pd.concat([X_train[y_train == 0],X_good_down])
    y_train_bal = pd.concat([y_train[y_train == 0],y_good_down])
    
    print('Number of negative \'good_review\' values in training set:', X_train_bal[y_train_bal == 0].shape[0])
    print('Number of positive \'good_review\' values in training set:',  X_train_bal[y_train_bal == 1].shape[0])
    

    
    return X_train_bal, y_train_bal


    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    

