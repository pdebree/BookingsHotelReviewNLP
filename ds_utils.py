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

from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

# library for stemming
import nltk
from nltk.corpus import stopwords 

def basic_eda(df, df_name):
    """
    getting some basic information about each dataframe
    shape of dataframe i.e. number of rows and columns
    total number of rows with null values
    total number of duplicates
    data types of columns
    Args:
        df (dataframe): dataframe containing the data for analysis
        df_name (string): name of the dataframe
    """
    print(df_name.upper())
    print()
    print(f"Rows: {df.shape[0]} \t Columns: {df.shape[1]}")
    print()
    print(f"Total null rows: {df.isnull().sum().sum()}")
    print(f"Percentage null rows: {round(df.isnull().sum().sum() / df.shape[0] * 100, 2)}%")
    print()
    print(f"Total duplicate rows: {df[df.duplicated(keep=False)].shape[0]}")
    print(f"Percentage dupe rows: {round(df[df.duplicated(keep=False)].shape[0] / df.shape[0] * 100, 2)}%")
    print()
    print(df.dtypes)
    print("-----\n")
    
def plot_dist_by_dim(data, column, dim):
    """
    Plots the given column against the registration station in the data.
    The function assumes data is a dataframe, column is string (existing column in data),
    and data has a registered column too.
    """
    total_count = data.groupby([column, dim])[column].count()
    pct_contact_type = total_count/data.groupby(column)[column].count()
    pct_contact_type = pct_contact_type.unstack()
    print(pct_contact_type.sort_values([1]))
    plt.rcParams['figure.dpi'] = 360
    plt.rcParams['figure.figsize'] = (3.2, 2)
    # set the font name for a font family
    # plt.rcParams.update({'font.sans-serif':'Helvetica'})
    sns.set(style="whitegrid")
    pct_contact_type.sort_values([1]).plot(kind="bar", stacked=True, color=['#003F5C', '#FFA600', '#BC5090'])
    sns.despine(left=True)
    plt.title(f"{column} group distribution", size=10, color='#4F4E4E', fontweight="bold")
    plt.xlabel('')
    plt.xticks(size=8, color='#4F4E4E', rotation=90)
    plt.yticks(size=8, color='#4F4E4E')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()
    


def prepro_plot(data, column, dep_column, norm=False):
    """
    Takes in data and two columns, in the dataframe, then looks at the types of the columns in order to return 
    an appropriate distribution representation. Data is a dataframe including the two columns, column is a potential
    independent variable, dep_column is a potential dependent variable. norm takes in whether value_counts should be 
    reported as sums or as proportions of the total.
    
    Parameters: data, column, dep_column, norm
    
    Returns:
        (Categorical, Categorical) 
        (Categorical, Numeric)
        (Numeric, Numeric) 
        (Numeric, Cateforical)
    """
    
    # Checks the datatype of the independent column (X) is an object
    if pd.api.types.is_object_dtype(data[column]):
       
        # Returns value counts for independent variable
        print("Value Counts For", column)
        print(data[column].value_counts(normalize=norm))
        
        # Checks whether the dependent variable is categorical
        if pd.api.types.is_object_dtype(data[dep_column]):
            
            # Creates percentage barcharts to see how the independent variable changes amongst the categories 
            total_count = data.groupby([column, dep_column])[column].count()
            pct_dep = total_count/data.groupby(column)[column].count()
            pct_dep = pct_dep.unstack().sort_values(by=column)
            
            
            pct_dep.plot(kind="bar", stacked=True)
            sns.despine(left=True)
            plt.title(f"{column} distribution", size=10)
            plt.xlabel('')
            plt.xticks(size=8, rotation=90)
            plt.yticks(size=8)
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.show()
        
        # If dependent variable is 
        else:
            # Creates boxplot of how dependent variable is distributed in different categories
            sns.boxplot(x=column, y=dep_column, data=data)
            plt.show()
        

            
    # Checks that the datatype of dependent is numeric 
    elif pd.api.types.is_numeric_dtype(data[column]):
        
        # Returns statistical summary of numeric dependent variable
        print("Description of", column)
        print(data[column].describe())
        
        # If independent variable is categorical, creates boxplots 
        if pd.api.types.is_object_dtype(data[dep_column]):
            sns.boxplot(x=column, y=dep_column, data=data)
            plt.show()
        
        # If both variables are numeric, creates scatterplot
        elif pd.api.types.is_numeric_dtype(data[dep_column]):
            sns.scatterplot(x=column, y=dep_column, data=data)
            plt.show()
        else:
            print("Inappropriate Independent Variable Datatype")
            
    else:
        print("Inappropriate Dependent Variable Datatype")
            
    

def read_csv_pd(filepath, separator=",", index=False):
    """
    Converts a csv file, from a filepath, to a pandas DataFrame
    
    Creates, and returns, the DataFrame and prints the shape of the created DataFrame and 
    whether duplicated or missing values were found (usedin no_duplicates_missing). This functionality is 
    useful when reading in known, cleaned files to ensure they take the expected form.
    
    Parameters
    ----------
    filepath: string
                a filepath to the location of the .csv file to be read in. 
    
    separator: string 
                a string of the character used to separate data points
    
    index: boolean
                indicates whether the first column of the csv file should be read as an index
    
    Returns
    -------
    df: Pandas DataFrame
                a Pandas DataFrame of the csv file 
                
                
    
    Examples
    --------
    >>> read_csv_pd('data/clouds.csv')
    df
    
    See Also
    --------
    no_duplicates_missing: reports on whether missing values or duplicates were found in the DataFrame 
    """

    #assert 

    # check whether the index should be read in as the first column
    if (index==True):
        # read in csv file with index as the first column
        df = pd.read_csv(filepath, sep=separator, index_col=[0])
    else:
        # read in csv file with no index
        df = pd.read_csv(filepath, sep=separator, index_col=None)

    # print the shape of the created DataFrame
    print(f'The DataFrame contains {df.shape[0]} rows and {df.shape[1]} columns.')

    # check if missing values or duplicates were found
    no_duplicates_missing(df, report=True)

    return df 

def no_duplicates_missing(df, report=False):
    """
    Takes in a recently read-in pandas dataframe and returns True if there are no duplicates and 
    no missing values. 
    
    Parameters
    ----------
    df : pandas DataFrame
                The DataFrame to be checked for missing or duplicated values. 
    report: boolean
                Whether or not the number of missing values should be reported.
    
    Returns
    -------
    found_dups: boolean
                A boolean value of whether no missing values were found and no duplicates were found
    
    Examples
    --------
    >>> no_duplicates_missing(df, report=True)
    5 missing values and 2 duplicate rows found.
    
    """

    # check whether duplicates and missing values 
    if (report==True):
        print("There are", df.isna().sum().sum(), "missing values and",
              df.duplicated().sum().sum(), "duplicated rows.")

    # returns boolean of whether either missing or duplicated values were found
    found = (df.isna().sum().sum() == 0) & (df.isnull().sum().sum() == 0)  

    return found   




def remove_duplicates_and_missing(df):
    """
    Takes a dataframe and removes all duplicated rows, and then all missing values.
    As there may be duplicates that contain missing values the number of missing 
    values is re-reported before removal (to check for a correct removal and as 
    a sanity check). 
    
    Parameters
    ----------
    df : pandas DataFrame
                A DataFrame containing the data to be altered
                
    Returns
    -------
    df_clean : pandas DataFrame
                A DataFrame containing the altered data
                 
    """
    
    # report the number of duplicates and missing values
    print("There are", df.isna().sum().sum(), "missing values and",
          df.duplicated().sum().sum(), "duplicated rows.\n")
    
    # remove all duplicates
    df_no_dupes = df[df.duplicated() == False]
    
    # report number of diuplicates removed
    print(len(df) - len(df_no_dupes), 'duplicates removed.\n')
    
    # report number of missing values after duplicates removed
    print(df_no_dupes.isna().sum().sum(), 'missing values remaining')
    
    # drop missing values
    df_clean = df_no_dupes.dropna()
    
    print("\nResulting DataFrame has", df_clean.isna().sum().sum(), "missing values and",
          df_clean.duplicated().sum().sum(), "duplicated rows. There are", len(df_clean), 'rows remaining.')
    
    return df_clean

    

def remove_attribute_whitespace(df):
    """
    Takes a dataframe and changes the names of the columns in order to remove all whitespace. For each
    column replaces whitespace with an underscore.
    
    Parameters
    ----------
    df : Pandas DataFrame
                The dataFrame to which the changes will be made
    
    Returns
    -------
    df : Pandas DataFrme
                The updated dataframe
    """
    
    # Changing our column names to have only "_" values, and no empty spaces
    df.columns = df.columns.str.replace(" ", "_", regex=False)
    
    return df

def cat_var(df):
    """
    Takes a dataframe of categorical variables and returns distribution information.
    
    Reports the the value_counts for each category, if there are less than 20 unique values, and
    a barplot for the values of each category.
    
    Paratmeters
    -----------
    df : Pandas DataFrame
            dataframe of categorical variables.
    
    Returns
    -------
    No return value but printed value counts and visualisations
    
    Examples
    --------
    >>> cat_var(df)
    ** plot of all variables in df **
    
    Notes
    -----
    All columns in dataset must be categorical.
    Utilises seaborn's countplot to displot counts of variables in all categories.
    
    See Also
    -------
    num_var: similar function but for numeric variables. 
    """


    # Creates list of the dataframe columns
    variables = df.columns

    # Loops over the names of the dataframe columns
    for i in variables:
        # if there are less than 20 values print value count and show count plot
        if df[i].nunique() < 20:
            print(df[i].value_counts().to_frame()) 
            # Initialise barchart
            plt.figure()
            # plot a count plot of the unique categories and how often they appear
            sns.countplot(x=df[i], order = df[i].value_counts().index)
            plt.ylabel("Count") # use count as y label 
            plt.xlabel(i) # use column name as x label 
            plt.xticks(rotation=40) # format ticks
            plt.show() # show plot 
        # if more than 20 variables, report the number.
        else:
            print(f'Number of Unique {i} Values: {df[i].nunique()}')


    
def num_var(df, bins=20):
    """
    Takes a dataframe of numeric variables and returns distribution information.
    
    Reports the the summary statistics and a histogram of the distribution.
    
    Paratmeters
    -----------
    df : Pandas DataFrame
            DataFrame of numeric variables.
    bins : integer
            The number of bins to be included in the histogram (number of groups).
    
    Returns
    -------
    No return value printed summary statistics and visualisations
    
    Examples
    --------
    >>> num_var(df)
    ** plot of all variables in df **
    
    Notes
    -----
    All columns in dataset must be numeric (date values can be included but this must be established
        within the DataFrame passed in).
    Utilises seaborn's countplot to displot counts of variables in all categories.
    
    See Also
    -------
    num_var: similar function but for numeric variables. 
    """


    # Creates a list of column names
    variables = df.columns

    # loops over column names
    for i in variables:
        print(i,'Summary Statistics:')

        # shows description of variable distributions
        print(df[i].describe(datetime_is_numeric=True)) 

        # Initialise histogram 
        plt.figure()
        sns.histplot(x=df[i], bins=bins)

        if df[i].dtype != '<M8[ns]':
            # Include mean and median in reporting, for non-datetime types
            plt.axvline(np.median(df[i]), color='blue', label="Median")
            plt.axvline(np.mean(df[i]), color='red', label="Mean")

            # Include legend 
            plt.legend()
            
        # Rotation x labels for easier visability 
        plt.xticks(rotation=40)
        plt.show()



def num_cat_cols(df, datetime_is_numeric=False):
    """
    Create a lists of numeric and of categorical variables for the passed in dataframe. 
    
    Has the functionality of being able to chose if datetime is numeric or not (this can depend 
    on situations and number of unique appearances of datetime). Default is datetime is not numeric.
    Booleans are considered as categorical variables.
   
        
    Paratmeters
    -----------
    df: Pandas DataFrame
            A DataFrame with columns.
            
    date_time_is_numeric : boolean
            A boolean to 
    
    Returns
    -------
    num_col : list of strings
                List of names of numerical columns
    
    cat_col : list of strings
                List of names of categorical columns
    
    Also, prints the names of these columns
    Examples
    --------
    >>> numeric, categorical = num_cat_cols(student_info)
    The Numeric columns: 
        age,
        score
    
    The Categorical columns: 
        gender,
        class
        
        
    See Also
    -------
    pandas.select_dtypes : selects the columns in a dataframe with the passed in datatype
    """

    # checks whether datetime variables should be considered as numeric
    if datetime_is_numeric==True:
        # makes a list of the names of the numeric columns (including datetime)
        num_col = list(df.select_dtypes(["number","datetime64"]).columns)
        # makes a list of the names of categorical columns (all other columns)
        cat_col = list(list(set(df.columns.values) - set(num_col)))
    else:
        # makes a list of the names of the numeric columns (including datetime)
        num_col = list(df.select_dtypes(["number"]).columns)
        # makes a list of the names of categorical columns (all other columns)
        cat_col = list(list(set(df.columns.values) - set(num_col)))

    # prints the names of the categorical and numeric columns
    # .join iterates over the passed in list and prints it with the string it is passed on 
    print("The Numeric columns: \n\t", ",\n\t".join(num_col), sep="")
    print("")
    print("The Categorical columns: \n\t", ",\n\t".join(cat_col), sep="")

    # returns list of numeric columns and list of categorical columns 
    return num_col, cat_col


    
def equal_transform(df1, df2):
    """
    Takes in two series and returns whether they have equal value counts. 
    
    Useful when changing the value markers in columns, e.g. changing a Boolean series to a dummy series. 
        
    
    Paratmeters
    -----------
    df1 : Pandas Series (column of DataFrame)
            Series of values
    df2 : Pandas Series (column of DataFrame)
    
    
    Returns
    -------
    No return value but prints whether series are equal 
    
    Examples
    --------
    >>> equal_transform(df['color'], df['color_encoded'])
    Series are equal.
    
    
    See Also
    -------
    pandas.assert_series_equal : compares the value counts of series so differences in actual values are not an issue
    """


    # put the code here

    # Tries to check that the series are equal
    try: 
        pd.testing.assert_series_equal(df1.value_counts(),
                                 df2.value_counts(),
                                 check_names=False, check_index=False)
        print("Series are equal.")

    # If unable to assert that the series are equal, prints the following
    except:
        print("Series are not equal.")
    
    
    
def linear_reg(X, y, plots=False):
    """
    Performs a linear regression, using the X as the independent variables and y as the dependent variable to fit
    the regression. Returns the regression results for functionality of looking at attributes. 
    
    Optionally, creates residual distribution and qq plot.
    
    Parameters: X, y, plots 
    
    Returns: Regression Results
    """
    
    # Initialises the linear regression
    reg=sm.OLS(y,X)
    reg_res = reg.fit()
    # prints regression summary table
    print(reg_res.summary())
    
    # Optioonally, creates residual and qqplots
    if (plots == True):       
        gridspec.GridSpec(2,1)
        
        plt.figure(figsize=(20, 10))
        
        plt.subplot2grid((1,2), (0,0), colspan=1, rowspan=1)
        sns.scatterplot(reg_res.fittedvalues, reg_res.resid)
        

        plt.subplot2grid((1,2), (0,1), colspan=1, rowspan=1)
        stats.probplot(reg_res.resid, dist="norm", plot = plt)
        
        plt.show()    
    
    return reg_res
        
        
def logit_reg(X, y):
    """
    Runs a logistic regression, using X as the independent variables and y as the dependent to fit the model. 
    In fitting the model it creates probabilities for each data point for whether the outcome is 1 or 0. Also 
    prints the odds ratios for each variable. 
    
    Returns the regression results for functionality of looking at attributes.
    
    Paramters: X, y
    
    Return: Regression predictions
    
    
    """
    
    # Initialises and fits a logistic regression
    reg=sm.Logit(y, X)
    reg_res = reg.fit()
    print(reg_res.summary())
    
    cols = X.columns
    
    # Prints odds ratio for each variable (based on e^b0)
    print("\nOdds Ratios: ")
    for i in range(len(reg_res.params)):
        print(cols[i],round(np.exp(reg_res.params[i]),4)) 
    
    return reg_res


def logit_accuracy(y, reg_res, prob):
    """ 
    Calculates the accuracy of a logistic regression based on the input probablility threshold. Accuracy is tested by comparing 
    the actual values with predicted values based on the chosen threshold, prob. reg_res is the result of a logistic regression
    run on X, y. prob is the chosen probability threshold to test accuracy for. 
    
    Parameters: X, y, reg_res and prob 
    
    Returns: The accuracy of the test. 
    """
    
    # Predicts the values based on the fit linear model. 
    preds = np.where(reg_res > prob, 1, 0)
    
    return round(((preds == y.values).sum()/len(y))*100,2)
    

    
              
def higher_rsq(X, y, adds, p_vals=False, summary=False, plots=False):
    """
    Takes a current df of variables and sees which variable improves r^2 the most when performing a linear regression.
    X is the current independent variables (can be just const). y is the independent variable and adds is a df of all the
    potential variables to be added. Has the options to return p-values and summary statistics.
    
    Parameters: X, y, adds, 
    
    Returns: Prints which value increase the r-squared the most and the new r-squared.
    """
    
    # Initialises linear regression with old variables in order to retrieve current r-squared value  
    pre = sm.OLS(y, X)
    pre_res = pre.fit()
    
    # assigns old r-squared
    pre_rsqu = pre_res.rsquared
    
    # assings highest r-squared, will be changed if a regression creates a higher r-squared
    highest_rsqu = pre_rsqu
    
    # loops over the columns of the dataframe of potential new variables.
    for i in adds:
        # runs a regression with the new variable added to the X matrix
        reg = sm.OLS(y,pd.merge(left=X, right=adds[i], left_index=True, right_index=True))
        reg_res = reg.fit()
        
        # Assigns the r-squared, if higher than before and keeps track of which variable does this.
        if reg_res.rsquared > highest_rsqu:
            highest_rsqu = reg_res.rsquared
            improver = i 
            improver_results = reg_res
            improver_res = reg_res.summary() # keeps summary of best regression, for if summary=True
    
    # Checks if a higher r-squared has been found
    if highest_rsqu == pre_rsqu:
        print("No Variables Improved R-Squared", highest)
    else:
        print(improver, "improved R-Squared the most, the new R-Squared is", highest_rsqu, "\n")
     
    # prints summary if summary=True
    if summary == True:
        print(improver_res)
    # prints p-values if p_vals=True (done in elif because if summary=True they will be reported)
    elif p_vals == True:
        print(improver_res.p_values())
    
    # Plots the residuals and qqplots
    if plots == True:
        gridspec.GridSpec(1,2)
        
        plt.subplot2grid((1,2), (0,0), colspan=1, rowspan=1)
        sns.scatterplot(improver_res.fittedvalues, improver_res.resid)
        
        plt.subplot2grid((1,2), (0,1), colspan=1, rowspan=1)
        stats.probplot(improver_res.resid, dist="norm", plot = plt)
        plt.show()
        
    return improver_results
        
                                           
def equation_maker(reg_res):
    """
    Takes in a regression result and returns a formatted string of the equation
    
    Parameters: Result of a regression
    
    Returns: Equation of regression
    """
    
    eq = ""
    count = 1 
    
    for i in reg_res.params.index:
        if count==1:
            eq += "(" + str(round(reg_res.params[i], 4)) + ")"+ i 
            first=False
        elif (count %3 == 0):
            eq += "\n\t + (" +str(round(reg_res.params[i], 4)) + ")"+ i
            
        else:
            eq += " + (" +str(round(reg_res.params[i], 4)) + ")"+ i
        
        count += 1 
    return eq


                                           
def sk_logistic_reg(X, y, c=1, pen='L2', testsize=0.25, scalar=None): #, model_name=):
    
    # Split data into 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize)
    

    if scaler == 'standard':
        standardScaler = StandardScaler()
        standardScaler.fit(X_train)
        X_train = standardScaler.transform(X_train)
        X_test = standardScaler.transform(X_test)
        
    elif scaler == 'minmax':
        minmaxScaler = MinMaxScaler()  
        minmaxScaler.fit(X_train)
        X_train = minmaxScaler.transform(X_train)
        X_test = minmaxScaler.transform(X_test)
    
    logistic_regression_model = LogisticRegression(penalty=pen, C=c)
    logistic_regression_model.fit(X_train, y_train)
    
    if scaler != None:
        
        print('Using a',scaler, 'scaler,', end='')
    print(f'With a {test_size} test set')
    print(f'Train Accuracy Score {accuracy_score(y_train, logistic_regression_model.predict(X_train))}')
    print(f'Train Accuracy Score {accuracy_score(y_test, logistic_regression_model.predict(X_test))}')
          
    return logistic_regression_model
          
          
def sk_knn_class(k=5,scaler='standard', weight='uniform', testsize=0.25):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize)

    if scaler == 'minmax':
        minmaxScaler = MinMaxScaler()  
        minmaxScaler.fit(X_train)
        X_train = minmaxScaler.transform(X_train)
        X_test = minmaxScaler.transform(X_test)

    elif scaler == 'standard':
        standardScaler = StandardScaler()  
        standardScaler.fit(X_train)
        X_train = standardScaler.transform(X_train)
        X_test = standardScaler.transform(X_test)


    knn_model = KNeighborsClassifier(n_neighbors=k, weights=weight)
    knn_model.fit(X_train, y_train)
    print(f"No Neighbors:{k}, Weights:{weight}, Scaler:{scaler} and Test Size:{test_size}")
    print(f"Train set accuracy: {knn_model.score(X_train, y_train)}")
    print(f"Test set accuracy: {knn_model.score(X_test, y_test)}")   
    
    return knn_model
   


def ohe_sparse(sr):
    """
    Creates encodings for a numpy series and creates a sparse matrix of the encoded values.
    
    Uses sklearn's OneHotEncoder to turn a series of categorical values to a matrix of 
    the data rows and the categories as columns, with binary values (0 or 1) indicating 
    whether a row is associated with a particular category
        
    Paratmeters
    -----------
    sr: Pandas Series
            A Pandas series with categories.
    
    Returns
    -------
    df : Pandas DataFrame
            A Pandas DataFrame of the sparse encoding matrix category names as columns
    Examples
    --------
    >>> ohe_sparse(student_info['gender'])
    
    See Also
    -------
    sklearn.OneHotEncoder : 
    """

    # Initialise OneHotEncoder
    ohe = OneHotEncoder()

    # Fit the encoding and transform the data (needs to be transformed to a one dimensional numpy array
    encoded = ohe.fit_transform(np.array(sr).reshape(-1,1))                      

    # Creating a dense version of the matrix, to be added to a dataframe
    sparse_matrix = encoded.toarray()

    # Create dataframe version of dense matrix - uses original series index for index and values in series as columns
    df = pd.DataFrame(sparse_matrix, columns=ohe.categories_[0], dtype=int)

    # returns dataframe of encoded matri
    return df         
    
    
def ohe_dense(sr):
    """
    takes in a numpy series and creates a dense matrix of the encoded categorical values.
    """
    
    # Initialise OneHotEncoder
    ohe = OneHotEncoder()
    
    # Fit the encoding and transform the data
    encoded = ohe.fit_transform(np.array(sr).reshape(-1,1))                      
                                
    # Creating a dense version of the matrix, to be added to a dataframe
    dense_array = encoded.toarray()
    
    # Create dataframe version of dense matrix
    df = pd.DataFrame(dense_array, columns=ohe.categories_[0], dtype=int)
    
    return df                          
                 

def sk_logreg(X_train, X_test, y_train, y_test, c=1.0, pen='L2'):
    """
    Runs a logistic regression on the passed in data and returns the model, the training score and the testing score
    """
    lr = LogisticRegression(max_iter=2000, C=c)
    lr.fit(X_train, y_train)
    
    train_score = lr.score(X_train, y_train)
    test_score = lr.score(X_test, y_test)
    
    return lr, train_score, test_score


def bagofwords(train, test, stop_words_=None, min_df_=0, tokenizer_=None, max_features_=None):
    """
    Runs a CountVectorizer on the passed in training data, based on the parameters passed in, and returns the 
    model, the training data fit to the model and the testing data fit to the model
    """
    bow = CountVectorizer(stop_words=stop_words_, tokenizer=tokenizer_,min_df=min_df_)
    bow.fit(train)
    
    train_trans = bow.transform(train)
    test_trans = bow.transform(test)
    
    return bow, train_trans, test_trans


def my_tdidf(train, test, stop_words_=None, min_df_=0, tokenizer_=None, max_features_=None):
    """
    Runs a TDIDFVectorizer on the passed in training data, based on the parameters passed in, and returns the 
    model, the training data fit to the model and the testing data fit to the model
    """
    tdidf = TfidfVectorizer(stop_words=stop_words_, tokenizer=tokenizer_,min_df=min_df_,max_features=max_features_)
    tdidf.fit(train)
    
    train_trans = tdidf.transform(train)
    test_trans = tdidf.transform(test)
    
    return tdidf, train_trans, test_trans


def rev_tokenizer(words):
    """
    Custom tokeniser to omit stop words and perform stemming, from the nltk library, on a passed in string.
    This tokeniser will be applied to vectorisations. 
    """
    
    english_stopwords = stopwords.words('english')  
    stemy = nltk.stem.PorterStemmer()
    
    words_list = words.split(' ')
    stemmed_list = []
    
    for word in words_list:
        if (not word in english_stopwords) and (word!=''):

            stemmed_word = stemy.stem(word)
            stemmed_list.append(stemmed_word)

    return stemmed_list




def add_list_matrix(df, list_column, mean_threshold=0.05):
    """
    Creates a matrix of artist term encoding for each row of the data and adds to passed in DataFrame
    
    Takes in a pandas DataFrame and column to be binarised. Uses sklearn's MultiLabelBinarizer function 
    to create a appearance matrix of the elements in the list_column's lists for each row. This is then added 
    back to the original dataframe. However, it removes values that appear less than a decided amount of 
    times in the matrix (passed in as mean_threshold). This reduces the size of the returned data and ensure that no
    irrelevant variables are included. Default appearance is 5% of the data.
    

    
    Parameters
    ----------
    df : Pandas DataFrame
                DataFrame with the list_column in it.
                
    list_column : string
                Name of the column to be binarised (column must be made up of numpy arrays)
    
    mean_threshold : float
                Minimum percentage of times that a variable must appear to be included

    
    Returns
    -------
    df: Pandas DataFrame
                a Pandas DataFrame with the binarised variables merged with the original 

    Examples
    --------
    >>> add_list_matrix(df, 'shape' mean_threshold=0.1)
    df
    
    Note
    ----
    Original column is not removed in this stage
    
    See Also
    --------
    sklearn.MultiLabelBinarizer : 
    feat_encoding : performs overall feature encoding for dataframe (calls this function)
    """
        
    # Initialise multilabel binariser to take our column of lists to a dense matrix 
    mlb = MultiLabelBinarizer()
    
    # create dense matrix by fitting and transforming the selected column
    list_matrix = mlb.fit_transform(df[list_column])
    
    # create dataframe of matrix data using classes as column labels and original DataFrame index
    matrix_pd = pd.DataFrame(data=list_matrix, columns=mlb.classes_, 
                                  index=df.index)
    
    # calculate means for every column to get percentage appearance
    variable_means = matrix_pd.mean().sort_values(ascending=False)
    
    # select artist terms that appear more than the threshold
    relevant_variables = variable_means.loc[variable_means > mean_threshold].index.values

    df2 = pd.merge(df, matrix_pd[relevant_variables], 
                   left_index=True, right_index=True)
    
    return df2

    



    
    
    
    
