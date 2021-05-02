import pandas as pd
import numpy as np
import datetime
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, Normalizer, FunctionTransformer

warnings.filterwarnings('ignore')

# Dealing with Nans

def find_nulls(data):
    """
    This function will find all the nulls.
    :param data: pandas DataFrame
    :return: column_name: null_percentage
    """
    col_nulls = data.isna().sum()*100/len(data)
    
    only_nulls = {}
    for col in data.columns:
        if col_nulls[col] > 0:
            only_nulls[col] = col_nulls[col]
            
    return only_nulls
    
def kill_nulls(data, only_nulls, row_threshold=5, col_threshold=50):
    """
    This function removes rows with nulls if below the threshold and columns with nulls if above the threshold.
    :param data: pandas DataFrame
    :param only_nulls: dictionary like {col_name: null_percentage}
    :return: dataframe with removed columns or rows
    """
    new_data = data.copy()
    to_kill = []
    for col, nulls in only_nulls.items():
        if nulls >= col_threshold:
            to_kill.append(col)
    
    if not any(nulls for nulls in only_nulls.values() if nulls > row_threshold):
        new_data = new_data.dropna()
        
    new_data = new_data.drop(columns=to_kill)
    
    lost_data = round((len(data)-len(new_data))/len(data)*100)
    if lost_data > row_threshold:
        print(f'Watch out! You lost {lost_data}% of your data.')
    
    return new_data

def interpolation(data, model, y_name):
    """
    Fill nulls with interpolated values.
    :param data: pandas DataFrame
    :param only_nulls: dictionary like {col_name: null_percentage}
    :param model: regression model to be used
    :param y_name: name of your target variable
    :return: dataframe with nulls filled in with method that achieved the highest r-squared
    """
    methods = ['linear', 'time', 'index', 'values', 'pad', 'nearest', 'zero', 'slinear', 'quadratic',
               'cubic', 'polynomial'] # 'spline' and 'barycentric' takes too long
    max_score = 0
    
    data_num = data.select_dtypes(np.number)
    data_cat = data.select_dtypes(exclude=np.number)
    
    print('Calculating interpolations. This might take a while.')
    for method in methods:
        if method in ['polynomial', 'spline']:
            for order in range(3, 10, 2):
                try:
                    data_n = data_num.interpolate(method=method, order=order, random_state=42)
                    data_ = pd.DataFrame(data_n).reset_index(drop=True).merge(data_cat, left_index=True, right_index=True).dropna()
                    score = regression_benchmark(data_, model, y_name)
                except:
                    print(f'An error ocurred while interpolating with {method.upper()} method, order {order}.')
                    
                if score > max_score:
                    final_data = data_
                    final_method = (method, order)
                    max_score = score
        else:
            try:
                data_n = data_num.interpolate(method=method, random_state=42)
                data_ = pd.DataFrame(data_n).reset_index(drop=True).merge(data_cat, left_index=True, right_index=True).dropna()
                score = regression_benchmark(data_, model, y_name)
            except:
                    print(f'An error ocurred while interpolating with {method.upper()} method.')
                    
            if score > max_score:
                final_data = data_
                final_method = (method)
                max_score = score
    
    if max_score == 0:
        print('Skipping interpolation... score bellow zero.')
        return data, 0, None
                  
    print(f'-- Lost {len(data)-len(final_data)} rows dropping NaNs from categorical columns.\n')
    return final_data, max_score, final_method

def median_or_mean(data, only_nulls, model, y_name):
    """
    Fill nulls with mean if there are no considerable outliers or with median if there are.
    :param data: pandas DataFrame
    :param only_nulls: dictionary like {col_name: null_percentage}
    :param model: regression model to be used
    :param y_name: name of your target variable
    :return: dataframe with nulls filled in and r-squared for regression model
    """
    
    for column in only_nulls.keys():
        if data[column].dtype in [float, int]:
            if has_outliers(data[column]):
                print(column, '- Replacing nulls with MEDIAN.')
                data[column] = data[column].fillna(np.nanmedian(data[column]))
            else:
                print(column, '- Replacing nulls with MEAN.')
                data[column] = data[column].fillna(np.mean(data[column]))
        else:
            print('This function is not ready to take other then numerical columns. \
                   Are you up for the challenge of adjusting it?')
    
    score = regression_benchmark(data, model, y_name)
    print()
    return data, score

def has_outliers(col):
    """ Checks for outliers. """
    if max(col) - col.quantile(.75) > col.quantile(.5):
        return True
    
def regression_benchmark(data, model, y_name):
    """
    Gets R-squared for regression models.
    You can improve this function by adding other metrics, like MAE and RMSE.
    :param data: pandas DataFrame
    :param model: regression model to be used
    :param y_name: name of your target variable
    :return: r-squared of trained regression model.
    """
    
    X = pd.get_dummies(data.drop(y_name, axis=1))
    y = data[y_name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    model.fit(X_train, y_train)
    
    return model.score(X_test, y_test)

def deal_with_nulls(data, model, y_name, allow_drop=True):
    """
    This function will choose the "best" way to deal with nulls, according to R-squared.
    It is a good idea to change this function according to your notion of "best".
    :param data: pandas DataFrame
    :param model: regression model to be used
    :param y_name: name of your target variable
    :param allow_drop: takes boolean. When True, will drop nulls if the r-squared is better.
    :return: data with nulls filled in, using the method that achieved highest r-squared.
    """
    nulls = find_nulls(data)
    data = kill_nulls(data, nulls)
    
    score_drop_nulls = regression_benchmark(data.dropna(), model, y_name)
    data_median_mean, score_median_mean = median_or_mean(data, nulls, model, y_name)
    data_interpolate, score_interpolate, method = interpolation(data, model, y_name)
    
    if score_drop_nulls > score_median_mean and score_drop_nulls > score_interpolate and allow_drop is True:
        print('Dropping nulls worked with higher R-squared:', score_drop_nulls)
        data_dropped = data.dropna()
        print('Data lost:', (len(data)-len(data_dropped))/len(data), '\n')
        return data_dropped
    elif score_median_mean > score_interpolate:
        print('Median/Mean replacement worked with higher R-squared:', score_median_mean, '\n')
        
        return data_median_mean
    else:
        print(f'Interpolation replacement worked with higher R-squared: {score_interpolate}.\nMethod used: "{method}". \n')
        return data_interpolate

# dealing with outliers

def deal_with_outliers(data, y_name, threshold=2.5):
    print('Dealing with outliers on target variable...')
    Q1 = data[y_name].quantile(0.25)
    Q3 = data[y_name].quantile(0.75)
    IQR = Q3 - Q1
    final_data = data[(data[y_name] > Q1 - IQR * 2.5) & (data[y_name] < Q3 + IQR * 2.5)]
    print(f'With a {threshold} * IQR threshold, you lost {round((len(data)-len(final_data))/len(data)*100)}% of data.\n')
    return final_data

# scaling
# https://scikit-learn.org/stable/modules/preprocessing.html

def deal_with_scaling(data, model, y_name):
    """
    Fits a scaler and transform data.
    :param data: pandas DataFrame
    :param model: regression model to be used
    :param scaler: scaler for numerical data to be used
    :param y_name: name of your target variable
    :return: transformed data.
    """
    data = data.copy()
    
    if sum(data.isna().sum()) > 0:
        print('Unable to check best scaler for data. You have NaNs in there!')
        return None, None, None, None
    
    scalers = {
                'row-wise': [PowerTransformer(method='yeo-johnson'), PowerTransformer(method='box-cox'),
                             StandardScaler(), MinMaxScaler(), RobustScaler(),
                             FunctionTransformer(np.log1p, validate=True)],
                'col-wise': [QuantileTransformer(output_distribution='normal'), Normalizer()]
               }
    
    max_score = regression_benchmark(data, model, y_name)
    final_scaler = None
    final_model = model
    X = pd.get_dummies(data.drop(y_name, axis=1))
    y = data[y_name]
    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X, y, test_size=0.25, random_state=42)

    print('Testing different scalers. This might take a while.')
    
    for scaler in scalers['row-wise']: 
        
        X = pd.get_dummies(data.drop(y_name, axis=1))
        y = data[y_name]

        X_train, X_test, y_train_, y_test_ = train_test_split(X, y, test_size=0.25, random_state=42)
        
        try:
            scaler.fit(X_train)
            X_train_, X_test_ = (scaler.transform(X_train), scaler.transform(X_test))
            
            model.fit(X_train_, y_train_)
            score = model.score(X_test_, y_test_)
            
        except:
            print(f'An error ocurred while scaling with {scaler}.')
            continue
        
        if score > max_score:
            max_score = score
            final_scaler = scaler
            final_model = model
            X_train_final, X_test_final, y_train_final, y_test_final = X_train_, X_test_, y_train_, y_test_ 
        
    print('Almost there...')
    for scaler in scalers['col-wise']:
        
        X = data.drop(y_name, axis=1)
        y = data[y_name]
        
        try:
            X_num = scaler.fit_transform(X.select_dtypes(np.number))
            
            X_cat = pd.get_dummies(X.select_dtypes(exclude=np.number))

            X_ = np.concatenate((X_num, X_cat), axis=1)

            X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y, test_size=0.25, random_state=42)

            model.fit(X_train_, y_train_)
            score = model.score(X_test_, y_test_)
            
        except:
            print(f'An error ocurred while scaling with {scaler}.')
            continue
        
        if score > max_score:
            max_score = score
            final_scaler = scaler
            final_model = model
            X_train_final, X_test_final, y_train_final, y_test_final = X_train_, X_test_, y_train_, y_test_ 

    with open('final_scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)
        
    #if max_score == regression_benchmark(data, model, y_name):
     #   final_model =
      #  max_score = 
    print(f'The scaler chosen was {scaler}, with an r-squared of {max_score}.\nSaving scaler to "final_scaler.pkl".\n')
    
    return X_train_final, X_test_final, y_train_final, y_test_final, final_model, max_score, final_scaler


# model selection

def model_selection(data, models, y_name, allow_drop=True, outliers=1.5):
    """
    Checks R-squared for different regression models, dropping outliers in the target variable and testing different methods of nulls handling and scaling.
    :param data: pandas DataFrame.
    :param models: regression models to be used tested.
    :param y_name: name of your target variable.
    :param allow_drop: boolean, standard is True. If False, will not allow dropping nulls.
    :param outliers: threshold to multiply IQR.
    :return: Prints out the results and saves best model.
    """
    print("Sit back and let me do the boring and repetitive work for you...\n")
    time.sleep(5)
    
    start = time.time()
    
    max_score = 0
    max_benchmark = 0
    
    for model in models:
        print(f'Checking {model} model: \n')
        data_no_nulls = deal_with_nulls(data, model, y_name, allow_drop=allow_drop)
        benchmark = regression_benchmark(data_no_nulls, model, y_name)
        data_no_outliers = deal_with_outliers(data_no_nulls, y_name, threshold=outliers)
        print('Data length after removing outliers and dealing with nulls:', len(data_no_outliers), '\n')
        X_train_, X_test_, y_train_, y_test_, model, score, scaler = deal_with_scaling(data_no_outliers, model, y_name)
        
        if score > max_score:
            max_score = score
            final_model = model
            final_scaler = scaler
            X_test, y_test = X_test_, y_test_
            
        if benchmark > max_benchmark:
            max_benchmark = benchmark
            benchmark_model = model

            
        print('--------------\n')
        
    y_pred = final_model.predict(X_test)

    with open('final_model.pkl', 'wb') as file:
        pickle.dump(final_model, file)
        print('FINAL RESULTS \n--------------\n')
        print(f'The best benchmark regression model is {benchmark_model}, with R-squared of {max_benchmark}.')
        print(f'The regression model with best R-squared is {final_model}, with {final_scaler} scaler.\n')
        print('METRICS:')
        print('R-square =', max_score)
        print('MAE =', mean_absolute_error(y_test, y_pred))
        print('RMSE =', mean_squared_error(y_test, y_pred, squared=False))
        print('\nSaving model to "final_model.pkl".\n')
        
    end = time.time()
    
    print(f'It took {round(end-start)} seconds to get this done.')

    
def automate_choice(data, models, y_name, allow_drop=True):
    """This is not doing anything yet."""
    pass