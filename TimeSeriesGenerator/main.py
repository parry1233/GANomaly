from TimeSeriesGenerator.utils import generate_steady_series, generate_anomaly_series, generate_all_features
import numpy as np
import matplotlib.pyplot as plt
import pandas

from scipy.stats import trim_mean
from scipy.stats.mstats import trimmed_std
from scipy.stats import kurtosis, skew
from statsmodels.tsa.stattools import pacf
import cv2

def plt_signal(series):
    n = 0
    fig, ax = plt.subplots(1,2, dpi = 150, figsize = (10, 4))
    ax[0].plot(series[n].signal)
    ax[0].set_ylim(0, np.max(series[n].signal)*1.05)
    ax[0].set_xlabel('time')
    ax[0].set_ylabel('signal')

    ax[1].plot(series[n].signal[1:]-series[n].signal[:-1])
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('signal`s first derivative')
    plt.show()
    
def convert_to_dataframe(series):
    series_df = pandas.DataFrame(series)
    return series_df
    
def showDF(series_df):
    series_df.head()
    lengths_series = series_df['signal'].apply(lambda x: len(x))
    plt.figure(dpi = 100)
    lengths_series.hist(bins = 20)
    plt.xlabel('length of series')
    plt.ylabel('counts')
    plt.show()
    
def preprocess(series, p_cut = 0.025, nan_replace = 0.0): 
    
    # replace NaN values with zeros
    series = np.nan_to_num(series, nan = nan_replace)
    
    #center and normalize by the mean
    series_mean = trim_mean(series, p_cut)
    if series_mean == 0:
        return np.array([np.nan]*len(series))
    series = series - series_mean
    series = series/series_mean  
    
    return series

# below are a few function you may find useful

def trim_series(x, p_cut =0.025):
    """ 
    Discards p_cut of the smallest and the largets values from the series
    Returns:
    -------
        redcued series
    """
    N = len(x)
    N_start = int(p_cut*N)
    N_end = int((1-p_cut)*N)
    sorted_x = sorted(x)
    return sorted_x[N_start:N_end]

def autocorr(x, t=1):
    """calculates autocorrelation coefficient with lag t """
    return np.corrcoef(np.array([x[:-t], x[t:]]))[0,1]

def trimmed_kurtosis(x):
    """ calculate kurtosis for series without extreme values"""
    trimmed_x = trim_series(x)
    return kurtosis(trimmed_x)

def trimmed_skew(x):
    """ calculate skew for series without extreme values"""
    trimmed_x = trim_series(x)
    return skew(trimmed_x)

def feature(x):
    
    #YOUR CODE
    
    res = autocorr(x, t=1)
    return res

def generateFeature(series_df,series_df_anomaly):
    features_steady = generate_all_features(series_df['signal'])
    features_steady['target'] = 1
    # remove all bad values
    features_steady = features_steady[~features_steady.isin([np.nan,
                                                                np.inf,
                                                                -np.inf]).any(1)]


    features_anomaly = generate_all_features(series_df_anomaly['signal'])
    features_anomaly['target'] = -1
    # remove all bad values
    features_anomaly = features_anomaly[~features_anomaly.isin([np.nan,
                                                                np.inf,
                                                                -np.inf]).any(1)]


    # combine the two data frames into one
    #features = pandas.concat([features_anomaly, features_steady])
    #features.shape
    #print(features)
    #return features
    
    return (features_steady, features_anomaly)

def split(feature, train_rate=0.8):
    # reshaffle the data
    features_steady = features_steady.sample(frac = 1, random_state = 25)

    #define the split point
    n_split = int(features_steady.shape[0]*train_rate)
    train = features_steady[:n_split]
    test = features_steady[n_split:]
    return (train, test)


def prepareDataset(steady, anomaly):  # combine the data and convert pandas.DataFrame object to a numpy.array
    train = pandas.concat([steady]).reset_index(drop=True)
    test = pandas.concat([steady, anomaly ]).sample(frac=1).reset_index(drop=True)
    #print(train)
    X_train = train.drop(columns =['target']).values
    y_train = train['target'].values
    X_test = test.drop(columns =['target']).values
    y_test = test['target'].values
    
    X_train_reshape = NPreshape(X_train,5,5)
    X_test_reshape = NPreshape(X_test,5,5)
    
    return (X_train_reshape, y_train), (X_test_reshape, y_test)

def NPreshape(x,height,width):
    new_x = np.empty((len(x), width, height))
    return np.reshape(x,(len(x),height,width,1))

def dataPreprocess_Main():
    np.random.seed(1)
    steady_series = generate_steady_series(N_samples=500, max_size = 1000)
    anomaly_series = generate_anomaly_series(N_samples=500, max_size = 1000)
    
    steady_df=convert_to_dataframe(steady_series)
    anomaly_df=convert_to_dataframe(anomaly_series)
    
    
    steady_df['scaled_series'] = steady_df['signal'].apply(preprocess)
    anomaly_df['scaled_series'] = anomaly_df['signal'].apply(lambda x: preprocess(x))
    
    
    (f_steady, f_anomaly) = generateFeature(steady_df, anomaly_df)
    
    
    #(train_x_ok, test_x_ok) = split(steady_df,1.0) #use all normal to train (100% train, 0% test)
    
    #(x_ok, y_ok), (x_test, y_test) = prepareDataset(f_steady, f_anomaly)
    return prepareDataset(f_steady, f_anomaly)

if __name__=='__main__':
    np.random.seed(1)
    steady_series = generate_steady_series(N_samples=500, max_size = 1000)
    anomaly_series = generate_anomaly_series(N_samples=500, max_size = 1000)
    
    plt_signal(steady_series)
    plt_signal(anomaly_series)
    
    steady_df=convert_to_dataframe(steady_series)
    anomaly_df=convert_to_dataframe(anomaly_series)
    
    print(steady_df)
    print(anomaly_df)
    
    #showDF(steady_df)
    #showDF(anomaly_df)
    
    steady_df['scaled_series'] = steady_df['signal'].apply(preprocess)
    anomaly_df['scaled_series'] = anomaly_df['signal'].apply(lambda x: preprocess(x))
    
    
    # check different features using the function below
    # see if the values are different for steady and anomalous series
    '''
    plt.figure(dpi = 100)
    steady_df['scaled_series'].apply(feature).hist(bins = 10, alpha = 0.5, label = 'steady')
    anomaly_df['scaled_series'].apply(feature).hist(bins = 10, alpha = 0.5, label = 'anomaly')
    plt.xlabel('feature value')
    plt.ylabel('counts')
    plt.legend()
    plt.show()
    '''
    
    
    (f_steady, f_anomaly) = generateFeature(steady_df, anomaly_df)
    
    
    #(train_x_ok, test_x_ok) = split(steady_df,1.0) #use all normal to train (100% train, 0% test)
    
    (x_ok, y_ok), (x_test, y_test) = prepareDataset(f_steady, f_anomaly)
    print(x_ok)
    print(x_ok.shape)
    print(y_ok.shape)
    print(x_test)
    print(x_test.shape)
    print(y_test.shape)
    x_ok_reshape = NPreshape(x_ok,5,5)
    x_test_reshape = NPreshape(x_test,5,5)
    print(x_ok_reshape)
    print(x_ok_reshape.shape)
    print(x_test_reshape)
    print(x_test_reshape.shape)