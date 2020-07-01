import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_csv(
    '/Users/SnehPandya/Desktop/DeepLearningAGN/data/2020-06-24Mass_smoothL1_b1.csv', index_col=0)

# binning function
def binning(df, bin_spacing):
    dict_pred = {} #create empty dicts
    dict_pred_err = {}
    current_value = 7 #lower end mass value
    df = df.apply(pd.to_numeric, errors='ignore')
    while current_value < 11:
        dict_pred[current_value] = df.apply(lambda row: row.Mass_prediction if (
            current_value < row.Mass_ground_truth <= current_value + bin_spacing) else np.nan, axis=1).median()
        dict_pred_err[current_value] = df.apply(lambda row: row.Mass_prediction if (
            current_value < row.Mass_ground_truth <= current_value + bin_spacing) else np.nan, axis=1).std()
        current_value = round(current_value + bin_spacing, 1)

    dict_pred_err_keys = list(dict_pred_err.keys())
    for idx, key in enumerate(dict_pred_err_keys):
        if idx + 1 != len(dict_pred_err_keys):
            next_key = dict_pred_err_keys[idx + 1]
        # df['Mass_error'] = df.apply(lambda row: dict_pred_err[key] if  (key < row['Mass_prediction'] <= next_key) else np.nan,axis=1)
        for i in range(len(df)):
            if (key < df['Mass_prediction'][i]) and (df['Mass_prediction'][i] < next_key):
                df['Mass_error'][i] = dict_pred_err[key]

        # else:
        #     df['Mass_error'] = 0
    df.to_csv(
        '/Users/SnehPandya/Desktop/DeepLearningAGN/data/2020-06-24Mass_smoothL1_b1_err.csv')


binning(df, .3)
