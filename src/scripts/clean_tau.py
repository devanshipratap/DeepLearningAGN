import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_csv(
    '/Users/SnehPandya/Desktop/DeepLearningAGN/data/TAU_matched_new.csv', index_col=0)
# print(df.shape)
df = df.drop(columns=['ID.1', 'z.1', 'ERR'])
df = df.drop(df[df.tau == 5].index)  # unphysical tau
df = df.drop(df[df.tau == 7.5].index)  # unphysical tau
df = df.drop(df[df.tau < 0].index)
df = df.drop(df[df.M_tau < 0].index)
df = df.drop(df[df.M_tau > 12].index)
## dropped 470 data points ##


df['m_err_low'] = df.apply(lambda row: (row.tau_lim_lo + 2.29) / 0.56, axis=1)
df['m_err_hi'] = df.apply(lambda row: (row.tau_lim_hi + 2.29) / 0.56, axis=1)


def binning(df, bin_spacing):
    dict_tau = {}
    dict_pred = {}
    dict_pred_err = {}
    current_value = 4
    while current_value < 13:
        dict_tau[current_value] = df.apply(lambda row: row.M_tau if (
            current_value < row.Mass_ground_truth <= current_value + bin_spacing) else np.nan, axis=1).median()
        dict_pred[current_value] = df.apply(lambda row: row.Mass_prediction if (
            current_value < row.Mass_ground_truth <= current_value + bin_spacing) else np.nan, axis=1).median()
        dict_pred_err[current_value] = df.apply(lambda row: row.Mass_prediction if (
            current_value < row.Mass_ground_truth <= current_value + bin_spacing) else np.nan, axis=1).std()
        current_value = round(current_value + bin_spacing, 1)

    return dict_tau, dict_pred, dict_pred_err


def plotting(df):
    bin_spacing = .3
    dict_tau, dict_pred, dict_pred_err = binning(df, .3)
    for key in dict_tau.keys():
        list = [key, key + bin_spacing]
        plt.scatter(key, dict_tau[key])
        plt.scatter(key, dict_pred[key])

    plt.show()
plotting(df)


# df.to_csv('/Users/SnehPandya/Desktop/DeepLearningAGN/data/running_median_fixed_std.csv')
