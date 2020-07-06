import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df_MSE = pd.read_csv(
    '/Users/SnehPandya/Desktop/DeepLearningAGN/data/2020-05-26Mass_nll_b1_matched.csv', index_col=0)
df_smoothL1 = pd.read_csv('/Users/SnehPandya/Desktop/DeepLearningAGN/data/2020-06-24Mass_smoothL1_b1_err_matched.csv',index_col=0)


def binning(df, bin_spacing):
    dict_tau = {}
    dict_pred = {}
    dict_pred_err = {}
    dict_tau_err = {}
    current_value = 7
    while current_value < 11:
        dict_tau[current_value] = df.apply(lambda row: row.M_tau if (
            current_value < row.Mass_ground_truth <= current_value + bin_spacing) else np.nan, axis=1).median()
        dict_pred[current_value] = df.apply(lambda row: row.Mass_prediction if (
            current_value < row.Mass_ground_truth <= current_value + bin_spacing) else np.nan, axis=1).median()
        dict_pred_err[current_value] = df.apply(lambda row: row.Mass_prediction if (
            current_value < row.Mass_ground_truth <= current_value + bin_spacing) else np.nan, axis=1).std()
        dict_tau_err[current_value] = df.apply(lambda row: row.M_tau if (
            current_value < row.Mass_ground_truth <= current_value + bin_spacing) else np.nan, axis=1).std()
        current_value = round(current_value + bin_spacing, 1)

    return dict_tau, dict_pred, dict_pred_err, dict_tau_err


def plotting(df):
    plt.figure(figsize=(6, 6))
    sns.set(font='Times New Roman')
    plt.title('Comparison with Traditional Method ResNet')
    plt.xlabel('LOG(M/M_sun)')
    plt.ylabel('LOG(M/M_sun)')
    plt.ylim(7, 11)
    plt.xlim(7, 11)

    bin_spacing = .4
    dict_tau, dict_pred, dict_pred_err, dict_tau_err = binning(df, .4)
    for key in dict_tau.keys():
        list = np.arange(key,key+bin_spacing,.1)
        if len(list) != 5:
            list = np.append(list,8.2)
        tau_rm = plt.plot(list,
                          [dict_tau[key]] * 5, color='darkred')

        nn_rm = plt.plot(list,
                         [dict_pred[key]] * 5, color='darkblue')
        tau_rm_error = plt.errorbar((key + (key + bin_spacing)) / 2,
                                    dict_tau[key], dict_tau_err[key], ls='', color='darkred', alpha=.5, capsize=4)
        nn_rm_error = plt.errorbar((key + (key + bin_spacing)) / 2,
                                   dict_pred[key], dict_pred_err[key], ls='', color='darkblue', alpha=.5, capsize=4)
    ground_truth = plt.plot(df['Mass_ground_truth'],
                            df['Mass_ground_truth'], color='black')
    tau = plt.scatter(df['Mass_ground_truth'], df['Mass_prediction'],
                      s=1, color='yellow', label='NN prediction', alpha=.4)
    nn = plt.scatter(df['Mass_ground_truth'], df['M_tau'],
                     color='darkorange', s=1, alpha=.4, label='best fit damping')
    # plt.legend((tau_rm, nn_rm,), labels=('best-fit damping running median',
                                                 # 'neural network prediction running median'))

    plt.show()


plotting(df_MSE)
plotting(df_smoothL1)


# df.to_csv('/Users/SnehPandya/Desktop/DeepLearningAGN/data/running_median_fixed_std.csv')
