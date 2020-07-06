"""
script to run all relevant data analysis on NN Results
"""
# importing packages
import pandas as pd
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error

# defining DataAnalysis class


class DataAnalysis():

    # initializing dataframe attributes
    def __init__(self, PATH, X, Y, ERROR, df_X=True, df_Y=True, df_ERROR=True):
        self.PATH = PATH
        self.X = X
        self.Y = Y
        self.ERROR = ERROR
        self.df = pd.read_csv(self.PATH)
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]
        self.df = self.df.apply(pd.to_numeric, errors='ignore')
        self.df_X = self.df[self.X]
        self.df_Y = self.df[self.Y]
        self.df_ERROR = self.df[self.ERROR]

    # pearson summary statistics
    def pvalue(self):
        pearson = stats.pearsonr(self.df_X, self.df_Y)
        return pearson[1]

    def pearson(self):
        pearson = stats.pearsonr(self.df_X, self.df_Y)
        return pearson[0]

    # spearman summary statistics
    def spearman(self):
        spearman = stats.spearmanr(self.df_X, self.df_Y)
        return spearman[0]

    # Root Mean Squared Error
    def RMSE(self):
        RMSE = sqrt(mean_squared_error(self.df_X, self.df_Y))
        return RMSE

    def mean_error(self):
        mean_error = self.df_ERROR.mean()
        return mean_error

    # Outlier Fraction
    def OLF(self, n_sigma):
        count = 0
        for i in range(len(self.df)):
            lower_bound = self.df_Y[i] - n_sigma * self.df_ERROR[i]
            upper_bound = self.df_Y[i] + n_sigma * self.df_ERROR[i]
            point = self.df_X[i]

            if point >= lower_bound and point <= upper_bound:
                count += 1
        OLF = 1 - (count / len(self.df))
        return OLF

    # plotting results
    def plot(self):
        plt.figure(figsize=(8, 5))
        sns.set_context('paper')
        sns.set(font='times new roman')
        scatter = plt.scatter(self.df_X, self.df_Y, color='black',
                              alpha=.5, s=10, zorder=1)
        plt.errorbar(self.df_X, self.df_Y, yerr=self.df_ERROR,
                     ls='', ecolor='grey', alpha=.2, zorder=0)
        line = plt.plot(self.df_X, self.df_X, color='blue', zorder=2)
        plt.title(str(self.PATH))
        plt.xlabel(str(self.X))
        plt.ylabel(str(self.Y))
        plt.legend((scatter, line), labels=('Ground truth', 'NN prediction'))
        plt.show()

    def joint_plot(self):
        sns.jointplot(self.df_X, self.df_Y, color='darkblue', kind='reg',
                  scatter_kws={'s': 10, 'alpha': .5}, line_kws={'color': 'black'})
        plt.xlim(7,10.5)
        plt.ylim(7,10)
        plt.xlabel('Mass Ground Truth')
        plt.ylabel('Mass Prediction')
        plt.annotate('Pearson = ' + str(self.pearson())[0:6] + '\nSpearman = ' + str(self.spearman())[0:6] + '\nP-Value = ' + str(self.pvalue())[0:6], xy=(.1, .9),
                 xycoords='axes fraction', size=10)
        plt.show()
### end of class ###


path_to_csv = '/Users/SnehPandya/Desktop/DeepLearningAGN/data/merged_simulated2.csv'
ground_truth = 'Mass_ground_truth'
prediction = 'Mass_prediction'
error = 'z_ground_truth'

NN_results = DataAnalysis(path_to_csv, ground_truth, prediction, error)
NN_results.pearson()
NN_results.spearman()
NN_results.RMSE()
NN_results.OLF(1)
NN_results.OLF(2)
NN_results.mean_error()
NN_results.plot()
NN_results.joint_plot()
