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
    def pearson(self):
        pearson = stats.pearsonr(self.df_X, self.df_Y)
        print('p-value:', pearson[1], ', Pearson coefficient:', pearson[0])

    # spearman summary statistics
    def spearman(self):
        spearman = stats.spearmanr(self.df_X, self.df_Y)
        print('p-value:', spearman[1], ', Spearman coefficient:', spearman[0])

    # Root Mean Squared Error
    def RMSE(self):
        RMSE = sqrt(mean_squared_error(self.df_X, self.df_Y))
        print('RMSE is', RMSE)

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
        print('Outlier Fraction is', float(OLF) * 100,
              '%', 'within', n_sigma, 'standard deviation')

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
        plt.title(str(self.Y) + ' vs ' + str(self.X))
        plt.xlabel(str(self.X))
        plt.ylabel(str(self.Y))
        plt.legend((scatter, line), labels=('Ground truth', 'NN prediction'))
        plt.show()

### end of class ###


path_to_csv = '/Users/SnehPandya/Desktop/DeepLearningAGN/stats_for_joshua/v0526_mass_nll.csv'
ground_truth = 'mass_ground_truth'
prediction = 'mass_prediction'
error = 'mass_error_prediction'

NN_results = DataAnalysis(path_to_csv, ground_truth, prediction, error)
NN_results.pearson()
NN_results.spearman()
NN_results.RMSE()
NN_results.OLF(1)
NN_results.OLF(2)
