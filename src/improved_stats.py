"""
script to run all relevant data analysis on NN Results
"""
# importing packages
import pandas as pd
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
        print('p-value:', pearson[0], ', Pearson coefficient:', pearson[1])

    # spearman summary statistics
    def spearman(self):
        spearman = stats.spearmanr(self.df_X, self.df_Y)
        print('p-value:', spearman[0], ', Spearman coefficient:', spearman[1])

    # Root Mean Squared Error
    def RMSE(self):
        for i in range(len(self.df)):
            result = np.mean((self.df_X - self.df_Y)**2)
            result = result ** (1 / 2)
        print('RMSE is', result)

    # Ooutlier Fraction
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
        sns.set_context('paper')
        sns.set(font='times new roman')
        plt.scatter(self.df_X, self.df_Y, color='black',
                    alpha=.5, s=10, zorder=1)
        plt.errorbar(self.df_X, self.df_Y, yerr=self.df_ERROR,
                     ls='', ecolor='grey', alpha=.2, zorder=0)
        plt.plot(self.df_X, self.df_X, color='blue', zorder=2)
        plt.title(str(self.X) + ' vs ' + str(self.Y))
        plt.xlabel(str(self.X))
        plt.ylabel(str(self.Y))
        plt.legend()
        plt.show()

### end of class ###


path_to_csv = '/Users/SnehPandya/Desktop/DeepLearningAGN/data/2020-05-26Mass_nll_b1.csv'
ground_truth = 'Mass_ground_truth'
prediction = 'Mass_prediction'
error = 'Mass_error'

csv = DataAnalysis(path_to_csv, ground_truth, prediction, error)
print(csv.OLF(1))
