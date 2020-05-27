"""
Use Pearson and Spearman correlation coefficients
to summarize results of our neural network
to
"""
import pandas as pd
from scipy import stats

# Reading the data to be summarized
DF = pd.read_csv('../../data/merged_simulated.csv')

# Cleaning unphysical masses
DF = DF[(DF.Mass_ground_truth != 0)]

########## PEARSON STATISTICS ##########

# Using stats's pearsonr function
PEARSON1 = stats.pearsonr(DF['z_ground_truth'], DF['Mass_ground_truth'])
PEARSON2 = stats.pearsonr(DF['z_ground_truth'], DF['Mass_prediction'])
PEARSON3 = stats.pearsonr(DF['z_prediction'], DF['Mass_prediction'])
PEARSON4 = stats.pearsonr(DF['Mass_ground_truth'], DF['Mass_prediction'])
PEARSON5 = stats.pearsonr(DF['z_ground_truth'], DF['z_prediction'])
PEARSON6 = stats.pearsonr(DF['Mass_ground_truth'], DF['z_prediction'])

# [0] contains the coefficient, [1] returns p value
print('Pearson statistics for mass ground truth vs z ground truth: ', PEARSON1)
print('Pearson statistics for mass prediction vs z ground truth: ', PEARSON2)
print('Pearson statistics for mass prediction vs z prediction: ', PEARSON3)
print('Pearson statistics for mass prediction vs mass ground truth: ', PEARSON4)
print('Pearson statistics for z prediction vs z ground truth: ', PEARSON5)
print('Pearson statistics for z prediction vs mass ground truth: ', PEARSON6)

########## SPEARMAN STATISTICS ##########

# Using stats's spearmanr function
SPEARMAN1 = stats.spearmanr(DF['z_ground_truth'], DF['Mass_ground_truth'])
SPEARMAN2 = stats.spearmanr(DF['z_ground_truth'], DF['Mass_prediction'])
SPEARMAN3 = stats.spearmanr(DF['z_prediction'], DF['Mass_prediction'])
SPEARMAN4 = stats.spearmanr(DF['Mass_ground_truth'], DF['Mass_prediction'])
SPEARMAN5 = stats.spearmanr(DF['z_ground_truth'], DF['z_prediction'])
SPEARMAN6 = stats.pearsonr(DF['Mass_ground_truth'], DF['z_prediction'])

# [0] contains the coefficient, [1] returns p value
print('Spearman statistics for mass ground truth vs z ground truth: ', SPEARMAN1)
print('Spearman statistics for mass prediction vs z ground truth: ', SPEARMAN2)
print('Spearman statistics for mass prediction vs z prediction: ', SPEARMAN3)
print('Spearman statistics for mass prediction vs mass ground truth: ', SPEARMAN4)
print('Spearman statistics for z prediction vs z ground truth: ', SPEARMAN5)
print('Spearman statistics for z prediction vs mass ground truth: ', SPEARMAN6)
