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
PEARSON5 = stats.pearsonr(DF['z_ground_truth'], DF['z_prediction'])
PEARSON4 = stats.pearsonr(DF['Mass_ground_truth'], DF['Mass_prediction'])

# We only want the coefficient at the 0th index, [1] returns p value
print('Pearson statistics for z ground truth vs mass ground truth: ', PEARSON1[0])
print('Pearson statistics for z ground truth vs mass prediction: ', PEARSON2[0])
print('Pearson statistics for z prediction vs mass prediction: ', PEARSON3[0])
print('Pearson statistics for mass ground truth vs mass prediction: ', PEARSON4[0])
print('Pearson statistics for z ground truth vs z prediction: ', PEARSON5[0])

########## SPEARMAN STATISTICS ##########

# Using stats's spearmanr function
SPEARMAN1 = stats.spearmanr(DF['z_ground_truth'], DF['Mass_ground_truth'])
SPEARMAN2 = stats.spearmanr(DF['z_ground_truth'], DF['Mass_prediction'])
SPEARMAN3 = stats.spearmanr(DF['z_prediction'], DF['Mass_prediction'])
SPEARMAN4 = stats.spearmanr(DF['z_ground_truth'], DF['z_prediction'])
SPEARMAN5 = stats.spearmanr(DF['Mass_ground_truth'], DF['Mass_prediction'])

print('Spearman statistics for z ground truth vs mass ground truth: ', SPEARMAN1[0])
print('Spearman statistics for z ground truth vs mass prediction: ', SPEARMAN2[0])
print('Spearman statistics for z prediction vs mass prediction: ', SPEARMAN2[0])
print('Spearman statistics for mass ground truth vs mass prediction: ', SPEARMAN4[0])
print('Spearman statistics for z ground truth vs z prediction: ', SPEARMAN5[0])
