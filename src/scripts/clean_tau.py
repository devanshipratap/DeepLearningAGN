import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_csv(
    '/Users/SnehPandya/Desktop/DeepLearningAGN/data/TAU_matched_new.csv', index_col=0)
print(df.shape)
df = df.drop(columns=['ID.1','z.1','ERR' ])
df = df.drop(df[df.tau == 5].index) #unphysical tau
df = df.drop(df[df.tau == 7.5].index) #unphysical tau
df = df.drop(df[df.tau < 0].index)
df = df.drop(df[df.M_tau < 0].index)
df = df.drop(df[df.M_tau > 12].index)
## dropped 470 data points ##


df['m_err_low'] = df.apply(lambda row: (row.tau_lim_lo + 2.29) / 0.56, axis=1)
df['m_err_hi'] = df.apply(lambda row: (row.tau_lim_hi + 2.29) / 0.56, axis=1)


df['M_4_4.5'] = df.apply(lambda row: row.M_tau if (4 < row.Mass_ground_truth < 4.5) else np.nan, axis=1).median()
df['M_4.5_5'] = df.apply(lambda row: row.M_tau if (4.5 < row.Mass_ground_truth < 5) else np.nan, axis=1).median()
df['M_5_5.5'] = df.apply(lambda row: row.M_tau if (5 < row.Mass_ground_truth < 5.5) else np.nan, axis=1).median()
df['M_5.5_6'] = df.apply(lambda row: row.M_tau if (5.5 < row.Mass_ground_truth < 6) else np.nan, axis=1).median()
df['M_6_6.5'] = df.apply(lambda row: row.M_tau if (6 < row.Mass_ground_truth < 6.5) else np.nan, axis=1).median()
df['M_6.5_7'] = df.apply(lambda row: row.M_tau if (6.5 < row.Mass_ground_truth < 7) else np.nan, axis=1).median()
df['M_7_7.3'] = df.apply(lambda row: row.M_tau if (7 < row.Mass_ground_truth < 7.3) else np.nan, axis=1).median()
df['M_7.3_7.6'] = df.apply(lambda row: row.M_tau if (7.3 < row.Mass_ground_truth < 7.6) else np.nan, axis=1).median()
df['M_7.6_7.9'] = df.apply(lambda row: row.M_tau if (7.6 < row.Mass_ground_truth < 7.9) else np.nan, axis=1).median()
df['M_7.9_8.2'] = df.apply(lambda row: row.M_tau if (7.9 < row.Mass_ground_truth < 8.2) else np.nan, axis=1).median()
df['M_8.2_8.5'] = df.apply(lambda row: row.M_tau if (8.2 < row.Mass_ground_truth < 8.5) else np.nan, axis=1).median()
df['M_8.5_8.8'] = df.apply(lambda row: row.M_tau if (8.5 < row.Mass_ground_truth < 8.8) else np.nan, axis=1).median()
df['M_8.8_9.1'] = df.apply(lambda row: row.M_tau if (8.8 < row.Mass_ground_truth < 9.1) else np.nan, axis=1).median()
df['M_9.1_9.4'] = df.apply(lambda row: row.M_tau if (9.1 < row.Mass_ground_truth < 9.4) else np.nan, axis=1).median()
df['M_9.4_9.7'] = df.apply(lambda row: row.M_tau if (9.4 < row.Mass_ground_truth < 9.7) else np.nan, axis=1).median()
df['M_9.7_10'] = df.apply(lambda row: row.M_tau if (9.7 < row.Mass_ground_truth < 10) else np.nan, axis=1).median()
df['M_10_10.5'] = df.apply(lambda row: row.M_tau if (10 < row.Mass_ground_truth < 10.5) else np.nan, axis=1).median()
df['M_10.5_11'] = df.apply(lambda row: row.M_tau if (10.5 < row.Mass_ground_truth < 11) else np.nan, axis=1).median()
df['M_11_11.5'] = df.apply(lambda row: row.M_tau if (11 < row.Mass_ground_truth < 11.5) else np.nan, axis=1).median()
df['M_11.5_12'] = df.apply(lambda row: row.M_tau if (11.5 < row.Mass_ground_truth < 12) else np.nan, axis=1).median()
df['M_12_12.5'] = df.apply(lambda row: row.M_tau if (12 < row.Mass_ground_truth < 12.5 ) else np.nan, axis=1).median()
df['M_12.5_13'] = df.apply(lambda row: row.M_tau if (12.5 < row.Mass_ground_truth < 13 ) else np.nan, axis=1).median()
df['error_up_median'] = df.apply(lambda row: row.m_err_hi if (12.5 < row.Mass_ground_truth < 13 ) else np.nan, axis=1)
df['error_lo_median'] = df.apply(lambda row: row.m_err_low if (12.5 < row.Mass_ground_truth < 13 ) else np.nan, axis=1)
print(df['error_lo_median'].median())
print(df['error_lo_median'].median())


# df.to_csv('/Users/SnehPandya/Desktop/DeepLearningAGN/data/running_median_fixed.csv')
