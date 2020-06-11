import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('/Users/SnehPandya/Desktop/DeepLearningAGN/data/running_median_fixed.csv')
df = df.apply(pd.to_numeric, errors = 'ignore')
df1= df[(df['Mass_ground_truth'] >= 4) & (df['Mass_ground_truth'] <= 4.5)]
print(df1)
df2= df[(df['Mass_ground_truth'] >= 4.5) & (df['Mass_ground_truth'] <= 5)]
print(df2['Mass_ground_truth'])
df3= df[(df['Mass_ground_truth'] >= 5) & (df['Mass_ground_truth'] <= 5.5)]
df4= df[(df['Mass_ground_truth'] >= 5.5) & (df['Mass_ground_truth'] <= 6)]
df5= df[(df['Mass_ground_truth'] >= 6) & (df['Mass_ground_truth'] <= 6.5)]
df6= df[(df['Mass_ground_truth'] >= 6.5) & (df['Mass_ground_truth'] <= 7)]
df7= df[(df['Mass_ground_truth'] >= 7) & (df['Mass_ground_truth'] <= 7.3)]
df8= df[(df['Mass_ground_truth'] >= 7.3) & (df['Mass_ground_truth'] <= 7.6)]
df9= df[(df['Mass_ground_truth'] >= 7.6) & (df['Mass_ground_truth'] <= 7.9)]
df10= df[(df['Mass_ground_truth'] >= 7.9) & (df['Mass_ground_truth'] <= 8.2)]
df11= df[(df['Mass_ground_truth'] >= 8.2) & (df['Mass_ground_truth'] <= 8.5)]
df12= df[(df['Mass_ground_truth'] >= 8.5) & (df['Mass_ground_truth'] <= 8.8)]
df13= df[(df['Mass_ground_truth'] >= 8.8) & (df['Mass_ground_truth'] <= 9.1)]
df14= df[(df['Mass_ground_truth'] >= 9.1) & (df['Mass_ground_truth'] <= 9.4)]
df15= df[(df['Mass_ground_truth'] >= 9.4) & (df['Mass_ground_truth'] <= 9.7)]
df16= df[(df['Mass_ground_truth'] >= 9.7) & (df['Mass_ground_truth'] <= 10)]
df17= df[(df['Mass_ground_truth'] >= 10) & (df['Mass_ground_truth'] <= 10.5)]
df18= df[(df['Mass_ground_truth'] >= 10.5) & (df['Mass_ground_truth'] <= 11)]
df19= df[(df['Mass_ground_truth'] >= 11) & (df['Mass_ground_truth'] <= 11.5)]
df20= df[(df['Mass_ground_truth'] >= 11.5) & (df['Mass_ground_truth'] <= 12)]
df21= df[(df['Mass_ground_truth'] >= 12) & (df['Mass_ground_truth'] <= 12.5)]
df22= df[(df['Mass_ground_truth'] >= 12.5) & (df['Mass_ground_truth'] <= 13)]

# print(df['Mass_ground_truth'])
plt.figure(figsize = (6,6))
sns.set(font='Times New Roman')
plt.title('Comparison with Traditional Method')
plt.xlabel('LOG(M/M_sun)')
plt.ylabel('LOG(M/M_sun)')
plt.ylim(7,10.5)
ground_truth = plt.plot(df['Mass_ground_truth'],df['Mass_ground_truth'],color='black',zorder=2,label = 'ground_truth')
prediction = plt.scatter(df['Mass_ground_truth'],df['Mass_prediction'],s=10,color='blue',zorder=1, label = 'nn prediction')
# bestfit = plt.scatter(df['Mass_ground_truth'], df['M_tau'],color = 'orange',s=1,alpha = .7)
damping = plt.scatter(df1['Mass_ground_truth'],df1['M_4_4.5'],s=1,color='red',zorder=1)
plt.scatter(df2['Mass_ground_truth'],df2['M_4.5_5'],s=1,color='red',zorder=1)
plt.scatter(df3['Mass_ground_truth'],df3['M_5_5.5'],s=1,color='red',zorder=1)
plt.scatter(df4['Mass_ground_truth'],df4['M_5.5_6'],s=1,color='red',zorder=1)
plt.scatter(df5['Mass_ground_truth'],df5['M_6_6.5'],s=1,color='red',zorder=1)
plt.scatter(df6['Mass_ground_truth'],df6['M_6.5_7'],s=1,color='red',zorder=1)
plt.scatter(df7['Mass_ground_truth'],df7['M_7_7.3'],s=1,color='red',zorder=1)
plt.scatter(df8['Mass_ground_truth'],df8['M_7.3_7.6'],s=1,color='red',zorder=1)
plt.scatter(df9['Mass_ground_truth'],df9['M_7.6_7.9'],s=1,color='red',zorder=1)
plt.scatter(df10['Mass_ground_truth'],df10['M_7.9_8.2'],s=1,color='red',zorder=1)
plt.scatter(df11['Mass_ground_truth'],df11['M_8.2_8.5'],s=1,color='red',zorder=1)
plt.scatter(df12['Mass_ground_truth'],df12['M_8.5_8.8'],s=1,color='red',zorder=1)
plt.scatter(df13['Mass_ground_truth'],df13['M_8.8_9.1'],s=1,color='red',zorder=1)
plt.scatter(df14['Mass_ground_truth'],df14['M_9.1_9.4'],s=1,color='red',zorder=1)
plt.scatter(df15['Mass_ground_truth'],df15['M_9.4_9.7'],s=1,color='red',zorder=1)
plt.scatter(df16['Mass_ground_truth'],df16['M_9.7_10'],s=1,color='red',zorder=1)
plt.scatter(df17['Mass_ground_truth'],df17['M_10_10.5'],s=1,color='red',zorder=1)
plt.scatter(df18['Mass_ground_truth'],df18['M_10.5_11'],s=1,color='red',zorder=1)
plt.scatter(df19['Mass_ground_truth'],df19['M_11_11.5'],s=1,color='red',zorder=1)
plt.scatter(df20['Mass_ground_truth'],df20['M_11.5_12'],s=1,color='red',zorder=1)
plt.scatter(df21['Mass_ground_truth'],df21['M_12_12.5'],s=1,color='red',zorder=1)
plt.scatter(df22['Mass_ground_truth'],df22['M_12.5_13'],s=1,color='red',zorder=1)
plt.legend((ground_truth, prediction, damping), labels = ('ground truth', 'NN prediction','best-fit damping running median'))
# error = [df['m_err_low'],df['m_err_hi']]
# plt.errorbar(df2['Mass_ground_truth'],df2['M_4.5_5'], yerr = error, ls='',alpha = .4,color ='red',zorder=0,label = 'error (non-gaussian)')
plt.show()
