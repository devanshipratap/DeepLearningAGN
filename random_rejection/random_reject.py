# importing packages
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.transform import resize

# path to folder containing original data
PATH = '/Users/SnehPandya/Desktop/Black_Hole_NN/raw_data/original_LC/'
FILES = os.listdir(PATH)
ROOT_FOLDER = '/Users/SnehPandya/Desktop/DeepLearningAGN/random_rejection/npy_files/'

# make sure to change seed between runs
SEED = np.random.seed(1)

# fraction to randomly remove from LC data
REMOVE_PERCENT = .1

# Read light curve files
for file_name in FILES:

    with open('/Users/SnehPandya/Desktop/Black_Hole_NN/raw_data/original_LC/70', 'r') as f:  # opening LC files
        next(f)  # skip first row
        df = pd.DataFrame(l.rstrip().split() for l in f)  # define dataframe

    # converting str to float
    df = df.apply(pd.to_numeric, errors='ignore')

    # renaming columns
    df.columns = ['MJD_u', 'u', 'u_error', 'MJD_g', 'g', 'g_error', 'MJD_r', 'r',
                  'r_error', 'MJD_i', 'i', 'i_error', 'MJD_z', 'z', 'z_error', 'RA', 'DEC']

    # filtering unphysical magnitudes across all bands
    df_u = df[(df['u'] > 1) & (df['u'] < 99)]
    df_g = df[(df['g'] > 1) & (df['g'] < 99)]
    df_r = df[(df['r'] > 1) & (df['r'] < 99)]
    df_i = df[(df['i'] > 1) & (df['i'] < 99)]
    df_z = df[(df['z'] > 1) & (df['z'] < 99)]

    # randomly remove percentage of data by row
    df_u = df_u.sample(frac=1 - REMOVE_PERCENT, random_state=SEED)
    df_g = df_g.sample(frac=1 - REMOVE_PERCENT, random_state=SEED)
    df_r = df_r.sample(frac=1 - REMOVE_PERCENT, random_state=SEED)
    df_i = df_i.sample(frac=1 - REMOVE_PERCENT, random_state=SEED)
    df_z = df_z.sample(frac=1 - REMOVE_PERCENT, random_state=SEED)

    # sort in ascending order
    df_u = df_u.sort_values(by='MJD_u')
    df_g = df_g.sort_values(by='MJD_g')
    df_r = df_r.sort_values(by='MJD_r')
    df_i = df_i.sort_values(by='MJD_i')
    df_z = df_z.sort_values(by='MJD_z')

    # Gather the 5 band data and set mjd init = 0
    u_data = round(df_u['MJD_u']) - round(df_u['MJD_u'].iloc[0])
    g_data = round(df_g['MJD_g']) - round(df_g['MJD_g'].iloc[0])
    r_data = round(df_r['MJD_r']) - round(df_r['MJD_r'].iloc[0])
    i_data = round(df_i['MJD_i']) - round(df_i['MJD_i'].iloc[0])
    z_data = round(df_z['MJD_z']) - round(df_z['MJD_z'].iloc[0])

    # Convert 5 band data into 5 (bands) x 3340 (days) image data
    Images = np.zeros((5, 3340))

    for i, day in enumerate(u_data):
        Images[0, int(day)] += df_u['u'].iloc[i]

    for i, day in enumerate(g_data):
        Images[1, int(day)] += df_g['g'].iloc[i]

    for i, day in enumerate(r_data):
        Images[2, int(day)] += df_r['r'].iloc[i]

    for i, day in enumerate(i_data):
        Images[3, int(day)] += df_i['i'].iloc[i]

    for i, day in enumerate(z_data):
        Images[4, int(day)] += df_z['z'].iloc[i]

    # Reshape image into 167 x 100
    reshape_img = Images.reshape(167, 100)

    # padding zeros to make the 167 x 100 image -> 224 x 224
    Padding_images = np.zeros((224, 224))
    Padding_images[:167, :100] = reshape_img

    final_image = np.asarray((reshape_img / 40) * 255., dtype=np.int32)
    # sanity check for how image looks, doesnt work at the moment
    # plt.plot(final_image)
    # plt.show()

    # save numpy files
    np.save(ROOT_FOLDER + 'lc_image_{}.npy'.format(file_name), reshape_img)
