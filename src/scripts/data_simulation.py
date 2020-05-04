"""
Module to generate a 10x data set of simulated light curves
"""
import os
import gc
import pandas as pd
import numpy as np
from skimage.transform import resize


# Save the simulated light curve files from this path
SAVE_PATH = './simulated_data/'

# Create the above directory if it does not exist
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

def saving_images(df, file_name):
    """
    Save the simulated light curve as a .npy file
    """
    u_data = round(df[0]) - round(df[0].iloc[0])
    g_data = round(df[3]) - round(df[3].iloc[0])
    r_data = round(df[6]) - round(df[6].iloc[0])
    i_data = round(df[9]) - round(df[9].iloc[0])
    z_data = round(df[12]) - round(df[12].iloc[0])

    images = np.zeros((5, 3340))

    # Using normal distribution as noise with sigma = error in the color channel (u, g, r, i z)
    for i, day in enumerate(u_data):
        images[0, int(day)] += df[1].iloc[i] + np.random.normal(0, df[2].iloc[i])
    for i, day in enumerate(g_data):
        images[1, int(day)] += df[4].iloc[i] + np.random.normal(0, df[5].iloc[i])
    for i, day in enumerate(r_data):
        images[2, int(day)] += df[7].iloc[i] + np.random.normal(0, df[8].iloc[i])
    for i, day in enumerate(i_data):
        images[3, int(day)] += df[10].iloc[i] + np.random.normal(0, df[11].iloc[i])
    for i, day in enumerate(z_data):
        images[4, int(day)] += df[13].iloc[i] + np.random.normal(0, df[14].iloc[i])

    # Remove anomalies and reshape image into 167 x 100
    images = np.clip(images, 0, None)
    reshape_img = images.reshape(167, 100)

    # Resize images to 224 x 224 (same as ImageNet)
    resize_img = reshape_img.copy()
    resize_img = resize(resize_img, (224, 224))
    np.save(SAVE_PATH + 'LC_images_{}.npy'.format(file_name), resize_img)

    del images, reshape_img, resize_img

# Path where the original light curves reside
PATH = './light_curves/stripe82/'
FILES = os.listdir(PATH)

# Loop using 10 random seeds
for seed in range(10):
    np.random.seed(seed=seed)

    for idx, file in enumerate(FILES):
        # Priming data
        with open(PATH + file, 'r') as f:
            # Skip 0th row
            next(f)
            data_frame = pd.DataFrame(l.rstrip().split() for l in f)

        # Converting type to float64
        for j in range(17):
            data_frame[j] = pd.to_numeric(data_frame[j])
        saving_images(data_frame, file)

        # For testing:
        # if idx >= 10:
        #     break

        gc.collect()
