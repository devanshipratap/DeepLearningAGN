"""
Convert SDSS Stripe 82 light curves to images
as .npy files to be readable by neural network
"""
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.transform import resize

# Create full_train folder
ROOT_FOLDER = "./full_train/"
if not os.path.exists(ROOT_FOLDER):
    os.mkdir(ROOT_FOLDER)

# Path to folder containing raw light curves
PATH = '../Black_holes_NN/QSO_LCs/QSO_S82/'
FILES = os.listdir(PATH)



### Read Light curve file
for file_name in FILES:
    #print(file_name)
# file_name = "./QSO_LCs/QSO_S82/1001265"
# name = "1001265"


    with open(PATH + file_name,'r') as f:
        next(f) # skip first row
        #next(f)
        df = pd.DataFrame(l.rstrip().split() for l in f)


    ### convert str into numeric
    for i in range(14):
        df[i] = pd.to_numeric(df[i], errors='ignore')

    ### clean up anomalies for different passband



    df_u = df[(50>df[1]) & (df[1]>1)]
    df_g = df[(50>df[4]) & (df[4]>1)]
    df_r = df[(50>df[7]) & (df[7]>1)]
    df_i = df[(50>df[10]) & (df[10]>1)]
    df_z = df[(50>df[13]) & (df[13]>1)]

    #print("df shape:",df.shape)
    df = df[(50>df[1]) & (df[1]>1)]
    df = df[(50>df[4]) & (df[4]>1)]
    df = df[(50>df[7]) & (df[7]>1)]
    df = df[(50>df[10]) & (df[10]>1)]
    df = df[(50>df[13]) & (df[13]>1)]
    #print("df shape:",df.shape)
    #print(df[1].max())
#     df_u = df[df[1]<50]
#     df_g = df[df[4]<50]
#     df_r = df[df[7]<50]
#     df_i = df[df[10]<50]
#     df_z = df[df[13]<50]
    ###generating data
    Images = np.zeros((5, 3340))
    try:
        u_data = round(df[0]) - round(df[0].iloc[0])
        g_data = round(df[3]) - round(df[3].iloc[0])
        r_data = round(df[6]) - round(df[6].iloc[0])
        i_data = round(df[9]) - round(df[9].iloc[0])
        z_data = round(df[12]) - round(df[12].iloc[0])




        for i, day in enumerate(u_data):
            Images[0, int(day)] = 0
            Images[0, int(day)] += df[1].iloc[i]#3631 * 10**(df[1].iloc[i]/-2.5)

        for i, day in enumerate(g_data):
            Images[1, int(day)] = 0
            Images[1, int(day)] += df[4].iloc[i]#3631 * 10**(df[4].iloc[i]/-2.5)

        for i, day in enumerate(r_data):
            Images[2, int(day)] = 0
            Images[2, int(day)] += df[7].iloc[i]#3631 * 10**(df[7].iloc[i]/-2.5)

        for i, day in enumerate(i_data):
            Images[3, int(day)] = 0
            Images[3, int(day)] += df[10].iloc[i]#3631 * 10**(df[10].iloc[i]/-2.5)

        for i, day in enumerate(z_data):
            Images[4, int(day)] = 0
            Images[4, int(day)] += df[13].iloc[i]#3631 * 10**(df[13].iloc[i]/-2.5)
    except:
        print("failed data")
        pass




    ### remove anomalies and reshape image into 167X100
    Images = np.clip(Images, 0, None)

    reshape_img = Images.reshape(167, 100)

    ### Option1: resize images to 224X224 (same as ImageNet)

    from skimage.transform import resize
    resize_img = reshape_img.copy()
    resize_img = resize(resize_img, (224, 224), anti_aliasing=True)


    np.save(ROOT_FOLDER + "LC_images_{}.npy".format(file_name), resize_img)



    ### save image as png file with 0-255 format
#     final_image = np.asarray( (resize_img / 0.001) * 255., dtype=np.int32)
#     cv2.imwrite(root_folder + "LC_images_{}.png".format(file_name), final_image)
    # print(resize_img.shape)

    # plt.imshow(resize_img)
    # plt.colorbar()
    # plt.title("resize_with_interpolation")
    # plt.show()



    ### Option2: padding zeros to make the 167X100 image -> 224X224

    Padding_images = np.zeros((224, 224))

    Padding_images[:167, :100] = reshape_img
