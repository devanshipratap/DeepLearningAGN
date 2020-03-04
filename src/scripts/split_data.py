"""
Module to split data into training and testing sets
"""
import os
from shutil import copyfile
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_FOLDER = "./full_train/"
TRAIN_FOLDER = "train/"
TEST_FOLDER = "test/"
FULL_DATA = pd.read_csv("clean_full_data.csv")

if not os.path.exists(ROOT_FOLDER + TRAIN_FOLDER):
    os.mkdir(ROOT_FOLDER + TRAIN_FOLDER)

if not os.path.exists(ROOT_FOLDER + TEST_FOLDER):
    os.mkdir(ROOT_FOLDER + TEST_FOLDER)

TRAIN, TEST = train_test_split(FULL_DATA, test_size=0.2, random_state=42)
TRAIN.to_csv(ROOT_FOLDER + TRAIN_FOLDER + "train.csv")
TEST.to_csv(ROOT_FOLDER + TEST_FOLDER + "test.csv")

for i in range(TRAIN.shape[0]):
    ID = TRAIN['ID'].iloc[i]
    img_path = ROOT_FOLDER + 'LC_images_' + str(ID) + '.npy'
    try:
        copyfile(img_path, ROOT_FOLDER + TRAIN_FOLDER + 'LC_images_' + str(ID) + '.npy')
    except:
        print("Missing: ", ID)

for i in range(TEST.shape[0]):
    ID = TEST['ID'].iloc[i]
    img_path = ROOT_FOLDER + 'LC_images_' + str(ID) + '.npy'
    try:
        copyfile(img_path, ROOT_FOLDER + TEST_FOLDER + 'LC_images_' + str(ID) + '.npy')
    except:
        print("Missing: ", ID)
