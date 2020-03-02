"""
Read SDSS Stripe 82 data and clean it
Generate a .csv file of cleaned data (data in a desirable format)
Source: http://faculty.washington.edu/ivezic/macleod/qso_dr7/Southern.html
"""
import pandas as pd
from astropy.io import ascii

# reading the raw data file
DATA = ascii.read("./DB_QSO_S82.dat")

# extracting useful columns from the DATA file
ID = DATA.field("col1")
RA = DATA.field('col2')
DEC = DATA.field('col3')
Z = DATA.field("col7")
BH_MASS = DATA.field("col8")

# converting the ID to an integer 
ID = [int(i) for i in ID]

# generating columns for the cleaned Stripe 82 data
X_TRAIN = pd.DataFrame(ID, columns=['ID'])
X_TRAIN['ra'] = RA
X_TRAIN['dec'] = DEC
X_TRAIN['z'] = Z
X_TRAIN['BH_mass'] = BH_MASS
X_TRAIN.to_csv("clean_DB_QSO_S82.csv")
