"""
Read SDSS DR7 data and clean it
Generate a .csv file of cleaned data (data in a desirable format)
Source: http://quasar.astro.illinois.edu/BH_mass/dr7.htm
"""
import pandas as pd
from astropy.io import fits

# Reading the raw data file
FILE_PATH = './dr7_bh_Nov19_2013.fits'
DR7_BH = fits.open(FILE_PATH)

# Extracting useful columns from the DR7_BH file
DATA = DR7_BH[1].data
SDSS_NAME = DATA.field('SDSS_NAME')
RA = DATA.field('ra')
DEC = DATA.field('dec')
M = DATA.field('LOGBH')

# Generating columns for the cleaned DR7 data
X_TRAIN = pd.DataFrame(SDSS_NAME, columns=['SDSS_Name'])
X_TRAIN['ra'] = RA
X_TRAIN['dec'] = DEC
X_TRAIN['M'] = M

# Generate csv file of cleaned DR7 data
X_TRAIN.to_csv('clean_dr7.csv')
