# Weighing Black Holes Using Deep Learning
_**Contributors: Prof. Xin Liu, Joshua Yao-Yu Lin, Devanshi Pratap and Sneh Pandya**_

The goal of this project is to introduce a new, interdisciplinary method in weighing supermassive black holes (SMBH).

# Abstract
Supermassive black holes are ubiquitously found at the center of galaxies. Also known as quasars, they are actively accreting hot gas and material and are extremely difficult to observe, let alone observe the necessary information to determine mass.

There is currently no efficient method for accurately weighing supermassive black holes outside of our galaxy.  Developing a catalog of quasar masses is important in understanding large scale structure evolution of galaxies, as well using them as “standard candles” in astronomy. The emergence of astroinformatics and continuous applications of computer science, and most notably deep learning (DL) in astronomy has motivated this project.

The goal of this project is to develop an algorithm that weighs SMBH using quasar time series instead of atomic spectra. There are theoretical reasons to believe this relationship between time series data and black hole mass exist, and is pioneered Prof. Xin Liu. The theorized non-linearity motivates the use of DL in this project. The questions addressed are whether current methods in DL can be used to make accurate black hole mass predictions of known data sets.

# Results
_Will attach images later._

# Code
## Getting data in desired format

**plot_LC_stripe82.py**: To plot the Stripe 82 Light Curve data in a plot. Also, cleans unphysical data like negative magnitudes.

**data_matching.py**: To match the Stripe 82 and the DR7 catalog data to get most relevant information.

**LC_to_image.py**: To convert the matched data into images.

## Deep learning
_Will include files later._

# References
_Will include a detailed list later._
