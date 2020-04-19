# %%
import pandas as pd
import pandas_profiling

import numpy as np
import math
from datetime import datetime
import os
import glob

import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

# Library used for country name evaluation
import country_converter as coco
from opencage.geocoder import OpenCageGeocode
from pprint import pprint

# %%

## WHAT WE NEED AS OUTPUT 

# we need:
    # confirmed cases
    # deaths
    # ricoverati 
    # healed/recovered
    # number of individual

# group

    # counties
    # states
    # location
    # age
    # gender
    # pollution


## WHAT WE NEED AS OUTPUT 

    # quantili 
        #sample summision
        #id,10,20,30,40,50,60,70,80,90
        # data,10,20,30,40,50,..,90



datadir = "data/international/"

# aggregate Our World in Data files
files = glob.glob(datadir + 'health/*.csv') + glob.glob(datadir + 'demographics/*.csv')
# files = ['health/share-of-adults-who-smoke.csv', 'health/share-deaths-heart-disease.csv', 'health/pneumonia-death-rates-age-standardized.csv', 'health/hospital-beds-per-1000-people.csv', 'health/physicians-per-1000-people.csv', 'demographics/median-age.csv']
# keys = ['Estimated prevalence (%)', 'Deaths - Cardiovascular diseases - Sex: Both - Age: All Ages (Percent) (%)', 'Hospital beds (per 1,000 people)', 'Physicians (per 1,000 people)', 'UN Population Division (Median Age) (2017) (years)']
# vals = ['Estimated prevalence of smokers (%)', 'Deaths from Heart Disease (%)', 'Hospital beds (per 1,000 people)', 'Physicians (per 1,000 people)', 'Median Age']

covid_df = pd.read_csv(datadir + 'covid/our_world_in_data/full_data.csv')
covid_df.head()
pandas_profiling.ProfileReport(covid_df)




# %% parh to directory
directory = 'data/us/covid/'
c_df = pd.read_csv(directory + 'JHU_daily_US.csv')
df = c_df.copy()
df.head()

# FIPS - Federal Information Processing Standards code that uniquely identifies counties within the USA.

# %%
 # check time format

time_format = "%d%b%Y %H:%M"
datetime.now().strftime(time_format)


# %%
