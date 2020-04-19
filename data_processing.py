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

    #sample summision
    #id,10,20,30,40,50,60,70,80,90
    # data(yyyy-mm-dd-FIPS),10,20,30,40,50,..,90
    # A prediction should be made for every quantile 
    # for every county for every day from April 1st 
    # to June 30th inclusive.


datadir = "/Users/francescabaldini/Documents/GitHub/caltech_covid_19_modeling/data/international/"

# # aggregate Our World in Data files
# files = glob.glob(datadir + 'health/*.csv') + glob.glob(datadir + 'demographics/*.csv')
# # files = ['health/share-of-adults-who-smoke.csv', 'health/share-deaths-heart-disease.csv', 'health/pneumonia-death-rates-age-standardized.csv', 'health/hospital-beds-per-1000-people.csv', 'health/physicians-per-1000-people.csv', 'demographics/median-age.csv']
# # keys = ['Estimated prevalence (%)', 'Deaths - Cardiovascular diseases - Sex: Both - Age: All Ages (Percent) (%)', 'Hospital beds (per 1,000 people)', 'Physicians (per 1,000 people)', 'UN Population Division (Median Age) (2017) (years)']
# # vals = ['Estimated prevalence of smokers (%)', 'Deaths from Heart Disease (%)', 'Hospital beds (per 1,000 people)', 'Physicians (per 1,000 people)', 'Median Age']

# covid_df = pd.read_csv(datadir + 'covid/our_world_in_data/full_data.csv')
# covid_df.head()
# pandas_profiling.ProfileReport(covid_df)




# %% parh to directory

directory = 'data/us/covid/'
c_df = pd.read_csv(directory + 'JHU_daily_US.csv')
df = c_df.copy()
df.head()
pandas_profiling.ProfileReport(df)

# FIPS - Federal Information Processing Standards code that uniquely identifies counties within the USA.

# %%

df.info(verbose=True)

# %%

df.describe()

# %%

df.columns

# %%

# Check for duplicate and missing values
df.duplicated()
df.isna().head()

# %%

df.head()


# %%
 # check time format

time_format = "%d%b%Y %H:%M"
datetime.now().strftime(time_format)


# %%

# Data Transform

# df["Date"]=pd.to_datetime(df["Date"])
# df["Date"].head(3)


# %%

df["FIPS"].value_counts()

# %% 
# Move 
date_df=df["Date"]
df.drop(labels=['Date'], axis=1,inplace = True)
df.insert(0, 'Date', date_df)

# %%
# df_new=pd.DataFrame()
df_new= df.groupby(["Date","FIPS"], as_index=False)["Confirmed", "Deaths","Recovered","Active"].sum()
df_new.set_index("Date", inplace=True)
# %%

df_new.head()



# %%
FIPS = df_new["FIPS"].to_list()

# %%

# %%

# save to csv

covid= df_new.copy()
covid.to_csv('covind_input.csv')

# %%

import copy

covid.head(3)
data = np.array(covid)

# %%
covid.head(3)


# df_ita = df[['totale_positivi','ricoverati_con_sintomi','deceduti','dimessi_guariti']]

# %%
healed = copy.deepcopy(data[:,1:])
total = 100000 #Assume that total Susceptible is 100000
for i in range(len(data)):
    data[i,0] -= data[i,2] + data[i,3] #calculated the currently comfirmed case
for dat in data:  
    dat[2] = (total - dat[4] - dat[1]) #calculate the rest of Susceptible
for dat in data:
    dat[1] = dat[3]
for i,dat in enumerate(data):
    dat[3] = healed[i][0]

# %%

import torch
import torch.nn as nn
import torch.nn.functional as F

# %%

class SIR(nn.Module):
    def __init__(self):
        super(SIR, self).__init__()
        self.lambda1 = nn.Linear(1,1, bias = False) 
        self.lambda2 = nn.Linear(1,1, bias = False) 
        self.lambda3 = nn.Linear(1,1) 
        self.lambda4 = nn.Linear(1,1, bias = False)
        
    def forward(self, x, n):
        x = x.reshape(5)
        ns = [torch.zeros(5,dtype=torch.float, requires_grad = False) for i in range(n)]
        ns[0] = x
        for i in range(1,n):
            ns[i][0] = ns[i-1][0] - self.lambda2(self.lambda2(ns[i-1][0].view(1,1)))\
                        -  self.lambda4(self.lambda4(ns[i-1][0].view(1,1))) +\
                        self.lambda3(self.lambda3((ns[i-1][0]/1000*ns[i-1][2]).view(1,1)))
            ns[i][1] = ns[i-1][1] + self.lambda2(self.lambda2(ns[i-1][0].view(1,1)))
            ns[i][2] = ns[i-1][2] - self.lambda3(self.lambda3((ns[i-1][0]/1000*ns[i-1][2]).view(1,1)))
            ns[i][3] = ns[i-1][3] + self.lambda4(self.lambda4(ns[i-1][0].view(1,1)))
        ns[1] = ns[1].reshape(1,5)
        for i in range(2,n):
            ns[i] = ns[i].reshape(1,5)
            ns[1] = torch.cat((ns[1],ns[i]), dim=0)
            ns[1] = ns[1].reshape(-1,5)
        return ns[1]

# %%

y = torch.from_numpy(data.astype(np.float32))/100

# %% 
print(y)
print(len(y))

# %%
#downscaled by 100 to avoid gradient explosion
sir = SIR()
opt = torch.optim.Adam(sir.parameters(), lr=0.003)
loss_func = nn.SmoothL1Loss()
best = 10000
for epoch in range(1000):
    out = sir(y[0], len(y))
    out = out.squeeze()
    loss = loss_func(out, y[1:])
    loss.backward()
    opt.step() 
    opt.zero_grad()
    if loss.item() < best:
        best = loss.item()
        torch.save(sir, 'sir.pkl')
    if epoch%100 == 0:        
        print('loss: {}'.format(loss.item()))

# %%

model = torch.load('sir.pkl')
window = 10
y = torch.from_numpy(data.astype(np.float32))/100
h = y[0]
res = model(h,85)
res = res.tolist()

# %%
