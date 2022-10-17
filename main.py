"""
@author: Isnan Nabawi
"""
print("..:: SMARTPHONE DSS ::..")
# start time
from datetime import datetime

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Time Start = ", current_time)

import matplotlib.pyplot as plt
import csv
import numpy as np

from matplotlib.colors import ListedColormap
from sklearn_som.som import SOM
from sklearn import preprocessing

# load gsm data
file = open('gsm.csv')
type(file)

# print header
gsm_spec_label = []
csvreader = csv.reader(file)
header = []
header = next(csvreader)

gsm_spec_label = np.array(header)
print(gsm_spec_label)

gsm_spec = []
for row in csvreader:
        gsm_spec.append([float(i) for i in row])

gsm_spec = np.array(gsm_spec)

# select feature
feature_selected = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

#print(gsm_spec_label[feature_selected[0]])

for x in feature_selected:
    print(gsm_spec_label[feature_selected[x-1]])

# normalize with min max scale
gsm_spec = preprocessing.minmax_scale(gsm_spec[:,feature_selected],axis=0)

# Extract features
#gsm_spec = gsm_spec[:, 1:20]

#print(gsm_spec)

# Build a 3x1 SOM (3 clusters)
som = SOM(m=3, n=2, dim=19, random_state=1234)

# Fit it to the data
som.fit(gsm_spec)

# Assign each datapoint to its predicted cluster
predictions = som.predict(gsm_spec)
predictions = np.array(predictions)
#print(predictions)

print(len(feature_selected))
# Plot the results
for d in feature_selected:
    y = predictions[:]
    x = gsm_spec[:,d-1]
    #colors = ['red', 'green', 'blue']
    #print(x)
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.title.set_text(gsm_spec_label[d])
    figname = 'result' + str(gsm_spec_label[d-1])
    plt.savefig(figname)
#/////////////////////////


now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Time Finished = ", current_time)