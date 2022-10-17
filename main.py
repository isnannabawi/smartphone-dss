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
#print(gsm_spec_label)

gsm_spec = []
for row in csvreader:
        gsm_spec.append([float(i) for i in row])

gsm_spec = np.array(gsm_spec)

#1 n_network_technology
#2 n_body_weight
#3 n_display_type
#4 n_display_size
#5 n_micro_sdcard
#6 n_sound_3p5mm
#7 n_bluetooth
#8 n_gps
#9 n_comms_usb
#10 n_usb_conn
#11 n_fingerprint
#12 n_android_ver
#13 n_main_camera_resolution
#14 n_main_camera_af
#15 n_main_camera_ois
#16 n_battery_capacity
#17 n_total_cpu_speed
#18 n_internal_memory
#19 n_ram

# select feature
#feature_selected = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
feature_selected = [2]

for x in range (len(feature_selected)):
    #print(feature_selected[x])
    print(gsm_spec_label[feature_selected[x]])

# normalize with min max scale
gsm_spec = preprocessing.minmax_scale(gsm_spec[:,feature_selected],axis=0)

# Extract features
#gsm_spec = gsm_spec[:, feature_selected]

#print(gsm_spec)

# Build a 3x1 SOM (3 clusters)
som = SOM(m=8, n=1, dim=len(feature_selected), random_state=0)

# Fit it to the data
som.fit(gsm_spec)

# Assign each datapoint to its predicted cluster
predictions = som.predict(gsm_spec)
predictions = np.array(predictions)
#print(predictions)

# Plot the results
for d in range (len(feature_selected)):
    y = predictions[:]
    x = gsm_spec[:,d]
    #colors = ['red', 'green', 'blue']
    #print(x)
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.title.set_text(gsm_spec_label[feature_selected[d]])
    figname = 'result' + str(gsm_spec_label[feature_selected[d]])
    plt.savefig(figname)
#/////////////////////////


now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Time Finished = ", current_time)