"""
@author: Isnan Nabawi
"""
print("..:: SMARTPHONE DSS ::..")
# start time
from datetime import datetime

now = datetime.now()

start_time = now.strftime("%H:%M:%S")
print("Time Start = ", start_time)

import matplotlib.pyplot as plt
import csv
import numpy as np

from matplotlib.colors import ListedColormap
from sklearn_som.som import SOM
from sklearn import preprocessing

# load gsm data
file = open('gsm.csv')
type(file)
gsm_spec_label = []
csvreader = csv.reader(file)
header = []
header = next(csvreader)

gsm_spec_label = np.array(header)

gsm_spec = []
for row in csvreader:
        gsm_spec.append([i for i in row])

gsm_spec = np.array(gsm_spec)



# select feature
#0 phone_id
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
#feature_selected = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
feature_selected = [2,4,16,17,19,13,18]

print("\nFeature used:")
label_feature_selected = []
for x in range (len(feature_selected)):
    label_feature_selected.append(gsm_spec_label[feature_selected[x]])
    print(x+1, '|', str(gsm_spec_label[feature_selected[x]]))

# normalize with min max scale
ori_gsm_spec = gsm_spec[:, [0] + feature_selected]
gsm_spec = preprocessing.minmax_scale(gsm_spec[:,feature_selected],axis=0)

# Extract features
#gsm_spec = gsm_spec[:, feature_selected]

#repeat for research
p = 0
q = 1

print("\nDimensions\tRoom Size\tCluster Created\tSilhouette Score\tDavis Bouldin Score\tCalinski-Harabasz Index")
report = ''

while p<10:
    p += 1
    q = 1
    while q<5:
        q += 1
        # Build SOM dimension
        som_m = p
        som_n = q
        som = SOM(m=som_m, n=som_n, dim=len(feature_selected), lr=1, max_iter = 10000, random_state=None)
        # print('\nDimensions =', som_m, 'x', som_n)
        # print('Room Size =', som_m*som_n)

        report = str(som_m) + 'x' + str(som_n)
        report = str(report) + '\t' + str(som_m*som_n)

        # Fit it to the data
        som.fit(gsm_spec)

        # Assign each datapoint to its predicted cluster
        predictions = som.predict(gsm_spec)
        predictions = np.array(predictions)

        # Add predicted cluster column to table for result export
        ori_gsm_spec = np.c_[ori_gsm_spec,predictions]
        #print('Cluster created =', max(predictions)+1)
        report = report + '\t' + str(max(predictions)+1)

        # Plot the results
        # for d in range (len(feature_selected)):
        #     y = predictions[:]
        # # y = gsm_spec[:,d-1]
        #     x = gsm_spec[:,d]
        #     colors = ['red', 'green', 'blue', 'lime', 'yellow','grey']
        #     fig, ax = plt.subplots()
        #     ax.scatter(x, y, c=predictions, cmap=ListedColormap(colors))
        #     ax.title.set_text(gsm_spec_label[feature_selected[d]])
        #     figname = 'result' + str(gsm_spec_label[feature_selected[d]])
        #     plt.savefig(figname)
        #/////////////////////////

        # Export Result in CSV
        # with open('result.csv', 'w', newline='', encoding='utf-8') as file:
        #     writer = csv.writer(file, delimiter=',')
        #     writer.writerow(["phone_id"] + label_feature_selected + ["cluster"]) # write header
        #     writer.writerows(ori_gsm_spec)

        # Calculate Silhoutte Score
        from sklearn.metrics import silhouette_score
        score = silhouette_score(gsm_spec, predictions, metric='euclidean')
        #print('Silhouette Score: %.3f' % score)
        report = report + str('\t%.3f' % score)

        # Calculate Davis Bouldin Score
        from sklearn.metrics import davies_bouldin_score
        score = davies_bouldin_score(gsm_spec, predictions)
        #print('Davis Bouldin Score: %.3f' % score)
        report = report + str('\t%.3f' % score)

        # Calculate Calinski-Harabasz Index
        from sklearn import metrics
        from sklearn.metrics import pairwise_distances
        score = metrics.calinski_harabasz_score(gsm_spec, predictions)
        report = report + str('\t%.3f' % score)

        # # Calculate Homogeneity Score
        # from sklearn import metrics
        # score = metrics.homogeneity_score(gsm_spec, predictions)
        # print('Homogeneity Score: %.3f' % score)

        # End of process
        now = datetime.now()

        current_time = now.strftime("%H:%M:%S")
        #print("\nTime Finished = ", current_time)
        #rint("Have a great day!\n")
        print(report)

