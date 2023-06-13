import time
total_start_time = time.time()

import array
import csv
import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklvq import GLVQ

matplotlib.rc("xtick", labelsize="small")
matplotlib.rc("ytick", labelsize="small")

# from sklearn.datasets import load_iris
# Contains also the target_names and feature_names, which we will use for the plots.
# iris = load_iris()

# data = iris.data
# labels = iris.target

p = 2
q = 6
start_time = time.time()
print("Dimension\tAccuracy\tF1 Score")
print("Cluster\tDataTrainRatio\tAccuracy\tF1 Score")
report_buff = []
import loadsomresult as ls
file_location = "./somresult/result_"+str(p)+"_"+str(q)+".csv"
data, labels, data_len = ls.loadsom(file_location,1)

ratio = 0
while(ratio<0.9):
    ratio = ratio + 0.1
    data_train_ratio = 0.5
    data_train_len = int(data_len*data_train_ratio)

    data_train = data[:data_train_len]
    labels_train = labels[:data_train_len]

    data_test = data[data_train_len:]
    labels_test = labels[data_train_len:]
    # data_test = [[207.0, 6.0, 2500.0, 4.0, 1.0, 13.0]]
    data_test_len = len(data_test)

    # print(data_train)
    # print("Data training size = "+str(len(data_train))+"/"+str(data_train_len*100/data_len)+"%")
    # print(len(labels_train))
    # print("Data test size     = "+str(data_test_len)+"/"+str(data_test_len*100/data_len)+"%")
    # print(len(labels_test))
    # print("Total data = "+str(data_train_len+data_test_len)+"/"+str(data_len))
    # data = [[1,2,3,4],[4,3,2,1],[1,1,1,1]]
    # data2 = [[4,3,1,1],[1,1,1,1],[1,2,3,4],[1,1,1,1],[1,1,1,1],[1,2,3,4]]
    # labels = [1,2,3]

    # Sklearn's standardscaler to perform z-transform
    # scaler = StandardScaler()
    scaler = MinMaxScaler()

    # Compute (fit) and apply (transform) z-transform
    data_train = scaler.fit_transform(data_train)
    data_test = scaler.fit_transform(data_test)
    # print(data_test)
    # The creation of the model object used to fit the data to.
    model = GLVQ(
        distance_type="squared-euclidean",
        activation_type="swish",
        activation_params={"beta": 2},
        solver_type="steepest-gradient-descent",
        solver_params={"max_runs": 20, "step_size": ratio},
    )

    # Train the model
    # start_time = time.time()
    # print("Training data...")
    model.fit(data_train, labels_train)
    # print("Finished. (%.5f seconds)" % (time.time() - start_time))

    # Predict the labels using the trained model
    # start_time = time.time()
    # print("Predicting...")
    predicted_labels = model.predict(data_test)
    # print("Finished. (%f seconds)" % (time.time() - start_time))
    #predicted_labels = model.predict([[1,2,3,4],[4,3,2,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])

    # print(labels_test)
    # print(predicted_labels)

    # To get a sense of the training performance we could print the classification report.
    # print(classification_report(labels_test, predicted_labels))
    # print(classification_report(predicted_labels, labels_test))

    # Calculate accuracy
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(labels_test, predicted_labels)
    # print('Accuracy: %f' % accuracy)

    # Calculate MSE
    # from sklearn.metrics import mean_squared_error
    # mse = mean_squared_error(labels_test,predicted_labels)

    # Calculate F1-Score
    from sklearn.metrics import f1_score
    f1s = f1_score(labels_test,predicted_labels, average='macro')

    report_buff.append([4,ratio,accuracy,f1s])
    print(str(6)+"\t"+str(ratio)+"\t"+str(accuracy)+"\t"+str(f1s))


print("Finished. (%s seconds)" % (time.time() - start_time))

# Export Result in CSV
with open('./report/lvq_result.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(["p"] + ["q"] + ["Accuracy"] + ["F1 Score"]) # write header
    writer.writerows(report_buff)
# colors = ["blue", "red", "green"]
# num_prototypes = model.prototypes_.shape[0]
# num_features = model.prototypes_.shape[1]

# fig, ax = plt.subplots(num_prototypes, 1)
# fig.suptitle("Prototype of each class")

# for i, prototype in enumerate(model.prototypes_):
#     # Reverse the z-transform to go back to the original feature space.
#     prototype = scaler.inverse_transform(prototype)

#     ax[i].bar(
#         range(num_features),
#         prototype,
#         color=colors[i],
#         label=iris.target_names[model.prototypes_labels_[i]],
#     )
#     ax[i].set_xticks(range(num_features))
#     if i == (num_prototypes - 1):
#         ax[i].set_xticklabels([name[:-5] for name in iris.feature_names])
#     else:
#         ax[i].set_xticklabels([], visible=False)
#         ax[i].tick_params(
#             axis="x", which="both", bottom=False, top=False, labelbottom=False
#         )
#     ax[i].set_ylabel("cm")
#     ax[i].legend()

# figname = 'resultlvq'
# plt.savefig(figname)