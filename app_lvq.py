from sklvq import GLVQ

#insert SOM result
import loadsomresult as ls
file_location = "./somresult/result_2_6.csv"
data, labels, data_len = ls.loadsom(file_location,1)

data_train_ratio = 1
data_train_len = int(data_len*data_train_ratio)

data_train = data[:data_train_len]
labels_train = labels[:data_train_len]

user_mode = 0
if user_mode:
    # insert the data from user
    n_body_weight = float(input("\nInsert body weight (gr)\n"))
    n_display_size  = float(input("\nInsert disp size (inch)\n"))
    n_battery_capacity  = float(input("\nInsert battery capacity (mAh)\n"))
    n_total_cpu_speed  = float(input("\nInsert cpu speed (GHz total)\n"))
    n_ram  = float(input("\nInsert ram (GB)\n"))
    n_main_camera_resolution  = float(input("\nInsert mian cam res (MP)\n"))
    n_internal_memory  = float(input("\nInsert internal memory (GB)\n"))

#insert test program
# n_body_weight = float(800)
# n_display_size  = float(10)
# n_battery_capacity  = float(3000)
# n_total_cpu_speed  = float(10)
# n_ram  = float(2)
# n_main_camera_resolution  = float(8)
# n_internal_memory  = float(64)
print("\n Please wait, we calculate your input...\n")

# data_test = [146,5,300,8,2,13,64]
# data_test = [144,5,3020,16,2,8,32]
# data_test = [460,10,7250,22,6,13,128]
# data_test = [215,6,5000,23,8,48,128]
data_test = [450,10,7250,17,4,8,64]

if user_mode:
    data_test = [n_body_weight,n_display_size,n_battery_capacity,n_total_cpu_speed,n_ram,n_main_camera_resolution,n_internal_memory]

# Sklearn's standardscaler to perform z-transform
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

# Sklearn's MinMaxScaler to perform z-transform
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Compute (fit) and apply (transform) z-transform
data_train2 = data_train
data_train = scaler.fit_transform(data_train)

# Scaling with data test
data_train2.append(data_test)
data_train2 = scaler.fit_transform(data_train2)
# Pick scaled data test
pick_data_test = data_train2[-1]
data_test_scaled = []
data_test_scaled.append(pick_data_test)
# print(data_test_scaled)

# The creation of the model object used to fit the data to.
model = GLVQ(
    distance_type="squared-euclidean",
    activation_type="swish",
    activation_params={"beta": 2},
    solver_type="steepest-gradient-descent",
    solver_params={"max_runs": 20, "step_size": 0.9},
)

# Train the model
model.fit(data_train, labels_train)
# print("Finished. (%.5f seconds)" % (time.time() - start_time))

# Predict the labels using the trained model
predicted_labels = model.predict(data_test_scaled)
predicted_labels = int(predicted_labels)

print("\nYour predicted class is "+str(predicted_labels))

print("\nThis is the phone in the class "+str(predicted_labels)+" :")
from csvkit.utilities.csvsql import CSVSQL
qcode = 'select * from result_2_6 where cluster="'+str(predicted_labels)+'"'
args = ['--query',qcode,'somresult/result_2_6.csv']
result  = CSVSQL(args)
print(result.main())

if user_mode:
    input("\nPress enter to exit\n")

exit()