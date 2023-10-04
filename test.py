from deepdirect.mutator import predict

import pickle

# List of file names
file_names = ['pre.pkl', 'rbd.pkl', 'same.pkl', 'x.pkl', 'y.pkl', 'z.pkl', 'noise.pkl']

# Dictionary to store loaded data
input_data = []

# Load data from each pickle file
for file_name in file_names:
    with open("./test_data/" + file_name, 'rb') as file:
        input_data.append(pickle.load(file)) 

predict_result = predict(input_data)
print(predict_result)
