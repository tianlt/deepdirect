from deepdirect.mutator import predict
from deepdirect.model import build_aa_mutator, encode_seq_int, decode_seq_int
from tensorflow.keras import backend as K
import deepdirect
import pickle
import numpy as np

# List of file names
file_names = ['pre.pkl', 'rbd.pkl', 'same.pkl', 'x.pkl', 'y.pkl', 'z.pkl', 'noise.pkl']

# Dictionary to store loaded data
input_data = []

# Load data from each pickle file
for file_name in file_names:
    with open("./test_data/" + file_name, 'rb') as file:
        input_data.append(pickle.load(file)) 

# from deepdirect import weights_i, weights_d

# relative_path_to_weight = '/weights/model_i_weights.h5'
# weight_file_path = pkg_resources.resource_filename('deepdirect', relative_path_to_weight)
aa_mutator = build_aa_mutator()
aa_mutator.load_weights(deepdirect.weights_i)
predict_result = aa_mutator.predict(input_data)
print(predict_result)

r_pre = decode_seq_int(K.argmax(predict_result, axis=-1).numpy()[0])
seq_pre = ''.join(np.array(r_pre))
print(r_pre)