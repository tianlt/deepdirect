from deepdirect.model import build_aa_mutator, decode_seq_int
from tensorflow.keras import backend as K
import deepdirect
import pickle
import numpy as np

# List of file names
file_names = ['pre.pkl', 'rbd.pkl', 'same.pkl', 'x.pkl', 'y.pkl', 'z.pkl', 'noise.pkl']

input_data = []
# Load data from each pickle file
for file_name in file_names:
    with open("./test_data/" + file_name, 'rb') as file:
        input_data.append(pickle.load(file)) 

aa_mutator = build_aa_mutator()
aa_mutator.load_weights(deepdirect.weights_i)
predict_result = aa_mutator.predict(input_data)
# print(predict_result)

r_pre = decode_seq_int(K.argmax(predict_result, axis=-1).numpy()[0])
seq_pre = ''.join(np.array(r_pre))
print(seq_pre)

###########################################################
aa_mutator2 = build_aa_mutator()
aa_mutator2.load_weights(deepdirect.weights_d)
predict_result2 = aa_mutator2.predict(input_data)
# print(predict_result)

r_pre2 = decode_seq_int(K.argmax(predict_result2, axis=-1).numpy()[0])
seq_pre2 = ''.join(np.array(r_pre))
print(seq_pre2)