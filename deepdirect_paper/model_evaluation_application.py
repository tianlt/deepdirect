# %%
import re
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import os
import tensorflow as tf
from collections import OrderedDict
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import time
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K
import pickle
from tqdm import tqdm
import numpy as np
import model_function
import seaborn as sns
from biopandas.pdb import PandasPdb
from tensorflow.keras.callbacks import EarlyStopping
from model_function import build_model, build_aa_mutator, AA_Mutation_GAN, generator_loss, discriminator_loss, \
    discriminator_loss_replace, generator_loss_replace, build_discriminator_replace_model
from ab_bind_data_extract import find_rbd
import statistics as st
from Bio import SeqIO

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# %%
random.seed(1)
aa_string = 'ARNDCEQGHILKMFPSTWYV'


def encode_seq_int(data='HELLOWORLD'):
    char_to_int = dict((c, i + 1) for i, c in enumerate(aa_string))
    integer_encoded = [char_to_int[char] for char in data]
    return integer_encoded


#######################################
def decode_seq_int(data='HELLOWORLD'):
    int_to_char = dict((i + 1, c) for i, c in enumerate(aa_string))
    # tf.one_hot(20,20) == 0, arg [max,...,0] or arg [0,...,0] == 0
    character_encoded = [int_to_char[int + 20] if int == 0 else int_to_char[int] for int in data]
    return character_encoded


def summarize_data(data):
    mean = st.mean(data)
    median = st.median(data)
    stdev = st.stdev([i * 10 for i in data]) / 10
    summary_dict = {
        'mean': mean,
        'median': median,
        'stdev': stdev}
    return summary_dict


def random_mutate(aa_pre_num, ratio):
    seq_len = len(aa_pre_num)
    mut_pos = random.sample(range(0, seq_len), int(seq_len * ratio))
    aa_aft_rd_num = aa_pre_num.copy()
    for index in mut_pos:
        aa_aft_rd_num[index] = random.randint(1, len(aa_string))
    return aa_aft_rd_num


def deep_direct_mutate(aa_mutator_final, input_pre, input_rbd, input_same, input_x, input_y, input_z, num=500):
    aa_mutator_final = aa_mutator_final
    pre = tf.repeat(input_pre, repeats=num, axis=0)
    rbd = tf.repeat(input_rbd, repeats=num, axis=0)
    same = tf.repeat(input_same, repeats=num, axis=0)
    x = tf.repeat(input_x, repeats=num, axis=0)
    y = tf.repeat(input_y, repeats=num, axis=0)
    z = tf.repeat(input_z, repeats=num, axis=0)
    input_noi = tf.random.normal(shape=(num, input_pre.shape[1], 1))
    input_noi = tf.cast(input_noi, tf.float32)

    out = aa_mutator_final.predict([pre, rbd, same, x, y, z, input_noi])
    out_num = K.argmax(out, axis=-1).numpy()

    out_one_hot = tf.one_hot(out_num, 20)
    bc = binding_affinity_predictor.predict([pre, out_one_hot, rbd, same, x, y, z])

    bc_list = bc.flatten()
    similarity_ratio_list = (K.sum(K.cast(K.argmax(pre, axis=-1) == out_num, tf.int64),
                                   axis=-1) / out_num.shape[1]).numpy()
    return out_num, bc_list, similarity_ratio_list


def write_fasta(data, name, file_path):
    data = ''.join(data)
    dict = {name: data}
    output_file = open(file_path, 'w')
    for i, j in dict.items():
        identifier = '>' + i + '\n'
        output_file.write(identifier)
        sequence = data + '\n'
        output_file.write(sequence)
    output_file.close()


def read_fasta(file_path):
    fasta_sequences = SeqIO.parse(open(file_path), 'fasta')
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
    return sequence


def mutation_summary(ori_seq, aft_seq, chain, residue_number_index):
    i = -1
    i_list = []
    pre_list = []
    aft_list = []
    for x, y in zip(ori_seq, aft_seq):
        i = i + 1
        if x != y:
            pre_list.append(x)
            aft_list.append(y)
            i_list.append(residue_number_index[i])
    mutation_df_dict = {'mutation_residue_num': i_list,
                        'mutation_aa_pre': pre_list,
                        'mutation_aa_aft': aft_list,
                        'mutation_chain': [chain] * len(aft_list)}
    return pd.DataFrame(mutation_df_dict)


def extract_residue_number(data):
    tmp = data.df['ATOM']
    cmp = 'placeholder'
    indices = []

    residue_number_insertion = (tmp['residue_number'].astype(str)
                                + tmp['insertion'])

    for num, ind in zip(residue_number_insertion, np.arange(tmp.shape[0])):
        if num != cmp:
            indices.append(ind)
        cmp = num

    result = tmp.iloc[indices]['residue_number']
    return result


########################################


ppdb_1r42 = PandasPdb().read_pdb('application/1r42.pdb')
residue_seq_1r42 = ppdb_1r42.amino3to1()
residue_name_1r42 = ''.join(residue_seq_1r42['residue_name'].tolist())
residue_name_1r42 = residue_name_1r42.replace('?', '')
pdb_name = ['model_1']
pre_dict = {}
pre_all_result_dict = {}
for i in pdb_name:
    ppdb = PandasPdb().read_pdb('application/result/models_novavax_pre/' + i + '.pdb')
    residue_seq = ppdb.amino3to1()  # chain a IR42, chain b c 7JJI
    ########################################################################
    # residue_name = ''.join(residue_seq['residue_name'].tolist())
    # residue_name.find(residue_name_1r42)
    ppdb_1r42_index = ppdb.amino3to1()['chain_id'][:len(residue_name_1r42) + 1].index[-1] - 1
    ppdb.df['ATOM'].at[:ppdb_1r42_index, ['chain_id']] = 'D'
    ppdb.to_pdb(path='application/result/models_novavax_pre/model_1_modified.pdb')

    #########################################################################
    residue_seq = ppdb.amino3to1()
    alpha_carbon_coordinate = ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].iloc[residue_seq.index + 1].to_numpy()
    same_index = [0 if aa == 'D' else 1 for aa in residue_seq['chain_id']]
    chain_index = residue_seq['chain_id'].tolist()
    rbd_index = find_rbd(alpha_carbon_coordinate, 50, same_index, 0.1)  # 50
    residue_num = ppdb.df['ATOM'][['residue_number']].iloc[residue_seq.index + 1].to_numpy()


# np.savetxt('application/result/rbd_index.csv', rbd_index, delimiter =", ",
#            fmt ='% s')
seq = ''.join(residue_seq['residue_name'])
with open("application/result/seq.txt", "w") as f:
    f.write(seq)

len(seq)
len(rbd_index)
# %% model assembly

aa_mutator_final_3 = build_aa_mutator()
aa_mutator_final_increase = build_aa_mutator()
aa_mutator_final_3.load_weights(
    'models/model_i_weights.h5')
aa_mutator_final_increase.load_weights(
    'models/model_d_weights.h5')

binding_affinity_predictor = build_model(activation='linear', f_num_1=32, f_num_2=64, f_num_3=128, f_num_4=256)
binding_affinity_predictor.load_weights(
    'models/binding_affinity_predictor_weights.h5')
# data formatting
input_pre = np.array(encode_seq_int(''.join(residue_seq['residue_name'])))
input_pre = np.reshape(input_pre, (-1, len(input_pre)))
input_pre = tf.one_hot(input_pre, 20)

input_rbd = np.reshape(rbd_index, (-1, len(rbd_index), 1))
input_same = np.reshape(same_index, (-1, len(same_index), 1))

input_x = alpha_carbon_coordinate[:, 0]
input_y = alpha_carbon_coordinate[:, 1]
input_z = alpha_carbon_coordinate[:, 2]

input_x = np.reshape(input_x, (-1, len(input_x), 1))
input_y = np.reshape(input_y, (-1, len(input_y), 1))
input_z = np.reshape(input_z, (-1, len(input_z), 1))
# normalize input
input_x = keras.utils.normalize(input_x, axis=1)
input_y = keras.utils.normalize(input_y, axis=1)
input_z = keras.utils.normalize(input_z, axis=1)

input_pre = tf.cast(input_pre, tf.float32)
input_rbd = tf.cast(input_rbd, tf.float32)
input_same = tf.cast(input_same, tf.float32)
input_x = tf.cast(input_x, tf.float32)
input_y = tf.cast(input_y, tf.float32)
input_z = tf.cast(input_z, tf.float32)

# %% deepdirect mutation
# random.seed(1)

deep_out_num_3, deep_bc_list_3, deep_similarity_ratio_list_3 = deep_direct_mutate(aa_mutator_final_3, input_pre,
                                                                                  input_rbd, input_same, input_x,
                                                                                  input_y, input_z, num=500)
deep_result_3 = {'deep_out_num': deep_out_num_3, 'deep_bc_list': deep_bc_list_3,
                 'deep_similarity_ratio_list': deep_similarity_ratio_list_3}
# with open('application/deep_result_3-bc-{:.3f}-ratio-{:.3f}.pkl'.format(summarize_data(deep_bc_list_3)['mean'],
#                                                                         summarize_data(deep_similarity_ratio_list_3)[
#                                                                           'mean']), 'wb') as f:
#     pickle.dump(deep_result_3, f)
print(summarize_data(deep_bc_list_3)['mean'])
print(summarize_data(deep_similarity_ratio_list_3)['mean'])

# increase
deep_out_num_increase, deep_bc_list_increase, deep_similarity_ratio_list_increase = deep_direct_mutate(aa_mutator_final_increase, input_pre,
                                                                                  input_rbd, input_same, input_x,
                                                                                  input_y, input_z, num=500)
deep_result_increase = {'deep_out_num': deep_out_num_increase, 'deep_bc_list': deep_bc_list_increase,
                 'deep_similarity_ratio_list': deep_similarity_ratio_list_increase}
# with open('application/deep_result_increase-bc-{:.3f}-ratio-{:.3f}.pkl'.format(summarize_data(deep_bc_list_increase)['mean'],
#                                                                         summarize_data(deep_similarity_ratio_list_increase)[
#                                                                           'mean']), 'wb') as f:
#     pickle.dump(deep_result_increase, f)

# with open('application/deep_result_increase-bc-5.352-ratio-0.719.pkl', 'rb') as f:
#     deep_result_increase = pickle.load(f)
print(summarize_data(deep_result_increase['deep_bc_list']))
print(summarize_data(deep_result_increase['deep_similarity_ratio_list'])['mean'])
# {'mean': 5.35245, 'median': 5.311718940734863, 'stdev': 0.4641916032715311}
# 0.7188012232415902
# %% random mutation
random.seed(1)
aa_pre = decode_seq_int(K.argmax(input_pre, axis=-1).numpy()[0])
aa_pre_num = encode_seq_int(aa_pre)

bc_rd_list = []
aa_aft_rd_num_list = []
t_range = tqdm(range(500))
for i in t_range:
    aa_aft_rd_num = random_mutate(aa_pre_num, 0.2)
    input_aa_aft_rd = np.reshape(aa_aft_rd_num, (-1, len(aa_aft_rd_num)))
    input_aa_aft_rd = tf.one_hot(input_aa_aft_rd, 20)
    input_aa_aft_rd = tf.cast(input_aa_aft_rd, tf.float32)
    aa_aft_rd_num_list.append(K.argmax(input_aa_aft_rd, axis=-1))
    input_aa_pre = np.reshape(aa_pre_num, (-1, len(aa_pre_num)))
    input_aa_pre = tf.one_hot(input_aa_pre, 20)
    input_aa_pre = tf.cast(input_aa_pre, tf.float32)

    bc = binding_affinity_predictor.predict(
        [input_aa_pre, input_aa_aft_rd, input_rbd, input_same, input_x, input_y, input_z])
    bc_rd_list.append(bc[0][0])

index = [1 if i <= 0 else 0 for i in bc_rd_list]
bc_rd_sub_list = [bc_rd_list[i] for i, j in enumerate(index) if j == 1]
bc_rd_sub_list_increase = [bc_rd_list[i] for i, j in enumerate(index) if j == 0]

# summarize
print('rd')
print(summarize_data(bc_rd_list))
print(summarize_data(bc_rd_sub_list))
print(summarize_data(bc_rd_sub_list_increase))

print('deep_3')
print(summarize_data(deep_bc_list_3))
print(summarize_data(deep_similarity_ratio_list_3))

# rd
# {'mean': 0.057169415, 'median': 0.00565633550286293, 'stdev': 0.4238367658017542}
# {'mean': -0.21783236, 'median': -0.15472634, 'stdev': 0.22646737477320616}
# {'mean': 0.3256494, 'median': 0.18739673, 'stdev': 0.3989381278825612}

# deep_3
# {'mean': -2.5013373, 'median': -2.5145487785339355, 'stdev': 0.4439167980458013}
# {'mean': 0.8756941896024465, 'median': 0.8756371049949032, 'stdev': 0.011168929676096477}

# save
# with open('application/bc_rd_list.pkl', 'wb') as f:
#     pickle.dump(bc_rd_list, f)
# with open('application/aa_aft_rd_num_list.pkl', 'wb') as f:
#     pickle.dump(aa_aft_rd_num_list, f)
# with open('application/bc_rd_sub_list.pkl', 'wb') as f:
#     pickle.dump(bc_rd_sub_list, f)


# with open('application/deep_bc_list_1.pkl', 'wb') as f:
#     pickle.dump(deep_bc_list_1, f)
# with open('application/deep_out_num_1.pkl', 'wb') as f:
#     pickle.dump(deep_out_num_1, f)

#
# load
# with open('application/bc_rd_list.pkl', 'rb') as f:
#     bc_rd_list = pickle.load(f)
# with open('application/bc_rd_sub_list.pkl', 'rb') as f:
#     bc_rd_sub_list = pickle.load(f)
# with open('application/bc_list.pkl', 'rb') as f:
#     bc_list = pickle.load(f)
# with open('application/out_num_list.pkl', 'rb') as f:
#     out_num_list = pickle.load(f)
# with open('application/aa_aft_rd_num_list.pkl', 'rb') as f:
#     aa_aft_rd_num_list = pickle.load(f)
#
#
bc_list = deep_bc_list_3
out_num_list = deep_out_num_3
bc_list_order = [bc_list[j] for i, j in sorted(enumerate(np.argsort(bc_list)))]
bc_rd_list_order = [bc_rd_list[j] for i, j in sorted(enumerate(np.argsort(bc_rd_list)))]
out_num_list_order = [out_num_list[j] for i, j in sorted(enumerate(np.argsort(bc_list)))]
aa_aft_rd_num_list_order = [aa_aft_rd_num_list[j] for i, j in sorted(enumerate(np.argsort(bc_rd_list)))]
deep_out_seq1 = decode_seq_int(out_num_list_order[0])  # deepdirect
rd_out_seq1 = decode_seq_int(aa_aft_rd_num_list_order[0].numpy()[0])  # random mutation

# mutated sequence info
out_one_hot = tf.reshape(out_num_list_order[0], (-1, out_num_list_order[0].shape[0]))
mut_position_summary = K.cast(K.argmax(input_pre, axis=-1) == out_one_hot, tf.int64)
where = tf.where(tf.equal(mut_position_summary, 0))

input_pre_summary = K.argmax(input_pre, axis=-1)
mutation_aa_pre = tf.gather_nd(input_pre_summary, where).numpy()
mutation_aa_aft = tf.gather_nd(out_one_hot, where).numpy()
chains = np.reshape(chain_index, (-1, len(chain_index)))
chains_summary = tf.gather_nd(chains, where).numpy().astype("U1")
residue_num_summary = tf.gather_nd(tf.reshape(residue_num, (1, -1)), where).numpy()
mutation_df_dict = {'mutation_aa_pre_num': mutation_aa_pre,
                    'mutation_aa_pre': decode_seq_int(mutation_aa_pre),
                    'mutation_aa_aft_num': mutation_aa_aft,
                    'mutation_aa_aft': decode_seq_int(mutation_aa_aft),
                    'mutation_chains': chains_summary,
                    'mutation_residue_num': residue_num_summary}
mutation_df = pd.DataFrame(mutation_df_dict)

# output fasta
deep_out_IR42_seq1 = deep_out_seq1[0:len(residue_name_1r42)]
deep_out_7JII_seq1 = deep_out_seq1[len(residue_name_1r42):]
deep_out_7JII_A_seq1 = deep_out_7JII_seq1[0:chain_index.count('A')]
deep_out_7JII_B_seq1 = deep_out_7JII_seq1[chain_index.count('A'):chain_index.count('A') + chain_index.count('B')]
deep_out_7JII_C_seq1 = deep_out_7JII_seq1[
                       chain_index.count('A') + chain_index.count('B'):chain_index.count('A') + chain_index.count(
                           'B') + chain_index.count('C')]
write_fasta(deep_out_IR42_seq1, 'deep_out_IR42_seq1', 'application/result/deep_out_IR42_seq1.fasta')
write_fasta(deep_out_7JII_A_seq1, 'deep_out_7JII_A_seq1', 'application/result/deep_out_7JII_A_seq1.fasta')
write_fasta(deep_out_7JII_B_seq1, 'deep_out_7JII_B_seq1', 'application/result/deep_out_7JII_B_seq1.fasta')
write_fasta(deep_out_7JII_C_seq1, 'deep_out_7JII_C_seq1', 'application/result/deep_out_7JII_C_seq1.fasta')

rd_out_IR42_seq1 = rd_out_seq1[0:len(residue_name_1r42)]
rd_out_7JII_seq1 = rd_out_seq1[len(residue_name_1r42):]
rd_out_7JII_A_seq1 = rd_out_7JII_seq1[0:chain_index.count('A')]
rd_out_7JII_B_seq1 = rd_out_7JII_seq1[chain_index.count('A'):chain_index.count('A') + chain_index.count('B')]
rd_out_7JII_C_seq1 = rd_out_7JII_seq1[
                     chain_index.count('A') + chain_index.count('B'):chain_index.count('A') + chain_index.count(
                         'B') + chain_index.count('C')]

write_fasta(rd_out_IR42_seq1, 'rd_out_IR42_seq1', 'application/result/rd_out_IR42_seq1.fasta')
write_fasta(rd_out_7JII_A_seq1, 'rd_out_7JII_A_seq1', 'application/result/rd_out_7JII_A_seq1.fasta')
write_fasta(rd_out_7JII_B_seq1, 'rd_out_7JII_B_seq1', 'application/result/rd_out_7JII_B_seq1.fasta')
write_fasta(rd_out_7JII_C_seq1, 'rd_out_7JII_C_seq1', 'application/result/rd_out_7JII_C_seq1.fasta')

seq_7JII_ori = ''.join(aa_pre)[len(residue_name_1r42):chain_index.count('A') + len(residue_name_1r42)]
write_fasta(seq_7JII_ori, 'seq_7JII_ori', 'application/result/seq_7JII_ori.fasta')

# %% find difference

seq_7JII_A = read_fasta('application/result/deep_out_7JII_A_seq1.fasta')
seq_7JII_B = read_fasta('application/result/deep_out_7JII_B_seq1.fasta')
seq_7JII_C = read_fasta('application/result/deep_out_7JII_C_seq1.fasta')
seq_7JII_ori = read_fasta('application/result/seq_7JII_ori.fasta')

residue_number = extract_residue_number(ppdb).to_numpy()
residue_number_A = residue_number[len(residue_name_1r42):chain_index.count('A') + len(residue_name_1r42)]
residue_number_B = residue_number[
                   chain_index.count('A') + len(residue_name_1r42):chain_index.count('B') + chain_index.count(
                       'A') + len(residue_name_1r42)]
residue_number_C = residue_number[chain_index.count('B') + chain_index.count('A') + len(residue_name_1r42):]


def mutation_summary(ori_seq, aft_seq, chain, residue_number_index):
    i = -1
    i_list = []
    pre_list = []
    aft_list = []
    for x, y in zip(ori_seq, aft_seq):
        i = i + 1
        if x != y:
            pre_list.append(x)
            aft_list.append(y)
            i_list.append(residue_number_index[i])
    mutation_df_dict = {'mutation_residue_num': i_list,
                        'mutation_aa_pre': pre_list,
                        'mutation_aa_aft': aft_list,
                        'mutation_chain': [chain] * len(aft_list)}
    return pd.DataFrame(mutation_df_dict)


a_df = mutation_summary(seq_7JII_ori, seq_7JII_A, 'A', residue_number_A)
b_df = mutation_summary(seq_7JII_ori, seq_7JII_B, 'B', residue_number_B)
c_df = mutation_summary(seq_7JII_ori, seq_7JII_C, 'C', residue_number_C)

mutation_df_7JII = pd.concat([a_df, b_df, c_df])
mutation_df_7JII.to_csv('application/result/mutation_df_7JII.csv')

#%%#########################################################################################
# subchain evaluation
with open('application/aa_aft_rd_num_list.pkl', 'rb') as f:
    aa_aft_rd_num_list = pickle.load(f)
with open('application/out_num_list.pkl', 'rb') as f:
    out_num_list = pickle.load(f)
pre_num = K.argmax(input_pre, axis=-1)
chain_d_ori = pre_num[0][0:len(residue_name_1r42)]
chain_abc_ori = pre_num[0][len(residue_name_1r42):]

chain_a_ori = chain_abc_ori[0:chain_index.count('A')]
chain_b_ori = chain_abc_ori[chain_index.count('A'):chain_index.count('A') + chain_index.count('B')]
chain_c_ori = chain_abc_ori[chain_index.count('A') + chain_index.count('B'):chain_index.count('A') + chain_index.count(
    'B') + chain_index.count('C')]


def extract_chain(data, chain):
    chain_abc = data[0][len(residue_name_1r42):]
    if chain == 'd':
        chain = data[0][0:len(residue_name_1r42)]
    elif chain == 'a':
        chain = chain_abc[0:chain_index.count('A')]
    elif chain == 'b':
        chain = chain_abc[chain_index.count('A'):chain_index.count('A') + chain_index.count('B')]
    elif chain == 'c':
        chain = chain_abc[chain_index.count('A') + chain_index.count('B'):chain_index.count('A') + chain_index.count(
            'B') + chain_index.count('C')]
    return chain


mut_d = [extract_chain(i, chain='d') for i in out_num_list]
mut_a = [extract_chain(i, chain='a') for i in out_num_list]
mut_b = [extract_chain(i, chain='b') for i in out_num_list]
mut_c = [extract_chain(i, chain='c') for i in out_num_list]

mut_d_rd = [extract_chain(i, chain='d') for i in aa_aft_rd_num_list]
mut_a_rd = [extract_chain(i, chain='a') for i in aa_aft_rd_num_list]
mut_b_rd = [extract_chain(i, chain='b') for i in aa_aft_rd_num_list]
mut_c_rd = [extract_chain(i, chain='c') for i in aa_aft_rd_num_list]


out_num_increase_list = [tf.convert_to_tensor([i]) for i in deep_result_increase['deep_out_num']]
mut_d_increase = [extract_chain(i, chain='d') for i in out_num_increase_list]
mut_a_increase = [extract_chain(i, chain='a') for i in out_num_increase_list]
mut_b_increase = [extract_chain(i, chain='b') for i in out_num_increase_list]
mut_c_increase = [extract_chain(i, chain='c') for i in out_num_increase_list]


def duplicate(testList, n):
    return [ele for ele in testList for _ in range(n)]


chain_d_ori = duplicate([chain_d_ori], len(mut_d))
chain_a_ori = duplicate([chain_a_ori], len(mut_d))
chain_b_ori = duplicate([chain_b_ori], len(mut_d))
chain_c_ori = duplicate([chain_c_ori], len(mut_d))
mut_d_whole_seq = [i.numpy().tolist() + j.numpy().tolist() + k.numpy().tolist() + l.numpy().tolist() for i, j, k, l in
                   zip(mut_d, chain_a_ori, chain_b_ori, chain_c_ori)]
mut_a_whole_seq = [i.numpy().tolist() + j.numpy().tolist() + k.numpy().tolist() + l.numpy().tolist() for i, j, k, l in
                   zip(mut_a, chain_d_ori, chain_b_ori, chain_c_ori)]
mut_b_whole_seq = [i.numpy().tolist() + j.numpy().tolist() + k.numpy().tolist() + l.numpy().tolist() for i, j, k, l in
                   zip(mut_b, chain_a_ori, chain_d_ori, chain_c_ori)]
mut_c_whole_seq = [i.numpy().tolist() + j.numpy().tolist() + k.numpy().tolist() + l.numpy().tolist() for i, j, k, l in
                   zip(mut_c, chain_a_ori, chain_b_ori, chain_d_ori)]

mut_d_whole_seq_increase = [i.numpy().tolist() + j.numpy().tolist() + k.numpy().tolist() + l.numpy().tolist() for i, j, k, l in
                   zip(mut_d_increase, chain_a_ori, chain_b_ori, chain_c_ori)]
mut_a_whole_seq_increase = [i.numpy().tolist() + j.numpy().tolist() + k.numpy().tolist() + l.numpy().tolist() for i, j, k, l in
                   zip(mut_a_increase, chain_d_ori, chain_b_ori, chain_c_ori)]
mut_b_whole_seq_increase = [i.numpy().tolist() + j.numpy().tolist() + k.numpy().tolist() + l.numpy().tolist() for i, j, k, l in
                   zip(mut_b_increase, chain_a_ori, chain_d_ori, chain_c_ori)]
mut_c_whole_seq_increase = [i.numpy().tolist() + j.numpy().tolist() + k.numpy().tolist() + l.numpy().tolist() for i, j, k, l in
                   zip(mut_c_increase, chain_a_ori, chain_b_ori, chain_d_ori)]

mut_d_whole_seq_rd = [i.numpy().tolist() + j.numpy().tolist() + k.numpy().tolist() + l.numpy().tolist() for i, j, k, l
                      in zip(mut_d_rd, chain_a_ori, chain_b_ori, chain_c_ori)]
mut_a_whole_seq_rd = [i.numpy().tolist() + j.numpy().tolist() + k.numpy().tolist() + l.numpy().tolist() for i, j, k, l
                      in zip(mut_a_rd, chain_d_ori, chain_b_ori, chain_c_ori)]
mut_b_whole_seq_rd = [i.numpy().tolist() + j.numpy().tolist() + k.numpy().tolist() + l.numpy().tolist() for i, j, k, l
                      in zip(mut_b_rd, chain_a_ori, chain_d_ori, chain_c_ori)]
mut_c_whole_seq_rd = [i.numpy().tolist() + j.numpy().tolist() + k.numpy().tolist() + l.numpy().tolist() for i, j, k, l
                      in zip(mut_c_rd, chain_a_ori, chain_b_ori, chain_d_ori)]

mut_d_whole_seq_one_hot = tf.one_hot(mut_d_whole_seq, 20)
mut_a_whole_seq_one_hot = tf.one_hot(mut_a_whole_seq, 20)
mut_b_whole_seq_one_hot = tf.one_hot(mut_b_whole_seq, 20)
mut_c_whole_seq_one_hot = tf.one_hot(mut_c_whole_seq, 20)

mut_d_whole_seq_one_hot_increase = tf.one_hot(mut_d_whole_seq_increase, 20)
mut_a_whole_seq_one_hot_increase = tf.one_hot(mut_a_whole_seq_increase, 20)
mut_b_whole_seq_one_hot_increase = tf.one_hot(mut_b_whole_seq_increase, 20)
mut_c_whole_seq_one_hot_increase = tf.one_hot(mut_c_whole_seq_increase, 20)

mut_d_whole_seq_one_hot_rd = tf.one_hot(mut_d_whole_seq_rd, 20)
mut_a_whole_seq_one_hot_rd = tf.one_hot(mut_a_whole_seq_rd, 20)
mut_b_whole_seq_one_hot_rd = tf.one_hot(mut_b_whole_seq_rd, 20)
mut_c_whole_seq_one_hot_rd = tf.one_hot(mut_c_whole_seq_rd, 20)

pre = tf.repeat(input_pre, repeats=len(mut_d), axis=0)
rbd = tf.repeat(input_rbd, repeats=len(mut_d), axis=0)
same = tf.repeat(input_same, repeats=len(mut_d), axis=0)
x = tf.repeat(input_x, repeats=len(mut_d), axis=0)
y = tf.repeat(input_y, repeats=len(mut_d), axis=0)
z = tf.repeat(input_z, repeats=len(mut_d), axis=0)

bc_d = binding_affinity_predictor.predict([pre, mut_d_whole_seq_one_hot, rbd, same, x, y, z])
bc_a = binding_affinity_predictor.predict([pre, mut_a_whole_seq_one_hot, rbd, same, x, y, z])
bc_b = binding_affinity_predictor.predict([pre, mut_b_whole_seq_one_hot, rbd, same, x, y, z])
bc_c = binding_affinity_predictor.predict([pre, mut_c_whole_seq_one_hot, rbd, same, x, y, z])

bc_d_increase = binding_affinity_predictor.predict([pre, mut_d_whole_seq_one_hot_increase, rbd, same, x, y, z])
bc_a_increase = binding_affinity_predictor.predict([pre, mut_a_whole_seq_one_hot_increase, rbd, same, x, y, z])
bc_b_increase = binding_affinity_predictor.predict([pre, mut_b_whole_seq_one_hot_increase, rbd, same, x, y, z])
bc_c_increase = binding_affinity_predictor.predict([pre, mut_c_whole_seq_one_hot_increase, rbd, same, x, y, z])



bc_d_rd = binding_affinity_predictor.predict([pre, mut_d_whole_seq_one_hot_rd, rbd, same, x, y, z])
bc_a_rd = binding_affinity_predictor.predict([pre, mut_a_whole_seq_one_hot_rd, rbd, same, x, y, z])
bc_b_rd = binding_affinity_predictor.predict([pre, mut_b_whole_seq_one_hot_rd, rbd, same, x, y, z])
bc_c_rd = binding_affinity_predictor.predict([pre, mut_c_whole_seq_one_hot_rd, rbd, same, x, y, z])

# with open('application/bc_d_increase.pkl', 'wb') as f:
#     pickle.dump(bc_d_increase, f)
# with open('application/bc_a_increase.pkl', 'wb') as f:
#     pickle.dump(bc_a_increase, f)
# with open('application/bc_b_increase.pkl', 'wb') as f:
#     pickle.dump(bc_b_increase, f)
# with open('application/bc_c_increase.pkl', 'wb') as f:
#     pickle.dump(bc_c_increase, f)

# with open('application/bc_d.pkl', 'wb') as f:
#     pickle.dump(bc_d, f)
# with open('application/bc_a.pkl', 'wb') as f:
#     pickle.dump(bc_a, f)
# with open('application/bc_b.pkl', 'wb') as f:
#     pickle.dump(bc_b, f)
# with open('application/bc_c.pkl', 'wb') as f:
#     pickle.dump(bc_c, f)

# with open('application/bc_d_rd.pkl', 'wb') as f:
#     pickle.dump(bc_d_rd, f)
# with open('application/bc_a_rd.pkl', 'wb') as f:
#     pickle.dump(bc_a_rd, f)
# with open('application/bc_b_rd.pkl', 'wb') as f:
#     pickle.dump(bc_b_rd, f)
# with open('application/bc_c_rd.pkl', 'wb') as f:
#     pickle.dump(bc_c_rd, f)
with open('application/bc_d_rd.pkl', 'rb') as f:
    bc_d_rd = pickle.load(f)
with open('application/bc_a_rd.pkl', 'rb') as f:
    bc_a_rd = pickle.load(f)
with open('application/bc_b_rd.pkl', 'rb') as f:
    bc_b_rd = pickle.load(f)
with open('application/bc_c_rd.pkl', 'rb') as f:
    bc_c_rd = pickle.load(f)

with open('application/bc_d.pkl', 'rb') as f:
    bc_d = pickle.load(f)
with open('application/bc_a.pkl', 'rb') as f:
    bc_a = pickle.load(f)
with open('application/bc_b.pkl', 'rb') as f:
    bc_b = pickle.load(f)
with open('application/bc_c.pkl', 'rb') as f:
    bc_c = pickle.load(f)

print(summarize_data(bc_d.flatten()))
print(summarize_data(bc_a.flatten()))
print(summarize_data(bc_b.flatten()))
print(summarize_data(bc_c.flatten()))
# {'mean': -1.995575, 'median': -2.0383739471435547, 'stdev': 0.6831908119953286}
# {'mean': -3.563173, 'median': -3.4927725791931152, 'stdev': 0.6975929717631778}
# {'mean': -2.6676855, 'median': -2.7189090251922607, 'stdev': 0.6289915357642614}
# {'mean': -2.710681, 'median': -2.738837957382202, 'stdev': 0.6595783397437139}
print(summarize_data(bc_d_rd.flatten()))
print(summarize_data(bc_a_rd.flatten()))
print(summarize_data(bc_b_rd.flatten()))
print(summarize_data(bc_c_rd.flatten()))
# {'mean': 0.10249928, 'median': 0.07007055729627609, 'stdev': 0.26427862617704223}
# {'mean': 0.20010649, 'median': 0.15590600669384003, 'stdev': 0.23737608632175053}
# {'mean': 0.18679492, 'median': 0.11103647202253342, 'stdev': 0.28710027467813487}
# {'mean': 0.029942581, 'median': -0.03419894725084305, 'stdev': 0.2104591164352992}
with open('application/bc_d_increase.pkl', 'rb') as f:
    bc_d_increase = pickle.load(f)
with open('application/bc_a_increase.pkl', 'rb') as f:
    bc_a_increase = pickle.load(f)
with open('application/bc_b_increase.pkl', 'rb') as f:
    bc_b_increase = pickle.load(f)
with open('application/bc_c_increase.pkl', 'rb') as f:
    bc_c_increase = pickle.load(f)

print(summarize_data(bc_d_increase.flatten()))
print(summarize_data(bc_a_increase.flatten()))
print(summarize_data(bc_b_increase.flatten()))
print(summarize_data(bc_c_increase.flatten()))
# {'mean': 4.759762, 'median': 4.761233329772949, 'stdev': 0.23186710616551087}
# {'mean': 4.9660525, 'median': 4.720758438110352, 'stdev': 1.5776746331287987}
# {'mean': 4.25458, 'median': 4.241671085357666, 'stdev': 0.6741813824633586}
# {'mean': 4.2670383, 'median': 4.262747764587402, 'stdev': 0.754919839764258}
# scatter plot
deepdirect_bd_dict = {'D-ACE2': bc_d.flatten(),
                      'D-N Chain A': bc_a.flatten(),
                      'D-N Chain B': bc_b.flatten(),
                      'D-N Chain C': bc_c.flatten()}

deepdirect_bd_increase_dict = {'D-ACE2': bc_d_increase.flatten(),
                      'D-N Chain A': bc_a_increase.flatten(),
                      'D-N Chain B': bc_b_increase.flatten(),
                      'D-N Chain C': bc_c_increase.flatten()}

random_bd_dict = {'R-ACE2': bc_d_rd.flatten(),
                  'R-N Chain A': bc_a_rd.flatten(),
                  'R-N Chain B': bc_b_rd.flatten(),
                  'R-N Chain C': bc_c_rd.flatten()}

deepdirect_bd_df = pd.DataFrame(deepdirect_bd_dict)
deepdirect_bd_increase_df = pd.DataFrame(deepdirect_bd_increase_dict)
random_bd_df = pd.DataFrame(random_bd_dict)

f = sns.scatterplot(data=deepdirect_bd_df, s=75)
f.set_xlabel("Num", fontsize=12)
f.set_ylabel("DDG kcal/mol", fontsize=12)
# plt.ylim(-6, 3)
plt.ylim(-10.5, 10.5)
plt.savefig('diagram/deepdirect_bd_chain.pdf')
plt.show()

f = sns.scatterplot(data=deepdirect_bd_increase_df, s=75)
f.set_xlabel("Num", fontsize=12)
f.set_ylabel("DDG kcal/mol", fontsize=12)
plt.ylim(-10.5, 10.5)
plt.savefig('diagram/deepdirect_bd_increase_chain.pdf')
plt.show()

f = sns.scatterplot(data=random_bd_df, s=75)
f.set_xlabel("Num", fontsize=12)
f.set_ylabel("DDG kcal/mol", fontsize=12)
# plt.ylim(-6, 3)
plt.ylim(-10.5, 10.5)
plt.savefig('diagram/random_bd_chain.pdf')
plt.show()

# boxplot
f = sns.boxplot(data=deepdirect_bd_df)
f.set_ylabel("DDG kcal/mol", fontsize=12)
# plt.ylim(-6, 3)
plt.ylim(-10.5, 10.5)
plt.savefig('diagram/deepdirect_bd_chain_boxplot.pdf')
plt.show()

f = sns.boxplot(data=random_bd_df)
f.set_ylabel("DDG kcal/mol", fontsize=12)
# plt.ylim(-6, 3)
plt.ylim(-10.5, 10.5)
plt.savefig('diagram/random_bd_chain_boxplot.pdf')
plt.show()

f = sns.boxplot(data=deepdirect_bd_increase_df)
f.set_ylabel("DDG kcal/mol", fontsize=12)
# plt.ylim(-6, 3)
plt.ylim(-10.5, 10.5)
plt.savefig('diagram/deepdirect_bd_chain_increase_boxplot.pdf')
plt.show()

# intergrated boxplot
with open('application/deep_result_3-bc--2.501-ratio-0.876.pkl', 'rb') as f:
    deep_result_3 = pickle.load(f)
with open('application/bc_rd_list.pkl', 'rb') as f:
    bc_rd_list = pickle.load(f)

# with open('application/bc_d.pkl', 'rb') as f:
#     bc_d = pickle.load(f)
# with open('application/bc_a.pkl', 'rb') as f:
#     bc_a = pickle.load(f)
# with open('application/bc_b.pkl', 'rb') as f:
#     bc_b = pickle.load(f)
# with open('application/bc_c.pkl', 'rb') as f:
#     bc_c = pickle.load(f)

# deep_result_bc = []
# deep_result_bc.append([i[0] for i in bc_d])
# deep_result_bc.append([i[0] for i in bc_a])
# deep_result_bc.append([i[0] for i in bc_b])
# deep_result_bc.append([i[0] for i in bc_c])


# integrated_bc_dict = {'D-N-ACE2': deep_result_3['deep_bc_list'],
#                       'R-N-ACE2': np.array(bc_rd_list)}
# integrated_bc_df = pd.DataFrame(integrated_bc_dict)

integrated_bc_dict = {'D-N-ACE2-i': deep_result_bc[0],
                      'R-N-ACE2': np.array(bc_rd_list),
                      'D-N-ACE2-d': deep_result_increase['deep_bc_list']}
integrated_bc_df = pd.DataFrame(integrated_bc_dict)


f = sns.boxplot(data=integrated_bc_df)
f.set_ylabel("DDG kcal/mol", fontsize=12)
# plt.ylim(-10.5, 10.5)
plt.savefig('diagram/bd_integrated_boxplot.pdf')
plt.show()

# metric
# aa_generator_final

final_model_loss = pd.read_csv('metric/model_i_loss.log')
final_d_loss = pd.DataFrame(final_model_loss['d_loss'].copy())
final_d_loss['num'] = range(len(final_d_loss))
final_g_loss = pd.DataFrame(final_model_loss['g_loss'].copy())
final_g_loss['num'] = range(len(final_g_loss))

# separate figure
sns.lineplot(data = final_d_loss, x = 'num', y = 'd_loss')
plt.savefig('diagram/final_d_loss.pdf')
plt.show()
# sns.scatterplot(data = final_d_loss, x = 'num', y = 'd_loss', s = 2)
# plt.show()

sns.lineplot(data = final_g_loss, x = 'num', y = 'g_loss')
plt.savefig('diagram/final_g_loss.pdf')
plt.show()
# sns.scatterplot(data = final_g_loss, x = 'num', y = 'g_loss', s = 2)
# plt.show()

# integrated figure

final_loss_integrated = pd.merge(final_g_loss, final_d_loss)
del final_loss_integrated['num']
f = sns.lineplot(data = final_loss_integrated, dashes = False)
f.set_xlabel("Num", fontsize = 15)
f.set_ylabel("Loss", fontsize = 15)
plt.savefig('diagram/final_loss_integrated.pdf')
plt.show()

#####################################################################################################################
# aa_generator_final increase

final_model_increase_loss = pd.read_csv('metric/model_d_loss.log')
final_d_increase_loss = pd.DataFrame(final_model_increase_loss['d_loss'].copy())
final_d_increase_loss['num'] = range(len(final_d_increase_loss))
final_g_increase_loss = pd.DataFrame(final_model_increase_loss['g_loss'].copy())
final_g_increase_loss['num'] = range(len(final_g_increase_loss))

# separate figure
sns.lineplot(data = final_d_increase_loss, x = 'num', y = 'd_loss')
plt.savefig('diagram/final_d_increase_loss.pdf')
plt.show()


sns.lineplot(data = final_g_increase_loss, x = 'num', y = 'g_loss')
plt.savefig('diagram/final_g_increase_loss.pdf')
plt.show()


# integrated figure

final_loss_increase_integrated = pd.merge(final_g_increase_loss, final_d_increase_loss)
del final_loss_increase_integrated['num']
f = sns.lineplot(data = final_loss_increase_integrated, dashes = False)
f.set_xlabel("Num", fontsize = 15)
f.set_ylabel("Loss", fontsize = 15)
plt.savefig('diagram/final_loss_increase_integrated.pdf')
plt.show()

