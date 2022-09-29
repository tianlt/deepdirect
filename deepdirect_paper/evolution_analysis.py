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
from sgt import SGT
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandarallel
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from collections import Counter

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

def write_fasta_from_list(data, name, file_path):
    output_file = open(file_path, 'w')
    for i in range(len(data)):
        seq = ''.join(data[i])
        dict = {name[i]: seq}
        for i, j in dict.items():
            identifier = '>' + i + '\n'
            output_file.write(identifier)
            sequence = j + '\n'
            output_file.write(sequence)
    output_file.close()
#%%#######################################
pdb_name = '7wpa'
pre_dict = {}
pre_all_result_dict = {}

ppdb = PandasPdb().read_pdb('application/' + pdb_name + '.pdb')
residue_seq = ppdb.amino3to1()
alpha_carbon_coordinate = ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].iloc[residue_seq.index + 1].to_numpy()
same_index = [0 if aa == 'D' else 1 for aa in residue_seq['chain_id']]
chain_index = residue_seq['chain_id'].tolist()
rbd_index = find_rbd(alpha_carbon_coordinate, 50, same_index, 0.1)  # 50
residue_num = ppdb.df['ATOM'][['residue_number']].iloc[residue_seq.index + 1].to_numpy()

# np.savetxt('application/result/rbd_index.csv', rbd_index, delimiter =", ",
#            fmt ='% s')
# seq = ''.join(residue_seq['residue_name'])
# with open("application/result/seq.txt", "w") as f:
#     f.write(seq)
#
# len(seq)
# len(rbd_index)
# %% model assembly
aa_mutator_final_3 = build_aa_mutator()
aa_mutator_final_3.load_weights(
    'models/33741-aa_mutator-num-0-dlr-0.000008-glr-0.000008-drt-0.529-grt-0.527-daff--1.673-gaff--1.707-epoch-1_weights.h5')
aa_mutator_final_increase = build_aa_mutator()
aa_mutator_final_increase.load_weights(
    'models/increase-84263-aa_mutator-num-0-dlr-0.000005-glr-0.000010-drt-0.543-grt-0.545-daff-4.258-gaff-4.479-epoch-1_weights.h5')
binding_affinity_predictor = build_model(activation='linear', f_num_1=32, f_num_2=64, f_num_3=128, f_num_4=256)
binding_affinity_predictor.load_weights(
    'models/17612-74607-binding_affinity_predictor_weights_lr-0.0001-tr-[[1.54566794]]-ev-[[1.52877662]]-te-[[1.82225001]]-tr_acc-0.7008-ev_acc-0.77-te_acc-0.734_train0.h5')
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
def deep_direct_mutate_evolution(aa_mutator_final, input_pre, input_rbd, input_same, input_x, input_y, input_z, num=500):
    aa_mutator_final = aa_mutator_final
    pre = tf.one_hot(input_pre, 20)
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

#%%###################################################
out_num_list = []
bc_list = []
similarity_ratio_list = []
deep_out_num, deep_bc_list, deep_similarity_ratio_list = deep_direct_mutate(aa_mutator_final_3, input_pre,
                                                                                  input_rbd, input_same, input_x,
                                                                                  input_y, input_z, num=1)
bc_list.append(deep_bc_list[0])
out_num_list.append(deep_out_num[0])
similarity_ratio_list.append(deep_similarity_ratio_list[0])

for _ in range(3):
    deep_out_num, deep_bc_list, deep_similarity_ratio_list = deep_direct_mutate_evolution(aa_mutator_final_3, deep_out_num,
                                                                                  input_rbd, input_same, input_x,
                                                                                  input_y, input_z, num=1)
    bc_list.append(deep_bc_list[0])
    out_num_list.append(deep_out_num[0])
    similarity_ratio_list.append(deep_similarity_ratio_list[0])

df = pd.DataFrame({'bc': bc_list,'similarity_ratio': similarity_ratio_list})
df
#out#####################################
# bc  similarity_ratio
# 0 -2.326916          0.890302
# 1 -1.031901          0.980922
# 2 -0.162329          0.976948
# 3 -0.276356          0.995760
#out#####################################
# with open('application/evolution_bc_sr_df.pkl', 'wb') as f:
#     pickle.dump(df, f)
# with open('application/evolution_out_num_list.pkl', 'wb') as f:
#     pickle.dump(out_num_list, f)

# %%
# model increase (decrease in binding affinity)

out_num_increase_list = []
bc_increase_list = []
similarity_ratio_increase_list = []
deep_out_increase_num = deep_out_num.copy()
for _ in range(2):
    deep_out_increase_num, deep_bc_list, deep_similarity_ratio_list = deep_direct_mutate_evolution(aa_mutator_final_increase, deep_out_increase_num,
                                                                                  input_rbd, input_same, input_x,
                                                                                  input_y, input_z, num=1)
    bc_increase_list.append(deep_bc_list[0])
    out_num_increase_list.append(deep_out_num[0])
    similarity_ratio_increase_list.append(deep_similarity_ratio_list[0])
df = pd.DataFrame({'bc': bc_increase_list,'similarity_ratio': similarity_ratio_increase_list})
df
#out#####################################
# bc  similarity_ratio
# 0  6.772795          0.914149
# 1  5.648748          0.954425
#out#####################################
# plot
bc_plot_list = []
bc_plot_list.append(np.array(bc_list))
bc_plot_list.append(bc_increase_list)
bc_plot_list = [item for sublist in bc_plot_list for item in sublist]
bc_plot_binding_affinity_list = [0, 0 + bc_plot_list[0], 0 + bc_plot_list[0] + bc_plot_list[1],
                0 + bc_plot_list[0] + bc_plot_list[1] + bc_plot_list[2],
                0 + bc_plot_list[0] + bc_plot_list[1] + bc_plot_list[2] + bc_plot_list[3],
                0 + bc_plot_list[0] + bc_plot_list[1] + bc_plot_list[2] + bc_plot_list[3] + bc_plot_list[4],
                0 + bc_plot_list[0] + bc_plot_list[1] + bc_plot_list[2] + bc_plot_list[3] + bc_plot_list[4] + bc_plot_list[5]]
# with open('application/bc_plot_list.pkl', 'wb') as f:
#     pickle.dump(bc_plot_list, f)
# with open('application/bc_plot_binding_affinity_list.pkl', 'wb') as f:
#     pickle.dump(bc_plot_binding_affinity_list, f)
with open('application/bc_plot_list.pkl', 'rb') as f:
    bc_plot_list = pickle.load(f)
with open('application/bc_plot_binding_affinity_list.pkl', 'rb') as f:
    bc_plot_binding_affinity_list = pickle.load(f)
bc_plot_binding_affinity_pd = pd.DataFrame(np.array(bc_plot_binding_affinity_list)).T
bc_plot_binding_affinity_pd.columns=['state_0', 'm_i_1','m_i_2','m_i_3','m_i_4','m_d_1','m_d_2']



plt.plot(bc_plot_binding_affinity_pd.T, linewidth=3)
plt.axhline(y=0, linewidth=1, linestyle='dashed', color='k')
# plt.axhline(x=0, linewidth=1, linestyle='dashed', color='k')
plt.axvline(x=4, linewidth=2, color = 'r',label = 'vline_multiple - full height')
plt.ylabel("DDG Kcal/mol")
plt.text(1, 7, 'Evolution',
         fontsize = 15, color = 'g')
plt.text(4.1, 7, 'Devolution',
         fontsize = 15, color = 'g')
# plt.xlabel("Sorted observations (4th NN)")
plt.savefig('diagram/bc_plot_binding_affinity.pdf')
plt.show()


# with open('application/evolution_bc_sr_increase_df.pkl', 'wb') as f:
#     pickle.dump(df, f)
# with open('application/evolution_out_num_increase_list.pkl', 'wb') as f:
#     pickle.dump(out_num_increase_list, f)
#%%#multiple##################################################
# random.seed(1)
# mul_out_num_list = []
# mul_bc_list = []
# mul_similarity_ratio_list = []
# mul_deep_out_num, mul_deep_bc_list, mul_deep_similarity_ratio_list = deep_direct_mutate(aa_mutator_final_3, input_pre,
#                                                                                   input_rbd, input_same, input_x,
#                                                                                   input_y, input_z, num=500)
# mul_bc_list.append(mul_deep_bc_list)
# mul_out_num_list.append(mul_deep_out_num)
# mul_similarity_ratio_list.append(mul_deep_similarity_ratio_list)
# t_range = tqdm(range(50))
# for _ in t_range:
#     mul_deep_out_num, mul_deep_bc_list, mul_deep_similarity_ratio_list = deep_direct_mutate_evolution(aa_mutator_final_3, mul_deep_out_num,
#                                                                                   input_rbd, input_same, input_x,
#                                                                                   input_y, input_z, num=500)
#     mul_bc_list.append(mul_deep_bc_list)
#     mul_out_num_list.append(mul_deep_out_num)
#     mul_similarity_ratio_list.append(mul_deep_similarity_ratio_list)

# save
# with open('application/evolution_mul_bc_list.pkl', 'wb') as f:
#     pickle.dump(mul_bc_list, f)
# with open('application/evolution_mul_out_num_list.pkl', 'wb') as f:
#     pickle.dump(mul_out_num_list, f)
# with open('application/evolution_mul_similarity_ratio_list.pkl', 'wb') as f:
#     pickle.dump(mul_similarity_ratio_list, f)



#%% cluster sequences
with open('application/evolution_mul_out_num_list.pkl', 'rb') as f:
    mul_out_num_list = pickle.load(f)


mutated_seq_set = mul_out_num_list[-1]
# extract spike
mutated_seq_set_s = [i[0:3178] for i in mutated_seq_set]
mutated_seq_set_decoded = [np.array(decode_seq_int(seq)) for seq in mutated_seq_set_s]

mutated_seq_set_decoded_df = pd.DataFrame({'id': ['seq_'+ str(i) for i in range(500)], 'sequence': mutated_seq_set_decoded})

sgt = SGT(kappa=1,
          flatten=True,
          lengthsensitive=False,
          mode='multiprocessing')

mutated_seq_set_sgt_df_k1 = sgt.fit_transform(mutated_seq_set_decoded_df)

# with open('application/evolution_mutated_seq_set_sgt_df_k1.pkl', 'wb') as f:
#     pickle.dump(mutated_seq_set_sgt_df_k1, f)

with open('application/evolution_mutated_seq_set_sgt_df_k1.pkl', 'rb') as f:
    mutated_seq_set_sgt_df_k1 = pickle.load(f)

sgt = SGT(kappa=5,
          flatten=True,
          lengthsensitive=False,
          mode='multiprocessing')

mutated_seq_set_sgt_df_k5 = sgt.fit_transform(mutated_seq_set_decoded_df)

# with open('application/evolution_mutated_seq_set_sgt_df_k5.pkl', 'wb') as f:
#     pickle.dump(mutated_seq_set_sgt_df_k5, f)
#
with open('application/evolution_mutated_seq_set_sgt_df_k5.pkl', 'rb') as f:
    mutated_seq_set_sgt_df_k5 = pickle.load(f)

# pca and cluster
# k5
mutated_seq_set_sgt_df_k5 = mutated_seq_set_sgt_df_k5.set_index('id')
pca = PCA(n_components=2)
pca.fit(mutated_seq_set_sgt_df_k5)
X_k5 = pca.transform(mutated_seq_set_sgt_df_k5)
print(np.sum(pca.explained_variance_ratio_))
df_k5 = pd.DataFrame(data=X_k5, columns=['PC_1', 'PC_2'])
df_k5.head()


# dbscan
nbrs_k5 = NearestNeighbors(n_neighbors=5).fit(df_k5)
neigh_dist_k5, neigh_ind_k5 = nbrs_k5.kneighbors(df_k5)
sort_neigh_dist_k5 = np.sort(neigh_dist_k5, axis=0)
k_dist_k5 = sort_neigh_dist_k5[:, 4]
plt.plot(k_dist_k5)
# plt.axhline(y=0.0065, linewidth=1, linestyle='dashed', color='k')
plt.axhline(y=0.01, linewidth=1, linestyle='dashed', color='k')
plt.ylabel("k-NN distance")
plt.xlabel("Sorted observations (4th NN)")
plt.savefig('diagram/k-NN_distance_k5.pdf')
plt.show()

# dbscan_k5 = DBSCAN(eps=0.0065).fit(df_k5)
dbscan_k5 = DBSCAN(eps=0.01).fit(df_k5)
Counter(dbscan_k5.labels_)

sns.scatterplot(data=df_k5, x="PC_1", y="PC_2", hue=dbscan_k5.labels_, palette="deep")
plt.savefig('diagram/dbscan_k5.pdf')
plt.show()

# with open('application/evolution_dbscan_k5_labels.pkl', 'wb') as f:
#     pickle.dump(dbscan_k5.labels_, f)

# sequece analysis###########################################
with open('application/evolution_dbscan_k5_labels.pkl', 'rb') as f:
    dbscan_k5_label = pickle.load(f)

with open('application/evolution_mutated_seq_set_sgt_df_k5.pkl', 'rb') as f:
    mutated_seq_set_sgt_df_k5 = pickle.load(f)

with open('application/evolution_mul_out_num_list.pkl', 'rb') as f:
    mul_out_num_list = pickle.load(f)

mutated_seq_set = mul_out_num_list[-1]

# pca.components_
pca_df = pd.DataFrame(pca.components_,columns=mutated_seq_set_sgt_df_k5.columns).T
pc_1 = np.array(pca_df[0]).reshape(20,20)
pc_2 = np.array(pca_df[1]).reshape(20,20)

# sns.scatterplot(data = pca_df, x = 'num', y = 'loss', s = 2)
sns.scatterplot(data = np.array(pca_df))
plt.show()

axis_labels = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
sns.heatmap(pc_1, xticklabels=axis_labels, yticklabels=axis_labels, vmin=-1, vmax=1, cmap="PiYG")
plt.savefig('diagram/heatmap_pc_1.pdf')
plt.show()
sns.heatmap(pc_2, xticklabels=axis_labels, yticklabels=axis_labels, vmin=-1, vmax=1, cmap="PiYG")
plt.savefig('diagram/heatmap_pc_2.pdf')
plt.show()




mutated_seq_set_decoded = [decode_seq_int(i) for i in mutated_seq_set]
write_fasta_from_list(mutated_seq_set_decoded, ['seq_' + str(i) for i in range(500)], 'application/result/mutated_seq_set_decoded.fasta')

cluster_index_0 = (dbscan_k5.labels_ == [0]).tolist()
cluster_index_1 = (dbscan_k5.labels_ == [1]).tolist()
cluster_index_2 = (dbscan_k5.labels_ == [2]).tolist()
mutated_seq_set_decoded_0 = [i for i, j in zip(mutated_seq_set_decoded, cluster_index_0) if j]
mutated_seq_set_decoded_1 = [i for i, j in zip(mutated_seq_set_decoded, cluster_index_1) if j]
mutated_seq_set_decoded_2 = [i for i, j in zip(mutated_seq_set_decoded, cluster_index_2) if j]
write_fasta_from_list(mutated_seq_set_decoded_0, ['seq_' + str(i) for i in range(len(mutated_seq_set_decoded_0))], 'application/result/mutated_seq_set_decoded_0.fasta')
write_fasta_from_list(mutated_seq_set_decoded_1, ['seq_' + str(i) for i in range(len(mutated_seq_set_decoded_1))], 'application/result/mutated_seq_set_decoded_1.fasta')
write_fasta_from_list(mutated_seq_set_decoded_2, ['seq_' + str(i) for i in range(len(mutated_seq_set_decoded_2))], 'application/result/mutated_seq_set_decoded_2.fasta')

