# %%
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import os
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
import numpy as np
import model_function
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping

from model_function import build_model, build_aa_mutator, AA_Mutation_GAN, generator_loss, discriminator_loss, \
    discriminator_loss_replace, generator_loss_replace, build_discriminator_replace_model

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"


tf.config.run_functions_eagerly(True)
with open('data/skempi_all_result_dict.pkl', 'rb') as f:
    skempi_all_result_dict = pickle.load(f)

with open('data/ab_all_result_dict.pkl', 'rb') as f:
    ab_all_result_dict = pickle.load(f)

with open('model_data/all_list_ab_bind_name.pkl', 'rb') as f:
    all_list_ab_bind_name = pickle.load(f)

a_list = all_list_ab_bind_name[0]
b_list = all_list_ab_bind_name[1]
c_list = all_list_ab_bind_name[2]
d_list = all_list_ab_bind_name[3]
x_list = all_list_ab_bind_name[4]
y_list = all_list_ab_bind_name[5]
z_list = all_list_ab_bind_name[6]
e_list = all_list_ab_bind_name[7]


def get_batch():
    # batch_n = 5
    batch_n = len(a_list)
    for i in range(batch_n):
        a = a_list[i]
        b = b_list[i]
        c = c_list[i]
        d = d_list[i]
        x = x_list[i]
        y = y_list[i]
        z = z_list[i]
        e = e_list[i]
        yield a,b,c,d,x,y,z,e
# load trained model

aa_mutator_phase_2 = build_aa_mutator()
aa_mutator_phase_2.load_weights('models/generator_step_weights.h5') 

discriminator_replace = build_discriminator_replace_model()

binding_affinity_predictor = build_model(activation='linear', f_num_1=32, f_num_2=64, f_num_3=128, f_num_4=256)
binding_affinity_predictor.load_weights('models/binding_affinity_predictor_weights.h5')

aa_mutator_final = build_aa_mutator()
aa_mutator_final.load_weights('models/model_i_weights.h5') #

#####################################################################################################################
# metric
# binding affinity predictor
# with open('metric/binding_affinity_predictor_loss.pkl', 'rb') as f:
#     loss = pickle.load(f)
# def flatten(t):
#     return [item[0] for sublist in t for item in sublist]
# loss = flatten(loss)
# pd_loss = pd.DataFrame(loss, columns=['loss'])
# pd_loss['num'] = range(len(loss))
# 
# # sns.displot(loss)
# sns.lineplot(data = pd_loss, x = 'num', y = 'loss')
# plt.show()
# sns.scatterplot(data = pd_loss, x = 'num', y = 'loss', s = 2)
# plt.show()

#####################################################################################################################
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