# %%
import tensorflow as tf
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
import string
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

with open('data/skempi_all_result_dict.pkl', 'rb') as f:
    skempi_all_result_dict = pickle.load(f)

with open('data/ab_all_result_dict.pkl', 'rb') as f:
    ab_all_result_dict = pickle.load(f)

ab_name = list(ab_all_result_dict.keys())
skempi_name = list(skempi_all_result_dict.keys())

# input data
# data = ab_all_result_dict[ab_name[1]]
# input_label = data['result']['subset_ddg']
# 
# input_rbd_index = [data['rbd_index']]*len(input_label)
# input_pre_mutated_seq = [data['result']['subset_pre_mutated_seq']]*len(input_label)
# input_aft_mutated_seq = data['result']['subset_after_mutated_seq']
# input_chain_index = [data['result']['subset_chain_index']]*len(input_label)
# input_same_index = [data['result']['subset_same_index']]*len(input_label)
# #added
# input_coordinate_loc = [data['subset_alpha_carbon_coordinate']]*len(input_label)


# len(input_rbd_index[0])
# len(input_pre_mutated_seq[0])
# len(input_after_mutated_seq[0])
# len(input_chain_index[0])

# build model
# set parameter
# latent_dim = 64
# seq num have to >= embedded integer + 1
# seq_num = 27
# out_len = 5
# index_num = 2
# f_num_1 = 64
# f_num_2 = 128
# f_num_3 = 256
# f_num_4 = 512
# k_size = 5
# drop_ratio = 0.2
# dense_1_num = 128
# dense_2_num = 64
# dense_3_num = 8
# dense_4_num = 1
#
aa_string = 'ARNDCEQGHILKMFPSTWYV'
def encode_seq_int(data= 'HELLOWORLD'):
    # seq_letter = string.ascii_uppercase
    seq_letter = aa_string
    char_to_int = dict((c, i+1) for i, c in enumerate(seq_letter))
    integer_encoded = [char_to_int[char] for char in data]
    return integer_encoded


# input_pre = np.array(encode_seq_int(''.join(input_pre_mutated_seq[0])))
# input_aft = np.array(encode_seq_int(''.join(input_aft_mutated_seq[0])))
# input_rbd = np.array(list(input_rbd_index[0]))
# input_same = np.array([int(x) for x in input_same_index[0]])
# input_coordinate = input_coordinate_loc[0]
# input_x = input_coordinate[:,0]
# input_y = input_coordinate[:,1]
# input_z = input_coordinate[:,2]
# 
# input_pre = np.reshape(input_pre, (-1,len(input_pre)))
# input_aft = np.reshape(input_aft, (-1,len(input_aft)))
# input_rbd = np.reshape(input_rbd, (-1,len(input_rbd),1))
# input_same = np.reshape(input_same, (-1,len(input_same),1))
# input_x = np.reshape(input_x, (-1,len(input_x),1))
# input_y = np.reshape(input_y, (-1,len(input_y),1))
# input_z = np.reshape(input_z, (-1,len(input_z),1))
# # normalize input
# input_x = keras.utils.normalize(input_x, axis=1)
# input_y = keras.utils.normalize(input_y, axis=1)
# input_z = keras.utils.normalize(input_z, axis=1)
# 
# input_pre = tf.one_hot(input_pre, 20) #@@@
# input_aft = tf.one_hot(input_aft, 20) #@@@






#original model####################################
def build_model(activation, latent_dim = 64, seq_num = 27, out_len = 5,f_num_1 = 64,
                        f_num_2 = 128,f_num_3 = 256,f_num_4 = 512,k_size = 5,drop_ratio = 0.2,
                        dense_1_num = 128,dense_2_num = 64,dense_3_num = 8,dense_4_num = 1):
    # build
    # input and embedding
    Input_pre = keras.Input(shape=(None,20), name='Input_pre') #@@@
    Input_aft = keras.Input(shape=(None,20), name='Input_aft') #@@@

    Input_rbd = keras.Input(shape=(None, 1), name='Input_rbd')
    Input_same = keras.Input(shape=(None, 1), name='Input_same')


    Input_x = keras.Input(shape=(None, 1), name='Input_x')
    Input_y = keras.Input(shape=(None, 1), name='Input_y')
    Input_z = keras.Input(shape=(None, 1), name='Input_z')

    diff_layer = layers.subtract([Input_pre, Input_aft])
    # lstm
    x_pre = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True), name='Bidirect_pre')(Input_pre) #@@@

    # print('Input_aft.shape')
    # print(Input_aft.shape)
    # x_aft = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True), name='Bidirect_aft')(Input_aft) #@@@

    x_diff = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True), name='Bidirect_diff')(diff_layer)

    x_rbd = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True), name='Bidirect_rbd')(Input_rbd)
    x_same = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True), name='Bidirect_same')(Input_same)
    x_x = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True), name='Bidirect_x')(Input_x)
    x_y = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True), name='Bidirect_y')(Input_y)
    x_z = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True), name='Bidirect_z')(Input_z)


    x_pre = layers.Reshape((-1, x_pre.shape[2], 1))(x_pre)
    # x_aft = layers.Reshape((-1, x_aft.shape[2], 1))(x_aft)
    ########################################################################
    x_diff = layers.Reshape((-1, x_diff.shape[2], 1))(x_diff)
    ########################################################################
    x_rbd = layers.Reshape((-1, x_rbd.shape[2], 1))(x_rbd)
    x_same = layers.Reshape((-1, x_same.shape[2], 1))(x_same)
    x_x = layers.Reshape((-1, x_x.shape[2], 1))(x_x)
    x_y = layers.Reshape((-1, x_y.shape[2], 1))(x_y)
    x_z = layers.Reshape((-1, x_z.shape[2], 1))(x_z)

    ########################################################################
    Concat_aux = layers.Concatenate(axis=3, name='Concat_aux')([x_rbd, x_same, x_x, x_y, x_z])  # (none, none, 128, 5)


    ###############################################################################
    # cnn
    def create_cnn_block(input, filter_num, kernel_size, drop_out=True):
        x = layers.Conv2D(filter_num, kernel_size, padding='same')(input)
        x = layers.BatchNormalization()(x)
        x = keras.activations.relu(x)
        x = layers.MaxPool2D(padding='same')(x)
        if drop_out:
            x = layers.Dropout(drop_ratio)(x)
        return x
    #################################################################################
    Concat_aux_conv = create_cnn_block(Concat_aux, f_num_1, k_size)
    Concat_aux_conv = create_cnn_block(Concat_aux_conv, f_num_2, k_size)
    Concat_x_pre_conv = create_cnn_block(x_pre, f_num_1, k_size)
    Concat_x_diff_conv = create_cnn_block(x_diff, f_num_1, k_size)

    Concat_x_pre_conv = create_cnn_block(Concat_x_pre_conv, f_num_2, k_size)
    Concat_x_diff_conv = create_cnn_block(Concat_x_diff_conv, f_num_2, k_size)
    Concat = layers.Concatenate(axis=3, name='Concat')([Concat_aux_conv, Concat_x_pre_conv, Concat_x_diff_conv])


    ##################################################################################
    Cnn_1 = create_cnn_block(Concat, f_num_1, k_size)
    Cnn_2 = create_cnn_block(Cnn_1, f_num_2, k_size)
    Cnn_3 = create_cnn_block(Cnn_2, f_num_3, k_size)
    Cnn_4 = create_cnn_block(Cnn_3, f_num_4, k_size, drop_out=False)
    MaxPooling = layers.GlobalMaxPooling2D()(Cnn_4)

    # dense
    Dense = layers.Dense(dense_1_num, activation='relu')(MaxPooling)
    Dense = layers.Dense(dense_2_num, activation='relu')(Dense)
    Dense = layers.Dense(dense_3_num)(Dense)
    # Dense = layers.BatchNormalization()(Dense)
    Dense = layers.LeakyReLU(alpha=0.2)(Dense)
    Dense = layers.Dense(dense_4_num)(Dense)

    if activation == 'sigmoid':
        # Dense = layers.BatchNormalization()(Dense)
        Dense = layers.Activation(keras.activations.sigmoid)(Dense)
    if activation == 'linear':
        # Dense = layers.BatchNormalization()(Dense)
        Dense = layers.Activation(keras.activations.linear)(Dense)


    model = keras.models.Model([Input_pre, Input_aft, Input_rbd, Input_same, Input_x, Input_y, Input_z], Dense)
    return model



model = build_model(activation='linear',f_num_1 = 32,f_num_2 = 64,f_num_3 = 128,f_num_4 = 256)


model.summary()


# %%


old_weights = model.get_weights()
def build_and_compile_binding_affinity_predictor(lr):
    model = build_model(activation='linear', f_num_1=32, f_num_2=64, f_num_3=128, f_num_4=256)
    model.set_weights(old_weights)
    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='mae', metrics=['mae'])
    return model


ab_bind_name = ab_name.copy()
for i in ab_name:
    if ab_all_result_dict[i]['result']['subset_ddg'].size == 0:
        ab_bind_name.remove(i)

skempi_bind_name = skempi_name.copy()
for i in skempi_name:
    if skempi_all_result_dict[i]['result']['subset_ddg'].size == 0 or \
            '?' in skempi_all_result_dict[i]['result']['subset_pre_mutated_seq'] or \
            len(skempi_all_result_dict[i]['result']['subset_after_mutated_seq']) != len(skempi_all_result_dict[i]['result']['subset_ddg']):
        skempi_bind_name.remove(i)

# %%
a_list = []
b_list = []
c_list = []
d_list = []
e_list = []
x_list = []
y_list = []
z_list = []
for i in ab_bind_name:
    data = ab_all_result_dict[i]
    input_label = data['result']['subset_ddg']
    input_rbd_index = [data['rbd_index']]*len(input_label)
    input_pre_mutated_seq = [data['result']['subset_pre_mutated_seq']]*len(input_label)
    input_aft_mutated_seq = data['result']['subset_after_mutated_seq']
    input_chain_index = [data['result']['subset_chain_index']]*len(input_label)
    input_same_index = [data['result']['subset_same_index']]*len(input_label)
    input_coordinate = data['subset_alpha_carbon_coordinate']

    a = np.array([np.array(encode_seq_int(''.join(x))).reshape(len(x)) for x in input_pre_mutated_seq])
    b = np.array([np.array(encode_seq_int(''.join(x))).reshape(len(x)) for x in input_aft_mutated_seq])
    a = tf.one_hot(a, 20)
    b = tf.one_hot(b, 20)
    c = np.array([np.array(list(x)).reshape(len(x)) for x in input_rbd_index])
    d = np.array(list(map(lambda x: [int(same) for same in x], input_same_index)))
    x = np.array([input_coordinate[:, 0]]*len(input_label))
    y = np.array([input_coordinate[:, 1]]*len(input_label))
    z = np.array([input_coordinate[:, 2]]*len(input_label))
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    y = np.reshape(y, (y.shape[0], y.shape[1], 1))
    z = np.reshape(z, (z.shape[0], z.shape[1], 1))
    c = np.reshape(c, (c.shape[0], c.shape[1], 1))
    d = np.reshape(d, (d.shape[0], d.shape[1], 1))
    # normalize coordinate
    x = keras.utils.normalize(x, axis=1)
    y = keras.utils.normalize(y, axis=1)
    z = keras.utils.normalize(z, axis=1)
    e = np.array(list(np.array(input_label).reshape(len(input_label), 1)))
    a_list.append(a)
    b_list.append(b)
    c_list.append(c)
    d_list.append(d)
    x_list.append(x)
    y_list.append(y)
    z_list.append(z)
    e_list.append(e)


# skempi data
skempi_a_list = []
skempi_b_list = []
skempi_c_list = []
skempi_d_list = []
skempi_e_list = []
skempi_x_list = []
skempi_y_list = []
skempi_z_list = []
# i = skempi_bind_name[1]
for i in skempi_bind_name:
    data = skempi_all_result_dict[i]
    input_label = data['result']['subset_ddg']
    input_rbd_index = [data['rbd_index']]*len(input_label)
    input_pre_mutated_seq = [data['result']['subset_pre_mutated_seq']]*len(input_label)
    input_aft_mutated_seq = data['result']['subset_after_mutated_seq']
    input_chain_index = [data['result']['subset_chain_index']]*len(input_label)
    input_same_index = [data['result']['subset_same_index']]*len(input_label)
    input_coordinate = data['subset_alpha_carbon_coordinate']

    a = np.array([np.array(encode_seq_int(''.join(x))).reshape(len(x)) for x in input_pre_mutated_seq])
    b = np.array([np.array(encode_seq_int(''.join(x))).reshape(len(x)) for x in input_aft_mutated_seq])
    a = tf.one_hot(a, 20)
    b = tf.one_hot(b, 20)
    c = np.array([np.array(list(x)).reshape(len(x)) for x in input_rbd_index])
    d = np.array(list(map(lambda x: [int(same) for same in x], input_same_index)))
    x = np.array([input_coordinate[:, 0]]*len(input_label))
    y = np.array([input_coordinate[:, 1]]*len(input_label))
    z = np.array([input_coordinate[:, 2]]*len(input_label))
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    y = np.reshape(y, (y.shape[0], y.shape[1], 1))
    z = np.reshape(z, (z.shape[0], z.shape[1], 1))
    c = np.reshape(c, (c.shape[0], c.shape[1], 1))
    d = np.reshape(d, (d.shape[0], d.shape[1], 1))
    # normalize coordinate
    x = keras.utils.normalize(x, axis=1)
    y = keras.utils.normalize(y, axis=1)
    z = keras.utils.normalize(z, axis=1)
    e = np.array(list(np.array(input_label).reshape(len(input_label), 1)))
    skempi_a_list.append(a)
    skempi_b_list.append(b)
    skempi_c_list.append(c)
    skempi_d_list.append(d)
    skempi_x_list.append(x)
    skempi_y_list.append(y)
    skempi_z_list.append(z)
    skempi_e_list.append(e)
# %%

a_list = a_list + skempi_a_list
b_list = b_list + skempi_b_list
c_list = c_list + skempi_c_list
d_list = d_list + skempi_d_list
e_list = e_list + skempi_e_list
x_list = x_list + skempi_x_list
y_list = y_list + skempi_y_list
z_list = z_list + skempi_z_list



# split train and test set
a_list, v_a_list, b_list, v_b_list, c_list, v_c_list, d_list, v_d_list,e_list, v_e_list, x_list, v_x_list, y_list, v_y_list, z_list, v_z_list = \
    train_test_split(a_list,b_list, c_list,d_list,e_list,x_list,y_list,z_list,train_size=0.7, random_state=1)

v_a_list, t_a_list, v_b_list, t_b_list, v_c_list, t_c_list, v_d_list, t_d_list, v_e_list, t_e_list, v_x_list, t_x_list, v_y_list, t_y_list, v_z_list, t_z_list = \
    train_test_split(v_a_list,v_b_list, v_c_list,v_d_list,v_e_list,v_x_list,v_y_list,v_z_list,train_size=0.5, random_state=1)



# %%


random.seed(1)
def get_batch_draw_sample(k, validation = False, test = False):
    if validation:
        for _ in range(k):
            batch_n = len(v_a_list)
            i = random.randint(0, batch_n - 1)
            sample_num = v_a_list[i].shape[0]
            j = random.randint(0, sample_num - 1)
            print(i)
            print(j)

            a = tf.reshape(v_a_list[i][j], [-1, tf.shape(v_a_list[i][j])[0], tf.shape(v_a_list[i][j])[1]])
            b = tf.reshape(v_b_list[i][j], [-1, tf.shape(v_b_list[i][j])[0], tf.shape(v_b_list[i][j])[1]])
            c = v_c_list[i][j].reshape(-1, v_c_list[i][j].shape[0])
            d = v_d_list[i][j].reshape(-1, v_d_list[i][j].shape[0])
            x = v_x_list[i][j].reshape(-1, v_x_list[i][j].shape[0])
            y = v_y_list[i][j].reshape(-1, v_y_list[i][j].shape[0])
            z = v_z_list[i][j].reshape(-1, v_z_list[i][j].shape[0])
            e = v_e_list[i][j]

            yield a, b, c, d, x, y, z, e
    if test:
        for _ in range(k):
            batch_n = len(t_a_list)
            i = random.randint(0, batch_n - 1)
            sample_num = t_a_list[i].shape[0]
            j = random.randint(0, sample_num - 1)
            print(i)
            print(j)

            a = tf.reshape(t_a_list[i][j], [-1, tf.shape(t_a_list[i][j])[0], tf.shape(t_a_list[i][j])[1]])
            b = tf.reshape(t_b_list[i][j], [-1, tf.shape(t_b_list[i][j])[0], tf.shape(t_b_list[i][j])[1]])
            c = t_c_list[i][j].reshape(-1, t_c_list[i][j].shape[0])
            d = t_d_list[i][j].reshape(-1, t_d_list[i][j].shape[0])
            x = t_x_list[i][j].reshape(-1, t_x_list[i][j].shape[0])
            y = t_y_list[i][j].reshape(-1, t_y_list[i][j].shape[0])
            z = t_z_list[i][j].reshape(-1, t_z_list[i][j].shape[0])
            e = t_e_list[i][j]

            yield a, b, c, d, x, y, z, e
    else:
        for _ in range(k):
            batch_n = len(a_list)
            i = random.randint(0, batch_n - 1)
            sample_num = a_list[i].shape[0]
            j = random.randint(0, sample_num - 1)
            print(i)
            print(j)


            a = tf.reshape(a_list[i][j], [-1, tf.shape(a_list[i][j])[0], tf.shape(a_list[i][j])[1]])
            b = tf.reshape(b_list[i][j], [-1, tf.shape(b_list[i][j])[0], tf.shape(b_list[i][j])[1]])
            c = c_list[i][j].reshape(-1, c_list[i][j].shape[0])
            d = d_list[i][j].reshape(-1, d_list[i][j].shape[0])
            x = x_list[i][j].reshape(-1, x_list[i][j].shape[0])
            y = y_list[i][j].reshape(-1, y_list[i][j].shape[0])
            z = z_list[i][j].reshape(-1, z_list[i][j].shape[0])
            e = e_list[i][j]

            yield a, b, c, d, x, y, z, e



#######################################################################################
# %%
# train binding affinity predictor
random_num_binding_aff = random.randint(1, 100000)
for i in [0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009]:
# for i in [0.0001]:
    zero = tf.zeros(1)
    random_num = random.randint(1, 100000)
    print('random_num')
    print(random_num)
    model = build_and_compile_binding_affinity_predictor(lr = i)
    N_EPOCHS = 1
    train_loss = []
    train_acc = []
    csv_logger = tf.keras.callbacks.CSVLogger(
                'metric/{}-{}-binding_affinity_predictor_lr-{}_train0.log'.format(random_num_binding_aff, random_num, i),
                separator=',', append=True)
    for epoch in range(N_EPOCHS):
        print("=" * 50)
        print("training model")
        print(epoch, "/", N_EPOCHS)


        for a, b, c, d, x, y, z, e in get_batch_draw_sample(5000):


            print('e.shape')
            print(e.shape)
            print('zero.shape')
            print(zero.shape)
            print('true value')
            print(e)
            print('predicted value')
            print(model.predict([a, b, c, d, x, y, z]))
            loss = abs(e - model.predict([a, b, c, d, x, y, z]))
            train_loss.append(loss)
            print(loss)
            # print(train_loss)
            print('avg')
            print(sum(train_loss)/len(train_loss))
            train_acc.append(np.sign(e[0]) == np.sign(model.predict([a, b, c, d, x, y, z])[0][0]))
            print("train")
            model.fit([a, b, c, d, x, y, z], e, batch_size=1, epochs=1, validation_data=([a, b, c, d, x, y, z], e), callbacks=[csv_logger])
            print("train_zero")
            model.fit([a, a, c, d, x, y, z], zero, batch_size=1, epochs=1,
                      validation_data=([a, b, c, d, x, y, z], zero))
            # print('train_acc')
            # print(train_acc)

    # validation
    N_EPOCHS = 1
    eval_loss = []
    eval_acc = []
    for epoch in range(N_EPOCHS):
        print("=" * 50)
        print("validating model")
        print(epoch, "/", N_EPOCHS)

        for a, b, c, d, x, y, z, e in get_batch_draw_sample(500, validation=True):
            print("validate")
            score = model.evaluate([a, b, c, d, x, y, z], e, batch_size=1)
            print('true value')
            print(e)
            print('predicted value')
            print(model.predict([a, b, c, d, x, y, z]))
            # loss = (e - model.predict([a, b, c, d, x, y, z]))**2
            loss = abs(e - model.predict([a, b, c, d, x, y, z]))

            eval_loss.append(loss)
            eval_acc.append(np.sign(e[0]) == np.sign(model.predict([a, b, c, d, x, y, z])[0][0]))


    test_loss = []
    test_acc = []
    for epoch in range(N_EPOCHS):
        print("=" * 50)
        print("testing model")
        print(epoch, "/", N_EPOCHS)

        for a, b, c, d, x, y, z, e in get_batch_draw_sample(500, test=True):
            print('test')
            score = model.evaluate([a, b, c, d, x, y, z], e, batch_size=1)
            print('true value')
            print(e)
            print('predicted value')
            print(model.predict([a, b, c, d, x, y, z]))
            # loss = (e - model.predict([a, b, c, d, x, y, z]))**2
            loss = abs(e - model.predict([a, b, c, d, x, y, z]))

            test_loss.append(loss)
            test_acc.append(np.sign(e[0]) == np.sign(model.predict([a, b, c, d, x, y, z])[0][0]))




    print('avg_train_loss')
    print(sum(train_loss)/len(train_loss))
    avg_train = sum(train_loss)/len(train_loss)
    avg_acc_train = sum(train_acc)/len(train_acc)
    print('sample_num')
    print(len(train_loss))
    with open('metric/{}-{}-binding_affinity_predictor_train_avg_loss_lr-{}_train0.pkl'.format(random_num_binding_aff,random_num,i), 'wb') as f:
        pickle.dump(avg_train, f)
    with open('metric/{}-{}-binding_affinity_predictor_train_loss_lr-{}_train0.pkl'.format(random_num_binding_aff,random_num,i), 'wb') as f:
        pickle.dump(train_loss, f)

    print('avg_eval_loss')
    print(sum(eval_loss)/len(eval_loss))
    print('sample_num')
    print(len(eval_loss))
    avg_eval = sum(eval_loss)/len(eval_loss)
    avg_acc_eval = sum(eval_acc) / len(eval_acc)
    with open('metric/{}-{}-binding_affinity_predictor_eval_avg_loss_lr-{}_train0.pkl'.format(random_num_binding_aff,random_num,i), 'wb') as f:
        pickle.dump(avg_eval, f)
    with open('metric/{}-{}-binding_affinity_predictor_eval_loss_lr-{}_train0.pkl'.format(random_num_binding_aff,random_num,i), 'wb') as f:
        pickle.dump(eval_loss, f)


    print('avg_test_loss')
    print(sum(test_loss)/len(test_loss))
    print('sample_num')
    print(len(test_loss))
    avg_test = sum(test_loss)/len(test_loss)
    avg_acc_test = sum(test_acc) / len(test_acc)

    with open('metric/{}-{}-binding_affinity_predictor_test_avg_loss_lr-{}_train0.pkl'.format(random_num_binding_aff,random_num,i), 'wb') as f:
        pickle.dump(avg_test, f)
    with open('metric/{}-{}-binding_affinity_predictor_test_loss_lr-{}_train0.pkl'.format(random_num_binding_aff,random_num,i), 'wb') as f:
        pickle.dump(test_loss, f)

    model.save_weights('models/{}-{}-binding_affinity_predictor_weights_lr-{}-tr-{}-ev-{}-te-{}-tr_acc-{}-ev_acc-{}-te_acc-{}_train0.h5'.format(random_num_binding_aff,random_num,i, avg_train, avg_eval, avg_test, avg_acc_train, avg_acc_eval, avg_acc_test))


# evaluate
for i in range(len(test_a_list_combined)):
    a = test_a_list_combined[i]
    b = test_b_list_combined[i]
    c = test_c_list_combined[i]
    d = test_d_list_combined[i]
    x = test_x_list_combined[i]
    y = test_y_list_combined[i]
    z = test_z_list_combined[i]
    e = test_e_list_combined[i]
    
    print('e')
    print(e)
    print('model.predict')
    print(model.predict([a, b, c, d, x, y, z]))
    print('model.evaluate')
    score = model.evaluate([a, b, c, d, x, y, z], e, batch_size=1)
    # print('model.fit')
    # model.fit([a, b, c, d, x, y, z], e, batch_size=1, epochs=1)
    # print('model.evaluate')
    # score = model.evaluate([a, b, c, d, x, y, z], e, batch_size=1)
    # print('model.fit')
    # model.fit([a, b, c, d, x, y, z], e, batch_size=1, epochs=1)
    # print('model.evaluate')
    # score = model.evaluate([a, b, c, d, x, y, z], e, batch_size=1)
    # a = test_skempi_a_list[i]
    # b = test_skempi_b_list[i]
    # c = test_skempi_c_list[i]
    # d = test_skempi_d_list[i]
    # x = test_skempi_x_list[i]
    # y = test_skempi_y_list[i]
    # z = test_skempi_z_list[i]
    # e = test_skempi_e_list[i]
    #
    # score = model.evaluate([a,b,c,d,x,y,z], e, batch_size=1)
# summarize number of compex of ab-bind

sum_complex = 0
for i in range(len(e_list)):
    sum_complex += len(e_list[i])
