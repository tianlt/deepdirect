print('1')
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
import string

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"


tf.config.run_functions_eagerly(True)
with open('data/skempi_all_result_dict.pkl', 'rb') as f:
    skempi_all_result_dict = pickle.load(f)

with open('data/ab_all_result_dict.pkl', 'rb') as f:
    ab_all_result_dict = pickle.load(f)

ab_name = list(ab_all_result_dict.keys())
skempi_name = list(skempi_all_result_dict.keys())

def encode_seq_int(data= 'HELLOWORLD'):
    aa_string = 'ARNDCEQGHILKMFPSTWYV'
    seq_letter = aa_string
    char_to_int = dict((c, i+1) for i, c in enumerate(seq_letter))
    integer_encoded = [char_to_int[char] for char in data]
    return integer_encoded

#

# %%
ab_bind_name = ab_name.copy()

for i in ab_name:
    if ab_all_result_dict[i]['result']['subset_ddg'].size == 0:
        ab_bind_name.remove(i)

a_list = []
b_list = []
c_list = []
d_list = []
e_list = []
x_list = []
y_list = []
z_list = []
for i in ab_bind_name:
# for i in ['3K2M']:
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
print('len(ab_name)')
print(len(ab_name))

all_list_ab_bind_name = [a_list, b_list, c_list, d_list, x_list, y_list, z_list, e_list]
with open('model_data/all_list_ab_bind_name.pkl', 'wb') as f:
    pickle.dump(all_list_ab_bind_name, f)


def get_batch():
    # batch_n = 1
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


# dataset = tf.data.Dataset.from_generator(get_batch, output_types=tf.float32)

# %%

#
## %%
def build_model(activation, latent_dim = 64, seq_num = 27, out_len = 5,f_num_1 = 64,
                        f_num_2 = 128,f_num_3 = 256,f_num_4 = 512,k_size = 5,drop_ratio = 0.2,
                        dense_1_num = 128,dense_2_num = 64,dense_3_num = 8,dense_4_num = 1):
    # build
    # input and embedding
    Input_pre = keras.Input(shape=(None,20), name='Input_pre') #@@@
    Input_aft = keras.Input(shape=(None,20), name='Input_aft') #@@@

    # Input_pre = keras.Input(shape=(None, ), name='Input_pre') #@@@
    # Input_aft = keras.Input(shape=(None, ), name='Input_aft') #@@@
    # ip = tf.one_hot(input_pre, 20) #@@@
    # ia = tf.one_hot(input_aft, 20) #@@@

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
    x_aft = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True), name='Bidirect_aft')(Input_aft) #@@@

    x_diff = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True), name='Bidirect_diff')(diff_layer)

    x_rbd = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True), name='Bidirect_rbd')(Input_rbd)
    x_same = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True), name='Bidirect_same')(Input_same)
    x_x = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True), name='Bidirect_x')(Input_x)
    x_y = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True), name='Bidirect_y')(Input_y)
    x_z = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True), name='Bidirect_z')(Input_z)


    x_pre = layers.Reshape((-1, x_pre.shape[2], 1))(x_pre)
    x_aft = layers.Reshape((-1, x_aft.shape[2], 1))(x_aft)
    ########################################################################
    x_diff = layers.Reshape((-1, x_diff.shape[2], 1))(x_diff)
    ########################################################################
    x_rbd = layers.Reshape((-1, x_rbd.shape[2], 1))(x_rbd)
    x_same = layers.Reshape((-1, x_same.shape[2], 1))(x_same)
    x_x = layers.Reshape((-1, x_x.shape[2], 1))(x_x)
    x_y = layers.Reshape((-1, x_y.shape[2], 1))(x_y)
    x_z = layers.Reshape((-1, x_z.shape[2], 1))(x_z)
    # Concat = layers.Concatenate(axis=3, name='Concat')([x_pre, x_aft, x_rbd, x_same, x_x, x_y, x_z])
    ########################################################################
    # Concat_1 = layers.Concatenate(axis=3, name='Concat_1')([x_pre, x_aft]) #(none, none, 128, 40)
    # Concat_1 = layers.Concatenate(axis=3, name='Concat_1')([x_rbd, x_same]) #(none, none, 128, 2)
    # Concat_2 = layers.Concatenate(axis=3, name='Concat_2')([x_x, x_y, x_z]) #(none, none, 128, 3)
    Concat_aux = layers.Concatenate(axis=3, name='Concat_aux')([x_rbd, x_same, x_x, x_y, x_z])  # (none, none, 128, 5)

    #
    # Mut_pos_Concat_1 = layers.Bidirectional(layers.LSTM(latent_dim_1, return_sequences=True), name='Mut_pos_Concat_1')(
    #     Concat_1)
    # Mut_pos_Concat_2 = layers.Bidirectional(layers.LSTM(latent_dim_1, return_sequences=True), name='Mut_pos_Concat_2')(
    #     Concat_2)
    # Concat = layers.Concatenate(axis=2, name='Concat')([Mut_pos_Concat_1, Mut_pos_Concat_2])
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
    Dense = layers.LeakyReLU(alpha=0.2)(Dense)
    Dense = layers.Dense(dense_4_num)(Dense)

    if activation == 'sigmoid':
        Dense = layers.Activation(keras.activations.sigmoid)(Dense)
    if activation == 'linear':
        Dense = layers.Activation(keras.activations.linear)(Dense)


    model = keras.models.Model([Input_pre, Input_aft, Input_rbd, Input_same, Input_x, Input_y, Input_z], Dense)
    return model

## %%
discriminator = build_model(activation='wgan')
binding_affinity_predictor = build_model(activation='linear')


## %%
@tf.custom_gradient
def binary_activation(x, threshold=0.5):
    activated_x = K.sigmoid(x)
    binary_activated_x = activated_x > threshold
    binary_activated_x = K.cast_to_floatx(binary_activated_x)
    def grad(upstream):
        return upstream*1
    return binary_activated_x, grad

@tf.custom_gradient
def where_func(x):#(Mutation_2, where, input_pre):
    where = tf.where(tf.equal(x, 1))
    def grad(upstream):
        return upstream*1
    return where, grad


class mutation_layer_block(layers.Layer):
    def __init__(self, mut_1 = 64, mut_2 = 32, mut_3 = 1): # mut_3 get the incompatnle shape of reshape func
        super(mutation_layer_block, self).__init__()
        self.mut_1 = mut_1
        self.mut_2 = mut_2
        self.mut_3 = mut_3

        self.lstm_1 = layers.LSTM(mut_1, return_sequences=True, name='Mutation_1')
        self.lstm_2 = layers.LSTM(mut_2, return_sequences=True, name='Mutation_2')
        self.timedistributed = layers.TimeDistributed(layers.Dense(20, activation='softmax', name='Mutation_3'))

    def call(self, inputs):
        global Mutation_2

        input_pre, input_mut_pos, Mut_pos_layer_3, Mut_pos_layer_3_reg = inputs  # Input_pre, Mut_pos_layer, Embedding_pre, Mut_pos_layer_2

        Mut_pos_layer_out = tf.squeeze(input_mut_pos, axis=2)
        where = tf.where(tf.equal(Mut_pos_layer_out, 1))
        Concat_1 = layers.Concatenate(axis=2, name='Concat_1')([input_pre, Mut_pos_layer_3, Mut_pos_layer_3_reg])
        
        Mutation_1 = self.lstm_1(Concat_1)
        Mutation_2 = self.lstm_2(Mutation_1)
        Mutation_3 = self.timedistributed(Mutation_2)
        Mutation_4 = tf.gather_nd(Mutation_3, where)
        aa_seq_out = tf.tensor_scatter_nd_update(input_pre, [where], [Mutation_4])
        return aa_seq_out



## %%
def build_aa_mutator(latent_dim_1 = 64,latent_dim_2 = 16,latent_dim_3 = 1,seq_num = 27,
            out_len = 5,f_num_1 = 64,f_num_2 = 128,f_num_3 = 256,f_num_4 = 512,k_size = 5,
            drop_ratio = 0.2,dense_1_num = 128,dense_2_num = 64,dense_3_num = 8,dense_4_num = 1):

    Input_pre = keras.Input(shape=(None,20), name='Input_pre') #@@@
    # Input_pre = keras.Input(shape=(None,), name='Input_pre') #@@@
    # ip = tf.one_hot(input_pre, 20)  # @@@
    # Embedding_pre = layers.Embedding(seq_num, out_len, input_length=None, name='Embedding_pre')(Input_pre)
    Input_rbd = keras.Input(shape=(None,1), name='Input_rbd')
    Input_same = keras.Input(shape=(None, 1), name='Input_same')  # added
    Input_x = keras.Input(shape=(None,1), name='Input_x')
    Input_y = keras.Input(shape=(None,1), name='Input_y')
    Input_z = keras.Input(shape=(None,1), name='Input_z')

    Input_noi = keras.Input(shape=(None,1), name='Input_noi')
    # is input the protein or the complex

    x_pre = layers.Bidirectional(layers.LSTM(latent_dim_1, return_sequences=True), name='Bidirect_pre_1')(Input_pre) #@@@
    x_rbd = layers.Bidirectional(layers.LSTM(latent_dim_1, return_sequences=True), name='Bidirect_rbd_1')(Input_rbd)
    x_same = layers.Bidirectional(layers.LSTM(latent_dim_1, return_sequences=True), name='Bidirect_same_1')(Input_same)
    x_x = layers.Bidirectional(layers.LSTM(latent_dim_1, return_sequences=True), name='Bidirect_x_1')(Input_x)
    x_y = layers.Bidirectional(layers.LSTM(latent_dim_1, return_sequences=True), name='Bidirect_y_1')(Input_y)
    x_z = layers.Bidirectional(layers.LSTM(latent_dim_1, return_sequences=True), name='Bidirect_z_1')(Input_z)
    x_noi = layers.Bidirectional(layers.LSTM(latent_dim_1, return_sequences=True), name='Bidirect_noi_1')(Input_noi)
    
    Concat_1 = x_noi
    Concat_2 = layers.Concatenate(axis=2, name='Concat_2')([x_pre, x_rbd, x_same, x_x, x_y, x_z])


    Mut_pos_Concat_1 = layers.Bidirectional(layers.LSTM(latent_dim_1, return_sequences=True), name='Mut_pos_Concat_1')(
        Concat_1)
    Mut_pos_Concat_2 = layers.Bidirectional(layers.LSTM(latent_dim_1, return_sequences=True), name='Mut_pos_Concat_2')(
        Concat_2)
    Concat = layers.Concatenate(axis=2, name='Concat')([Mut_pos_Concat_1, Mut_pos_Concat_2])




    Mut_pos_layer_1 = layers.Bidirectional(layers.LSTM(latent_dim_1, return_sequences=True), name='Mut_pos_layer_1')(Concat)
    Mut_pos_layer_2 = layers.Bidirectional(layers.LSTM(latent_dim_2, return_sequences=True), name='Mut_pos_layer_2')(Mut_pos_layer_1)
    # Mut_pos_layer_3 = layers.LSTM(latent_dim_3, return_sequences=True, name='Mut_pos_layer_3')(Mut_pos_layer_2)
    Mut_pos_layer_3 = layers.Bidirectional(layers.LSTM(10, return_sequences=True), name='Mut_pos_layer_3')(Mut_pos_layer_2)
    Mut_pos_layer_3_reg = layers.Dense(latent_dim_3, name='Mut_pos_layer_3_reg')(Mut_pos_layer_3) #@XXXXX

    # Mut_pos_layer_3_reg = layers.Dense(latent_dim_3, activity_regularizer=Mut_pos_rg, name='Mut_pos_layer_3_reg')(Mut_pos_layer_3) #@XXXXX
    Mut_pos_layer = layers.Dense(latent_dim_3, activation=binary_activation, name='Mut_pos_layer_4')(Mut_pos_layer_3_reg) #@XXXXX
    # tf.print('Mut_pos_layer')
    # tf.print(Mut_pos_layer)

    # Mut_pos_layer = layers.Dense(latent_dim_3, activation=binary_activation, activity_regularizer=Mut_pos_rg,name='Mut_pos_layer_4')(Mut_pos_layer_3)
    # aa_seq_out = mutation_layer_block()([Input_pre, Mut_pos_layer, Mut_pos_layer_3, Mut_pos_layer_3_reg])
    aa_seq_out = mutation_layer_block()([Input_pre, Mut_pos_layer, Mut_pos_layer_3, Mut_pos_layer_3_reg])
    # print('aa_seq_out_in_aa_mutator')
    # print(aa_seq_out.shape)
    model = keras.models.Model([Input_pre, Input_rbd, Input_same, Input_x, Input_y, Input_z, Input_noi], aa_seq_out)
    return model


aa_mutator = build_aa_mutator()

####################################################################################################
# %%



# %%

binding_affinity_predictor = build_model(activation='linear')
aa_mutator = build_aa_mutator()

@tf.custom_gradient
def arg_one_hot(x):
    x = K.argmax(x, axis=-1)
    # print('mutated sequence')
    # print(x)
    # print(x.shape)
    x = tf.one_hot(x, 20)
    def grad(upstream):
        return upstream*1
    return x, grad


class AA_Mutation_GAN(keras.Model):
    def __init__(self, discriminator, generator, binding_affinity_predictor, discriminator_extra_steps = 5, gp_weight=10.0, clip_value = 0.01,
                 discriminator_unchange_training = 3):


        super(AA_Mutation_GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.binding_affinity_predictor = binding_affinity_predictor
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self.d_steps = discriminator_extra_steps
        self.discriminator_unchange_training = discriminator_unchange_training
        self.gp_weight = gp_weight
        self.clip_value = clip_value
        self.d_similarity_ratio_tracker = keras.metrics.Mean(name="d_similarity_ratio")
        self.g_similarity_ratio_tracker = keras.metrics.Mean(name="g_similarity_ratio")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker, self.d_similarity_ratio_tracker, self.g_similarity_ratio_tracker]

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(AA_Mutation_GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        # self.loss_fn = loss_fn
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
    

    ###############################################################################################
    def train_step(self, data):
        # if self.replace_index == False:
        if replace_index == False:
        # input_pre, input_aft, input_rbd, input_same, input_x, input_y, input_z = data
        # print(data)
            input_pre = data[0][0]
            input_aft = data[0][1]
            input_rbd = data[0][2]
            input_same = data[0][3]
            input_x = data[0][4]
            input_y = data[0][5]
            input_z = data[0][6]

            # print('batch number processing')
            # print(input_pre.shape[0])

        # if sample_index <= self.replace_index:
        # if self.replace_index == True:
            batch_size = input_pre.shape[0]
            print('batch_size')
            print(batch_size)
            print('input_aft_in_train_out_step')
            print(input_aft.shape)

            for i in range(self.d_steps):
                input_noi = tf.random.normal(
                    shape=(batch_size, input_pre.shape[1], 1)
                )
                # input_noi = np.random.normal(0, 1, input_pre.shape[0] * input_pre.shape[1])
                # input_noi = np.reshape(input_noi, (input_pre.shape[0], input_pre.shape[1], 1))
                # Train the discriminator.
                with tf.GradientTape() as tape:
                    print('training discriminator step 1')
                    # predictions = self.discriminator([input_pre, mutated_aa, input_rbd, input_same, input_x, input_y, input_z])
                    mutated_aa = self.generator(
                        [input_pre, input_rbd, input_same, input_x, input_y, input_z, input_noi], training=True)  # added same
                    mutated_aa = arg_one_hot(mutated_aa)
                    predictions_fake = self.discriminator([input_pre, mutated_aa, input_rbd, input_same, input_x, input_y, input_z], training=True)
                    predictions_real = self.discriminator([input_pre, input_aft, input_rbd, input_same, input_x, input_y, input_z], training=True)
                    # d_cost = self.d_loss_fn(real=predictions_real, fake=predictions_fake)

                    original_len = tf.reshape(tf.repeat(mutated_aa.shape[1], mutated_aa.shape[0]),
                                              (mutated_aa.shape[0]))
                    similarity = K.sum(
                        K.cast(K.argmax(input_pre, axis=-1) == K.argmax(mutated_aa, axis=-1), tf.int64), axis=-1)

                    d_similarity_ratio = tf.cast(
                        tf.reduce_mean(tf.cast(similarity, tf.int32)) / tf.reduce_mean(tf.cast(original_len, tf.int32)),
                        tf.float32)

                    d_similarity_ratio = d_similarity_ratio.numpy()
                    print('d_similarity_ratio')
                    print(d_similarity_ratio)

                    # d_loss = self.d_loss_fn(real=predictions_real, fake=predictions_fake)
                    d_loss = self.d_loss_fn(real=predictions_real, fake=predictions_fake, original_len=original_len, similarity=similarity)

                    



                d_grads = tape.gradient(d_loss, self.discriminator.trainable_weights)

                self.d_optimizer.apply_gradients(
                    zip(d_grads, self.discriminator.trainable_weights)
                )
                # change
                for layer in self.discriminator.layers:
                    weights = layer.get_weights()
                    weights = [np.clip(weight,
                                       -self.clip_value,
                                       self.clip_value) for weight in weights]
                    layer.set_weights(weights)

                #train input pre is fake###############################################################################################
            for i in range(self.discriminator_unchange_training):

                with tf.GradientTape() as tape:
                    print('training discriminator step 2')
                    # predictions = self.discriminator([input_pre, mutated_aa, input_rbd, input_same, input_x, input_y, input_z])

                    predictions_fake = self.discriminator([input_pre, input_pre, input_rbd, input_same, input_x, input_y, input_z], training=True)
                    predictions_real = self.discriminator([input_pre, input_aft, input_rbd, input_same, input_x, input_y, input_z], training=True)
                    # d_cost = self.d_loss_fn(real=predictions_real, fake=predictions_fake)
                    # d_loss = self.d_loss_fn(real=predictions_real, fake=predictions_fake)

                    original_len = tf.reshape(tf.repeat(input_pre.shape[1], input_pre.shape[0]),
                                              (input_pre.shape[0]))
                    similarity = K.sum(
                        K.cast(K.argmax(input_pre, axis=-1) == K.argmax(input_pre, axis=-1), tf.int64), axis=-1)
                    print('similarity for input_pre as input')
                    print(similarity)
                    # d_loss = self.d_loss_fn(real=predictions_real, fake=predictions_fake)

                    d_loss = self.d_loss_fn(real=predictions_real, fake=predictions_fake, original_len=original_len,
                                            similarity=similarity)

                d_grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
                # print('d_grads')
                # print(d_grads)
                self.d_optimizer.apply_gradients(
                    zip(d_grads, self.discriminator.trainable_weights)
                )

                for layer in self.discriminator.layers:
                    weights = layer.get_weights()
                    weights = [np.clip(weight,
                                       -self.clip_value,
                                       self.clip_value) for weight in weights]
                    layer.set_weights(weights)

                ################################################################################################
            # Train the generator (note that we should *not* update the weights
            # of the discriminator)!
            input_noi = tf.random.normal(
                shape=(batch_size, input_pre.shape[1], 1)
            )
            with tf.GradientTape() as tape:
                print('training generator')
                fake_mutation = self.generator([input_pre, input_rbd, input_same, input_x, input_y, input_z, input_noi])
                # print('reference sequence')
                # print(K.argmax(input_pre, axis=-1))
                # print('fake sequence')
                # print(K.argmax(fake_mutation, axis=-1))
                # print('input_pre')
                # print(input_pre.shape)
                print('similarity')
                # print(K.argmax(input_pre, axis=-1) == K.argmax(fake_mutation, axis=-1))
                similarity = K.sum(K.cast(K.argmax(input_pre, axis=-1) == K.argmax(fake_mutation, axis=-1), tf.int64), axis=-1)
                # print(similarity)
                mut_index = K.cast(K.argmax(input_pre, axis=-1) == K.argmax(fake_mutation, axis=-1), tf.int64)
                # print(mut_index)
                # print(input_pre.shape)
                # print(fake_mutation.shape)
                where = tf.where(tf.equal(mut_index, 0))
                # print('where')
                # print(where)
                # print(K.argmax(input_pre, axis=-1).shape)
                # print(K.argmax(input_pre, axis=-1))
                #
                # print('first dimension')
                # print(K.argmax(input_pre, axis=-1)[0])
                # print(K.argmax(fake_mutation, axis=-1))
                pre_mut = tf.gather_nd(K.argmax(input_pre, axis=-1), where)
                aft_mut = tf.gather_nd(K.argmax(fake_mutation, axis=-1), where)
                print('pre_mut')
                print('aft_mut')
                print(pre_mut)
                print(aft_mut)
                # print(aft_mut.shape)

                dim_num = batch_size
                # for i in range(dim_num):
                #     print('batch_num ' + str(i))
                #     mut_index = K.cast(K.argmax(input_pre, axis=-1)[i] == K.argmax(fake_mutation, axis=-1)[i], tf.int64)
                #     where = tf.where(tf.equal(mut_index, 0))
                #     print('pre_mut')
                #     pre_mut = tf.gather_nd(K.argmax(input_pre, axis=-1)[i], where)
                #     print(pre_mut)
                #     print('aft_mut')
                #     aft_mut = tf.gather_nd(K.argmax(fake_mutation, axis=-1)[i], where)
                #     print(aft_mut)


                fake_aa_seq = arg_one_hot(fake_mutation)
                predictions = self.discriminator([input_pre, fake_aa_seq, input_rbd, input_same, input_x, input_y, input_z])
                # print('fake_mutation.shape')
                # print(fake_mutation.shape)
                # print('fake_mutation.len')
                # print(fake_mutation.shape[1])
                print('mutation_num')
                original_len = tf.reshape(tf.repeat(fake_mutation.shape[1], fake_mutation.shape[0]),
                                          (fake_mutation.shape[0]))
                print(tf.cast(original_len, tf.int32) - tf.cast(similarity, tf.int32))

                g_similarity_ratio = tf.cast(
                    tf.reduce_mean(tf.cast(similarity, tf.int32)) / tf.reduce_mean(tf.cast(original_len, tf.int32)),
                    tf.float32)

                g_similarity_ratio = g_similarity_ratio.numpy()

                print('g_similarity_ratio')
                print(g_similarity_ratio)
                # g_loss = self.loss_fn(misleading_labels, predictions)
                # g_loss = self.g_loss_fn(predictions)
                g_loss = self.g_loss_fn(predictions, original_len, similarity)

            grads = tape.gradient(g_loss, self.generator.trainable_weights)
            # print('fake_mutation.shape')
            # print(fake_mutation.shape)
            # print('fake_mutation.len')
            # print(fake_mutation.shape[1])
            # print('mutation_num')
            # original_len = tf.reshape(tf.repeat(fake_mutation.shape[1], fake_mutation.shape[0]), (fake_mutation.shape[0]))
            # print(tf.cast(original_len, tf.int32) - tf.cast(similarity, tf.int32))

            # print(input_pre.shape)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

            # Monitor loss.
            self.gen_loss_tracker.update_state(g_loss)
            self.disc_loss_tracker.update_state(d_loss)
            self.g_similarity_ratio_tracker.update_state(g_similarity_ratio)
            self.d_similarity_ratio_tracker.update_state(d_similarity_ratio)

        # d_similarity_ratio

        # if sample_index > self.replace_index:
        # if self.replace_index == True:
        if replace_index == True:
            input_pre = data[0][0]
            input_aft = data[0][1]
            input_rbd = data[0][2]
            input_same = data[0][3]
            input_x = data[0][4]
            input_y = data[0][5]
            input_z = data[0][6]

            print('start replacing training set')
            print("=" * 50)
            # print('batch number processing')
            # print(input_pre.shape[0])
            batch_size = input_pre.shape[0]

            input_noi = np.random.normal(0, 1, input_pre.shape[0] * input_pre.shape[1])
            input_noi = np.reshape(input_noi, (input_pre.shape[0], input_pre.shape[1], 1))

            # mutated_aa = self.generator([input_pre, input_rbd, input_x, input_y, input_z, input_noi])
            mutated_aa = self.generator([input_pre, input_rbd, input_same, input_x, input_y, input_z, input_noi])
            binding_score_original = self.binding_affinity_predictor([input_pre, input_aft, input_rbd, input_same, input_x, input_y, input_z])
            binding_score_mutated = self.binding_affinity_predictor([input_pre, mutated_aa, input_rbd, input_same, input_x, input_y, input_z])
            # label and mutated list
            # binding_score_ddg_in_list = []
            # mutated_aa_in_list = []
            # label_in_list = []
            #
            # binding_score_ddg_de_list = []
            # mutated_aa_de_list = []
            # label_de_list = []
            # in the situation binding affinity increased
            # batch also set as 1
            ddg = binding_score_mutated - binding_score_original
            # if self.select == 'increase':
            if select == 'increase':
                if ddg > 0:

                    input_aft, mutated_aa = mutated_aa, input_aft
                    # label_in_list.append(1)
                    # mutated_aa_in_list.append(mutated_aa)
                    # binding_score_ddg_in_list.append(ddg)
            # in the situation binding affinity decreased
            # if self.select == 'decrease':
            if select == 'decrease':
                if ddg <= 0:
                    input_aft, mutated_aa = mutated_aa, input_aft
                    # label_de_list.append(1)
                    # mutated_aa_de_list.append(mutated_aa)
                    # binding_score_ddg_de_list.append(ddg)

            # labels and concate training data
            labels_one = tf.ones((batch_size, 1), tf.int64)
            labels_zero = tf.zeros((batch_size, 1), tf.int64)
            labels = tf.concat([labels_one, labels_zero], 0)

            input_pre_dis = tf.concat([input_pre, input_pre], 0)
            mutated_aa_dis = tf.concat([input_aft, mutated_aa], 0)
            input_rbd_dis = tf.concat([input_rbd, input_rbd], 0)
            input_same_dis = tf.concat([input_same, input_same], 0)
            input_x_dis = tf.concat([input_x, input_x], 0)
            input_y_dis = tf.concat([input_y, input_y], 0)
            input_z_dis = tf.concat([input_z, input_z], 0)


            # Train the discriminator.
            with tf.GradientTape() as tape:
                print('training discriminator')
                # predictions = self.discriminator([input_pre, mutated_aa, input_rbd, input_same, input_x, input_y, input_z])
                predictions = self.discriminator([input_pre_dis, mutated_aa_dis, input_rbd_dis, input_same_dis, input_x_dis, input_y_dis,input_z_dis])

                d_loss = self.loss_fn(labels, predictions)
            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
            )
            # batch_size = tf.shape(input_pre)[0]
            misleading_labels = tf.ones((batch_size, 1), tf.int64)

            # Train the generator (note that we should *not* update the weights
            # of the discriminator)!
            with tf.GradientTape() as tape:
                print('training generator')
                # fake_mutation = self.generator([input_pre, input_rbd, input_x, input_y, input_z, input_noi])
                fake_mutation = self.generator([input_pre, input_rbd, input_same, input_x, input_y, input_z, input_noi]) #added same
                print('similarity_replace')
                # print(K.argmax(input_pre, axis=-1) == K.argmax(fake_mutation, axis=-1))
                print(K.sum(K.cast(K.argmax(input_pre, axis=-1) == K.argmax(fake_mutation, axis=-1), tf.int64), axis=-1))
                # print('fake_mutation')
                # print(fake_mutation)
                # print(fake_mutation.shape)
                fake_aa_seq = arg_one_hot(fake_mutation)
                predictions = self.discriminator(
                    [input_pre, fake_aa_seq, input_rbd, input_same, input_x, input_y, input_z])

                # print('misleading_labels')
                # print(misleading_labels)

                g_loss = self.loss_fn(misleading_labels, predictions)

            grads = tape.gradient(g_loss, self.generator.trainable_weights)
            print('fake_mutation.shape_replace')
            print(fake_mutation.shape)
            # print(fake_mutation.shape)
            # print(input_pre.shape)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

            # Monitor loss.
            self.gen_loss_tracker.update_state(g_loss)
            self.disc_loss_tracker.update_state(d_loss)


        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
            "d_ratio": self.d_similarity_ratio_tracker.result(),
            "g_ratio": self.g_similarity_ratio_tracker.result(),
            # "d_ratio": d_similarity_ratio,
            # "g_ratio": g_similarity_ratio,

        }

# Define the loss functions for the discriminator,
# which should be (fake_loss - real_loss).
def discriminator_loss(real, fake, original_len, similarity):
    mut_num_reduced = tf.reduce_mean(tf.cast(original_len, tf.int32) - tf.cast(similarity, tf.int32))
    # penalty = tf.cast(tf.reduce_mean(tf.cast(similarity, tf.int32))/tf.reduce_mean(tf.cast(original_len, tf.int32)), tf.float32)
    print('mut_num_reduced_d')
    print(mut_num_reduced)
    # print('penalty_d')
    # print(penalty)
    real_loss = tf.reduce_mean(real)
    fake_loss = tf.reduce_mean(fake)



    # return fake_loss - real_loss + 6*penalty
    return 10000*(fake_loss - real_loss)


# Define the loss functions for the generator.
def generator_loss(fake, original_len, similarity):
    mut_num_reduced = tf.reduce_mean(tf.cast(original_len, tf.int32) - tf.cast(similarity, tf.int32))
    similarity_ratio = tf.cast(tf.reduce_mean(tf.cast(similarity, tf.int32))/tf.reduce_mean(tf.cast(original_len, tf.int32)), tf.float32)
    mut_ratio = 0.06
    print('similarity_ratio')
    print(similarity_ratio)
    # penalty = (similarity_ratio - mut_ratio) ** 2
    penalty = 4 * (-(similarity_ratio - 0.8)) ** 4 + 1 * (similarity_ratio - 0.8) + 0.2 + similarity_ratio ** 2
    print('mut_num_reduced_g')
    print(mut_num_reduced)
    print('penalty_g')
    print(penalty)
    print('tf.reduce_mean(fake)')
    print(tf.reduce_mean(fake))
    return -50*(tf.reduce_mean(fake) - penalty / 5) #50
    
AA_Mutation_Gan = AA_Mutation_GAN(
        discriminator=discriminator, generator=aa_mutator, binding_affinity_predictor=binding_affinity_predictor,
        clip_value=0.1,
        discriminator_unchange_training=1
    )
old_weights = AA_Mutation_Gan.get_weights()
def build_and_compile_AA_Mutation_Gan(dlr, glr):
    AA_Mutation_Gan = AA_Mutation_GAN(
        discriminator=discriminator, generator=aa_mutator, binding_affinity_predictor=binding_affinity_predictor,
        clip_value=0.1,
        discriminator_unchange_training=1
    )
    AA_Mutation_Gan.set_weights(old_weights)

    AA_Mutation_Gan.compile(
        # d_optimizer=keras.optimizers.Adam(learning_rate=0.00008),
        # g_optimizer=keras.optimizers.Adam(learning_rate=0.00008),
        d_optimizer=keras.optimizers.Adam(learning_rate=dlr),
        g_optimizer=keras.optimizers.Adam(learning_rate=glr),
        # d_optimizer=keras.optimizers.RMSprop(learning_rate=5e-4),
        # g_optimizer=keras.optimizers.RMSprop(learning_rate=5e-4),
        # loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss,
    )
    return AA_Mutation_Gan

batch_end_d_loss_1 = list()
batch_end_g_loss_1 = list()
batch_index_1 = list()
class SaveBatchLoss_1(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        batch_end_d_loss_1.append(logs['d_loss'])
        batch_end_g_loss_1.append(logs['g_loss'])
        batch_index_1.append(batch)

batch_end_d_loss_2 = list()
batch_end_g_loss_2 = list()
batch_index_2 = list()
class SaveBatchLoss_2(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        batch_end_d_loss_2.append(logs['d_loss'])
        batch_end_g_loss_2.append(logs['g_loss'])
        batch_index_2.append(batch)

N_EPOCHS = 1
sample_index_1 = 0
replace_index = False

# parameter tuning and save model
d_similarity_ratio_list = []
g_similarity_ratio_list = []
i_para_list = []
j_para_list = []
batch_size_list = []
num_list = []
random_num = random.randint(1,100000)
for num in range(1):
    # for i in np.arange(0.00003, 0.00034, 0.0001):
    #     for j in np.arange(0.00003, 0.00034, 0.0001):
    for i in [3.0e-04, 3.0e-05, 9.0e-05, 3.0e-06]:
        for j in [3.0e-04, 3.0e-05, 9.0e-05, 3.0e-06]:
            # K.set_value(AA_Mutation_Gan.d_optimizer.learning_rate, i)
            # K.set_value(AA_Mutation_Gan.g_optimizer.learning_rate, j)
            for batch_size in [5, 10]:
            # for batch_size in [10]:
                AA_Mutation_Gan = build_and_compile_AA_Mutation_Gan(i, j)
                print(AA_Mutation_Gan.discriminator.weights)
                for epoch in range(N_EPOCHS):
                    print("=" * 50)
                    print(epoch, "/", N_EPOCHS)
                    for a,b,c,d,x,y,z,e in get_batch():
                        history_1 = AA_Mutation_Gan.fit([a,b,c,d,x,y,z], batch_size=batch_size, shuffle=True, callbacks=[SaveBatchLoss_1()])

                    print(AA_Mutation_Gan.discriminator.weights)

                    d_result = AA_Mutation_Gan.d_similarity_ratio_tracker.result()
                    g_result = AA_Mutation_Gan.g_similarity_ratio_tracker.result()
                    d_similarity_ratio_list.append(d_result)
                    g_similarity_ratio_list.append(g_result)
                    i_para_list.append(i)
                    j_para_list.append(j)
                    batch_size_list.append(batch_size)
                    num_list.append(num)
                    discriminator_step_1 = AA_Mutation_Gan.discriminator
                    generator_step_1 = AA_Mutation_Gan.generator

                    discriminator_step_1.save_weights('models/{}-discriminator_step_1_tf-num-{}-dlr-{:.6f}-glr-{:.6f}-drt-{:.3f}-grt-{:.3f}-b-{}_bap_weights.h5'.format(
                                                   random_num, num, i, j, d_result, g_result, batch_size))
                    generator_step_1.save_weights(
                        'models/{}-generator_step_1_tf-num-{}-dlr-{:.6f}-glr-{:.6f}-drt-{:.3f}-grt-{:.3f}-b-{}_bap_weights.h5'.format(
                            random_num, num, i, j, d_result, g_result, batch_size))


                    # tf.keras.models.save_model(discriminator_step_1, 'models/{}-discriminator_step_1_tf-num-{}-dlr-{:.6f}-glr-{:.6f}-drt-{:.3f}-grt-{:.3f}-b-{}_bap'.format(random_num,num,i,j,d_result,g_result, batch_size))
                    # tf.keras.models.save_model(generator_step_1, 'models/{}-generator_step_1_tf-num-{}-dlr-{:.6f}-glr-{:.6f}-drt-{:.3f}-grt-{:.3f}-b-{}_bap'.format(random_num,num,i,j,d_result,g_result, batch_size))

end


# fit for the second phase
# discriminator_step_2 = keras.models.load_model('models/13036-discriminator_step_1_tf-num-0-dlr-0.000003-glr-0.000003-drt-0.961-grt-0.958-b-5_bap')
# generator_step_2 = keras.models.load_model('models/13036-generator_step_1_tf-num-0-dlr-0.000003-glr-0.000003-drt-0.961-grt-0.958-b-5_bap')
# keras.models.load_model('models/13036-generator_step_1_tf-num-0-dlr-0.000003-glr-0.000003-drt-0.961-grt-0.958-b-5_bap')
# 
# AA_Mutation_Gan = AA_Mutation_GAN(
#         discriminator=discriminator_step_2, generator=generator_step_2, binding_affinity_predictor=binding_affinity_predictor,
#         clip_value=0.1,
#         discriminator_unchange_training=1
#     )
# 
# AA_Mutation_Gan.compile(
#     d_optimizer=keras.optimizers.Adam(learning_rate=0.00008),
#     g_optimizer=keras.optimizers.Adam(learning_rate=0.00008),
#     # d_optimizer=keras.optimizers.Adam(learning_rate=dlr),
#     # g_optimizer=keras.optimizers.Adam(learning_rate=glr),
#     # d_optimizer=keras.optimizers.RMSprop(learning_rate=5e-4),
#     # g_optimizer=keras.optimizers.RMSprop(learning_rate=5e-4),
#     # loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
#     g_loss_fn=generator_loss,
#     d_loss_fn=discriminator_loss,
# )
# 
# select = 'increase'
# replace_index = True
# for a, b, c, d, x, y, z, e in get_batch():
#     history_1 = AA_Mutation_Gan.fit([a, b, c, d, x, y, z], batch_size=batch_size, shuffle=True,
#                                     callbacks=[SaveBatchLoss_1()])
# 
# d_similarity_ratio_list = []
# g_similarity_ratio_list = []
# i_para_list = []
# j_para_list = []
# batch_size_list = []
# num_list = []
# random_num = random.randint(1,100000)
# for num in range(1):
#     # for i in np.arange(0.00003, 0.00034, 0.0001):
#     #     for j in np.arange(0.00003, 0.00034, 0.0001):
#     for i in [3.0e-04, 3.0e-05, 9.0e-05, 3.0e-06]:
#         for j in [3.0e-04, 3.0e-05, 9.0e-05, 3.0e-06]:
#             # K.set_value(AA_Mutation_Gan.d_optimizer.learning_rate, i)
#             # K.set_value(AA_Mutation_Gan.g_optimizer.learning_rate, j)
#             for batch_size in [5, 10]:
#             # for batch_size in [10]:
#                 AA_Mutation_Gan = build_and_compile_AA_Mutation_Gan(i, j)
#                 print(AA_Mutation_Gan.discriminator.weights)
#                 for epoch in range(N_EPOCHS):
#                     print("=" * 50)
#                     print(epoch, "/", N_EPOCHS)
#                     for a,b,c,d,x,y,z,e in get_batch():
#                         history_1 = AA_Mutation_Gan.fit([a,b,c,d,x,y,z], batch_size=batch_size, shuffle=True, callbacks=[SaveBatchLoss_1()])
# 
#                     print(AA_Mutation_Gan.discriminator.weights)
# 
#                     d_result = AA_Mutation_Gan.d_similarity_ratio_tracker.result()
#                     g_result = AA_Mutation_Gan.g_similarity_ratio_tracker.result()
#                     d_similarity_ratio_list.append(d_result)
#                     g_similarity_ratio_list.append(g_result)
#                     i_para_list.append(i)
#                     j_para_list.append(j)
#                     batch_size_list.append(batch_size)
#                     num_list.append(num)
#                     discriminator_step_1 = AA_Mutation_Gan.discriminator
#                     generator_step_1 = AA_Mutation_Gan.generator
# 
#                     tf.keras.models.save_model(discriminator_step_1, 'models/{}-discriminator_step_1_tf-num-{}-dlr-{:.6f}-glr-{:.6f}-drt-{:.3f}-grt-{:.3f}-b-{}_bap'.format(random_num,num,i,j,d_result,g_result, batch_size))
#                     tf.keras.models.save_model(generator_step_1, 'models/{}-generator_step_1_tf-num-{}-dlr-{:.6f}-glr-{:.6f}-drt-{:.3f}-grt-{:.3f}-b-{}_bap'.format(random_num,num,i,j,d_result,g_result, batch_size))
# 
# 


#
#
#
# with open('models/d_similarity_ratio_list-{}.pkl'.format(random_num), 'wb') as f:
#     pickle.dump(d_similarity_ratio_list, f)
# with open('models/g_similarity_ratio_list-{}.pkl'.format(random_num), 'wb') as f:
#     pickle.dump(g_similarity_ratio_list, f)
# with open('models/i_para_list-{}.pkl'.format(random_num), 'wb') as f:
#     pickle.dump(i_para_list, f)
# with open('models/j_para_list-{}.pkl'.format(random_num), 'wb') as f:
#     pickle.dump(j_para_list, f)
# with open('models/batch_size_list-{}.pkl'.format(random_num), 'wb') as f:
#     pickle.dump(batch_size_list, f)
# with open('models/num_list-{}.pkl'.format(random_num), 'wb') as f:
#     pickle.dump(num_list, f)

# dlr = [3.0e-04, 9.0e-05, 9.0e-05]
# glr = [9.0e-05, 9.0e-05, 3.0e-05]
# batch = [5, 5, 5]
# d_similarity_ratio_list = []
# g_similarity_ratio_list = []
# i_para_list = []
# j_para_list = []
# batch_size_list = []
# num_list = []
# random_num = random.randint(1,100000)
#
# for num in range(1):
#     # for i in np.arange(0.00003, 0.00034, 0.0001):
#     #     for j in np.arange(0.00003, 0.00034, 0.0001):
#     for k in range(len(dlr)):
#         i = dlr[k]
#         j = glr[k]
#         batch_size = batch[k]
#         AA_Mutation_Gan = build_and_compile_AA_Mutation_Gan(i, j)
#         print(AA_Mutation_Gan.discriminator.weights)
#         for epoch in range(N_EPOCHS):
#             print("=" * 50)
#             print(epoch, "/", N_EPOCHS)
#             for a,b,c,d,x,y,z,e in get_batch():
#                 history_1 = AA_Mutation_Gan.fit([a,b,c,d,x,y,z], batch_size=batch_size, shuffle=True, callbacks=[SaveBatchLoss_1()])
#
#             print(AA_Mutation_Gan.discriminator.weights)
#
#             d_result = AA_Mutation_Gan.d_similarity_ratio_tracker.result()
#             g_result = AA_Mutation_Gan.g_similarity_ratio_tracker.result()
#             d_similarity_ratio_list.append(d_result)
#             g_similarity_ratio_list.append(g_result)
#             i_para_list.append(i)
#             j_para_list.append(j)
#             batch_size_list.append(batch_size)
#             num_list.append(num)
#             discriminator_step_1 = AA_Mutation_Gan.discriminator
#             generator_step_1 = AA_Mutation_Gan.generator
#
#             tf.keras.models.save_model(discriminator_step_1, 'models/{}-discriminator_step_1_tf-num-{}-dlr-{:.6f}-glr-{:.6f}-drt-{:.3f}-grt-{:.3f}-b-{}'.format(random_num,num,i,j,d_result,g_result, batch_size))
#             tf.keras.models.save_model(generator_step_1, 'models/{}-generator_step_1_tf-num-{}-dlr-{:.6f}-glr-{:.6f}-drt-{:.3f}-grt-{:.3f}-b-{}'.format(random_num,num,i,j,d_result,g_result, batch_size))

# end
# for epoch in range(N_EPOCHS):
#     print("=" * 50)
#     print(epoch, "/", N_EPOCHS)
#     # history_1 = AA_Mutation_Gan.fit([flatten(a_list), flatten(b_list), flatten(c_list), flatten(d_list), flatten(x_list), flatten(y_list), flatten(z_list)], batch_size=3, shuffle=True, callbacks=[SaveBatchLoss_1()])
#     # acc = []
#     for a,b,c,d,x,y,z,e in get_batch():
#         # sample_index_1 = sample_index_1 + 1
#         history_1 = AA_Mutation_Gan.fit([a,b,c,d,x,y,z], batch_size=5, shuffle=True, callbacks=[SaveBatchLoss_1()])
#         # score = AA_Mutation_Gan.evaluate(a,b,c,d,x,y,z)
#         # print("score:", score)
#         # print(a.shape)


# tf.keras.models.save_model(AA_Mutation_Gan.discriminator, './models/discriminator_step_1')
# tf.keras.models.save_model(AA_Mutation_Gan.generator, './models/generator_step_1')
# discriminator_step_1.save('models/discriminator_step_1')
# discriminator_step_1 = AA_Mutation_Gan.discriminator
# generator_step_1 = AA_Mutation_Gan.generator
# 
# 
# tf.keras.models.save_model(discriminator_step_1, 'models/discriminator_step_1_tf')
# tf.keras.models.save_model(generator_step_1, 'models/generator_step_1_tf')
# a = keras.models.load_model('models/discriminator_step_1_tf')

# print(type(AA_Mutation_Gan.discriminator))
# with open('models/discriminator_step_1.pkl', 'wb') as f:
#     pickle.dump(discriminator_step_1, f)
#
# with open('models/generator_step_1.pkl', 'wb') as f:
#     pickle.dump(generator_step_1, f)
# discriminator_step_1.save('models/discriminator_step_1')
# tf.keras.models.save_model(AA_Mutation_Gan.binding_affinity_predictor, './models/binding_affinity_predictor_step_1')

# AA_Mutation_Gan.fit(dataset, epochs=10)
# tf.saved_model.save(AA_Mutation_Gan, 'AA_Mutation_Gan_step_1')
# tf.saved_model.load('AA_Mutation_Gan_step_1')
# AA_Mutation_Gan.save('./models/AA_Mutation_Gan_step_1')
# AA_Mutation_Gan.compute_output_shape(input_shape=(None, 256, 256, 3))
# tf.keras.models.save_model(AA_Mutation_Gan, './models/AA_Mutation_Gan_step_1')
# with open('AA_Mutation_Gan_step_1.pkl', 'wb') as f:
#     pickle.dump(AA_Mutation_Gan, f)
# pd.DataFrame(history_1.history).to_pickle('history_1.pkl')
# ax = pd.DataFrame(history_1.history).plot()
# fig = ax.get_figure()
# fig.savefig('plot.png')

# print("=" * 50)
# select = 'increase'
# replace_index = True
# 
# 
# for epoch in range(N_EPOCHS):
#     print("=" * 50)
#     print(epoch, "/", N_EPOCHS)
#     # acc = []
#     for a,b,c,d,x,y,z,e in get_batch():
#         # sample_index = sample_index + 1
#         history_2 = AA_Mutation_Gan.fit([a,b,c,d,x,y,z], batch_size=1, shuffle=True, callbacks=[SaveBatchLoss_2()])
# 
# 
# with open('batch_end_d_loss_1.pkl', 'wb') as f:
#     pickle.dump(batch_end_d_loss_1, f)
# with open('batch_end_g_loss_1.pkl', 'wb') as f:
#     pickle.dump(batch_end_g_loss_1, f)
# with open('batch_end_d_loss_2.pkl', 'wb') as f:
#     pickle.dump(batch_end_d_loss_2, f)
# with open('batch_end_g_loss_2.pkl', 'wb') as f:
#     pickle.dump(batch_end_g_loss_2, f)
# with open('batch_index_1.pkl', 'wb') as f:
#     pickle.dump(batch_index_1, f)
# with open('batch_index_2.pkl', 'wb') as f:
#     pickle.dump(batch_index_2, f)