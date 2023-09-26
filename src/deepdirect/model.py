import tensorflow as tf
# import matplotlib.pyplot as plt

# import datetime
# import pandas as pd
# import os
# from collections import OrderedDict
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras.layers import Layer
# from sklearn.model_selection import train_test_split
# from nltk.tokenize import word_tokenize
# from tensorflow.keras.preprocessing.text import one_hot
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import random
# import time
# from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K
# import pickle
import numpy as np
# import string

def encode_seq_int(data= 'HELLOWORLD'):
    aa_string = 'ARNDCEQGHILKMFPSTWYV'
    seq_letter = aa_string
    char_to_int = dict((c, i+1) for i, c in enumerate(seq_letter))
    integer_encoded = [char_to_int[char] for char in data]
    return integer_encoded

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
    Dense = layers.LeakyReLU(alpha=0.2)(Dense)
    Dense = layers.Dense(dense_4_num)(Dense)

    if activation == 'sigmoid':
        Dense = layers.Activation(keras.activations.sigmoid)(Dense)
    if activation == 'linear':
        Dense = layers.Activation(keras.activations.linear)(Dense)


    model = keras.models.Model([Input_pre, Input_aft, Input_rbd, Input_same, Input_x, Input_y, Input_z], Dense)
    return model

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

    # def get_config(self):
    #     config = super(mutation_layer_block, self).get_config()
    #     config.update({"mut_1": self.mut_1, "mut_2": self.mut_2, "mut_3": self.mut_3})
    #     return config

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

    # Input_pre = keras.Input(shape=(None,), name='Input_pre')
    Input_pre = keras.Input(shape=(None,20), name='Input_pre') #@@@
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
    Mut_pos_layer_3 = layers.Bidirectional(layers.LSTM(10, return_sequences=True), name='Mut_pos_layer_3')(Mut_pos_layer_2)
    Mut_pos_layer_3_reg = layers.Dense(latent_dim_3, name='Mut_pos_layer_3_reg')(Mut_pos_layer_3) #@XXXXX
    Mut_pos_layer = layers.Dense(latent_dim_3, activation=binary_activation, name='Mut_pos_layer_4')(Mut_pos_layer_3_reg) #@XXXXX
    aa_seq_out = mutation_layer_block()([Input_pre, Mut_pos_layer, Mut_pos_layer_3, Mut_pos_layer_3_reg])

    model = keras.models.Model([Input_pre, Input_rbd, Input_same, Input_x, Input_y, Input_z, Input_noi], aa_seq_out)
    return model

@tf.custom_gradient
def arg_one_hot(x):
    x = K.argmax(x, axis=-1)
    x = tf.one_hot(x, 20)
    def grad(upstream):
        return upstream*1
    return x, grad


class AA_Mutation_GAN(keras.Model):
    def __init__(self, discriminator, generator, binding_affinity_predictor, select, replace_index,
                 discriminator_extra_steps = 5, discriminator_replace_extra_steps = 5, gp_weight=10.0, clip_value = 0.01,
                 clip_replace_value = 0.01, discriminator_unchange_training = 3, generate_seq_num = 5):


        super(AA_Mutation_GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.binding_affinity_predictor = binding_affinity_predictor
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self.d_steps = discriminator_extra_steps
        self.d_replace_steps = discriminator_replace_extra_steps
        self.discriminator_unchange_training = discriminator_unchange_training
        self.generate_seq_num = generate_seq_num

        self.gp_weight = gp_weight
        self.clip_value = clip_value
        self.clip_replace_value = clip_replace_value
        self.d_similarity_ratio_tracker = keras.metrics.Mean(name="d_similarity_ratio")
        self.g_similarity_ratio_tracker = keras.metrics.Mean(name="g_similarity_ratio")
        self.d_binding_aff_tracker = keras.metrics.Mean(name="d_binding_aff")
        self.g_binding_aff_tracker = keras.metrics.Mean(name="g_binding_aff")
        self.replace_index = replace_index
        self.select = select
    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker,
                self.d_similarity_ratio_tracker, self.g_similarity_ratio_tracker,
                self.d_binding_aff_tracker, self.g_binding_aff_tracker]
        # return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(AA_Mutation_GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        # self.loss_fn = loss_fn
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def train_step(self, data):
        # if self.replace_index == False:
        if self.replace_index == False:
        # input_pre, input_aft, input_rbd, input_same, input_x, input_y, input_z = data
        # print(data)
            input_pre = data[0][0]
            input_aft = data[0][1]
            input_rbd = data[0][2]
            input_same = data[0][3]
            input_x = data[0][4]
            input_y = data[0][5]
            input_z = data[0][6]

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

                # Train the discriminator.
                with tf.GradientTape() as tape:
                    print('training discriminator step 1')
                    mutated_aa = self.generator(
                        [input_pre, input_rbd, input_same, input_x, input_y, input_z, input_noi], training=True)  # added same
                    mutated_aa = arg_one_hot(mutated_aa)
                    predictions_fake = self.discriminator([input_pre, mutated_aa, input_rbd, input_same, input_x, input_y, input_z], training=True)
                    predictions_real = self.discriminator([input_pre, input_aft, input_rbd, input_same, input_x, input_y, input_z], training=True)

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
                    d_loss = self.d_loss_fn(real=predictions_real, fake=predictions_fake, original_len=original_len, similarity=similarity)

                d_grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
                self.d_optimizer.apply_gradients(
                    zip(d_grads, self.discriminator.trainable_weights)
                )

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
                    predictions_fake = self.discriminator([input_pre, input_pre, input_rbd, input_same, input_x, input_y, input_z], training=True)
                    predictions_real = self.discriminator([input_pre, input_aft, input_rbd, input_same, input_x, input_y, input_z], training=True)


                    original_len = tf.reshape(tf.repeat(input_pre.shape[1], input_pre.shape[0]),
                                              (input_pre.shape[0]))
                    similarity = K.sum(
                        K.cast(K.argmax(input_pre, axis=-1) == K.argmax(input_pre, axis=-1), tf.int64), axis=-1)

                    d_loss = self.d_loss_fn(real=predictions_real, fake=predictions_fake, original_len=original_len,
                                            similarity=similarity)

                d_grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
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

                print('similarity')
                similarity = K.sum(K.cast(K.argmax(input_pre, axis=-1) == K.argmax(fake_mutation, axis=-1), tf.int64), axis=-1)
                fake_aa_seq = arg_one_hot(fake_mutation)
                predictions = self.discriminator([input_pre, fake_aa_seq, input_rbd, input_same, input_x, input_y, input_z])
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
                g_loss = self.g_loss_fn(predictions, original_len, similarity)

            grads = tape.gradient(g_loss, self.generator.trainable_weights)

            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

            # Monitor loss.
            self.gen_loss_tracker.update_state(g_loss)
            self.disc_loss_tracker.update_state(d_loss)
            self.g_similarity_ratio_tracker.update_state(g_similarity_ratio)
            self.d_similarity_ratio_tracker.update_state(d_similarity_ratio)

        # d_similarity_ratio

        # if sample_index > self.replace_index:
#######################################################################################################################################
        if self.replace_index == True:
            input_pre = data[0][0]
            input_aft = data[0][1]
            input_rbd = data[0][2]
            input_same = data[0][3]
            input_x = data[0][4]
            input_y = data[0][5]
            input_z = data[0][6]


            print('start replacing training set')
            print("=" * 100)
            batch_size = input_pre.shape[0]
            for i in range(self.d_replace_steps):
                print("=" * 25)
                input_noi = np.random.normal(0, 1, input_pre.shape[0] * input_pre.shape[1])
                input_noi = np.reshape(input_noi, (input_pre.shape[0], input_pre.shape[1], 1))

                mutated_aa = self.generator([input_pre, input_rbd, input_same, input_x, input_y, input_z, input_noi])
                mutated_aa_seq = arg_one_hot(mutated_aa)
                binding_score_mutated = self.binding_affinity_predictor([input_pre, mutated_aa_seq, input_rbd, input_same, input_x, input_y, input_z])
                print('binding_score_mutated in discriminator')
                print(binding_score_mutated)
                if self.select == 'increase':
                    if binding_score_mutated > 0:
                        label = tf.zeros((batch_size, 1)) #real
                        ano_label = tf.ones((batch_size, 1))
                        ano_data = tf.random.uniform([batch_size, 1], minval=-2, maxval=0)
                    else:
                        label = tf.ones((batch_size, 1))
                        ano_label = tf.zeros((batch_size, 1))
                        ano_data = tf.random.uniform([batch_size, 1], minval=0, maxval=2)

                if self.select == 'decrease':
                    if binding_score_mutated < 0:
                        label = tf.zeros((batch_size, 1))# real
                        ano_label = tf.ones((batch_size, 1))
                        ano_data = tf.random.uniform([batch_size, 1], minval=0, maxval=2)

                    else:
                        label = tf.ones((batch_size, 1))
                        ano_label = tf.zeros((batch_size, 1))
                        ano_data = tf.random.uniform([batch_size, 1], minval=-2, maxval=0)

                ###########################################################################################################
                original_len = tf.reshape(tf.repeat(mutated_aa_seq.shape[1], mutated_aa_seq.shape[0]),
                                          (mutated_aa_seq.shape[0]))

                similarity = K.sum(
                    K.cast(K.argmax(input_pre, axis=-1) == K.argmax(mutated_aa_seq, axis=-1), tf.int64), axis=-1)
                print('similarity')
                print(similarity)
                d_similarity_ratio = tf.cast(
                    tf.reduce_mean(tf.cast(similarity, tf.int32)) / tf.reduce_mean(tf.cast(original_len, tf.int32)),
                    tf.float32)

                d_similarity_ratio = d_similarity_ratio.numpy()
                d_aff = binding_score_mutated.numpy()

                ###########################################################################################################

                # Train the discriminator.
                with tf.GradientTape() as tape:
                    print('training discriminator')

                    score = self.discriminator([binding_score_mutated])
                    d_loss = self.d_loss_fn(label, score)

                grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
                self.d_optimizer.apply_gradients(
                    zip(grads, self.discriminator.trainable_weights)
                )
                #######################################
                with tf.GradientTape() as tape:
                    score = self.discriminator([ano_data])
                    d_loss = self.d_loss_fn(ano_label, score)

                grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
                self.d_optimizer.apply_gradients(
                    zip(grads, self.discriminator.trainable_weights)
                )
                ####################################################


            print("=" * 50)
            for i in range(self.generate_seq_num):
                misleading_labels = tf.zeros((batch_size, 1))
                with tf.GradientTape() as tape:
                    print('training generator ' + str(i))
                    input_noi = np.random.normal(0, 1, input_pre.shape[0] * input_pre.shape[1])
                    input_noi = np.reshape(input_noi, (input_pre.shape[0], input_pre.shape[1], 1))
                    fake_mutation = self.generator([input_pre, input_rbd, input_same, input_x, input_y, input_z, input_noi]) #added same
                    fake_aa_seq = arg_one_hot(fake_mutation)
                    binding_score_mutated = self.binding_affinity_predictor(
                        [input_pre, fake_aa_seq, input_rbd, input_same, input_x, input_y, input_z], training = False)


                    print('similarity_replace')
                    print(K.sum(K.cast(K.argmax(input_pre, axis=-1) == K.argmax(fake_mutation, axis=-1), tf.int64), axis=-1))
                    print('binding_score_mutated in generator')
                    print(binding_score_mutated)

                    mut_index = K.cast(K.argmax(input_pre, axis=-1) == K.argmax(fake_aa_seq, axis=-1), tf.int64)
                    where = tf.where(tf.equal(mut_index, 0))

                    pre_mut = tf.gather_nd(K.argmax(input_pre, axis=-1), where)
                    aft_mut = tf.gather_nd(K.argmax(fake_mutation, axis=-1), where)
                    print('pre_mut')
                    print(pre_mut)
                    print('aft_mut')
                    print(aft_mut)

                    predictions = self.discriminator(binding_score_mutated)
                    g_loss = self.g_loss_fn(misleading_labels, predictions)

                grads = tape.gradient(g_loss, self.generator.trainable_weights)

                #################################################################################################
                similarity = K.sum(K.cast(K.argmax(input_pre, axis=-1) == K.argmax(fake_mutation, axis=-1), tf.int64), axis=-1)

                original_len = tf.reshape(tf.repeat(fake_mutation.shape[1], fake_mutation.shape[0]),
                                          (fake_mutation.shape[0]))
                print('original_len')
                print(original_len)
                # print(tf.cast(original_len, tf.int32) - tf.cast(similarity, tf.int32))

                g_similarity_ratio = tf.cast(
                    tf.reduce_mean(tf.cast(similarity, tf.int32)) / tf.reduce_mean(tf.cast(original_len, tf.int32)),
                    tf.float32)

                g_similarity_ratio = g_similarity_ratio.numpy()

                g_aff = binding_score_mutated.numpy()
                #####################################################################################################

                self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

            # Monitor loss.
            self.gen_loss_tracker.update_state(g_loss)
            self.disc_loss_tracker.update_state(d_loss)
            self.g_similarity_ratio_tracker.update_state(g_similarity_ratio)
            self.d_similarity_ratio_tracker.update_state(d_similarity_ratio)
            self.g_binding_aff_tracker.update_state(g_aff)
            self.d_binding_aff_tracker.update_state(d_aff)


        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
            "d_ratio": self.d_similarity_ratio_tracker.result(),
            "g_ratio": self.g_similarity_ratio_tracker.result(),
            "d_aff": self.d_binding_aff_tracker.result(),
            "g_aff": self.g_binding_aff_tracker.result(),

        }

def discriminator_loss(real, fake, original_len, similarity):
    mut_num_reduced = tf.reduce_mean(tf.cast(original_len, tf.int32) - tf.cast(similarity, tf.int32))
    # penalty = tf.cast(tf.reduce_mean(tf.cast(similarity, tf.int32))/tf.reduce_mean(tf.cast(original_len, tf.int32)), tf.float32)
    print('mut_num_reduced_d')
    print(mut_num_reduced)

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




def discriminator_loss_replace(real, fake):
    real_loss = tf.reduce_mean(real)
    fake_loss = tf.reduce_mean(fake)
    return fake_loss - real_loss

def generator_loss_replace(fake):
    return -tf.reduce_mean(fake)

def build_discriminator_replace_model(dense_1_num = 16,dense_2_num = 16,dense_3_num = 8):
    # build
    Input = keras.Input(shape=(1,), name='Input') #@@@

    # dense
    Dense = layers.Dense(dense_1_num, activation='relu')(Input)
    Dense = layers.Dense(dense_2_num, activation='relu')(Dense)
    Dense = layers.Dense(dense_3_num, activation='relu')(Dense)
    Dense = layers.Dense(1, activation='sigmoid')(Dense)

    model = keras.models.Model([Input], Dense)
    return model