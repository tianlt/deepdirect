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

#%%
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
aa_mutator_phase_2.load_weights('models/generator_step_1_weights.h5') 

discriminator_replace = build_discriminator_replace_model()


binding_affinity_predictor = build_model(activation='linear', f_num_1=32, f_num_2=64, f_num_3=128, f_num_4=256)
binding_affinity_predictor.load_weights('models/binding_affinity_predictor_weights.h5')

# assemble gan
AA_Mutation_Gan = AA_Mutation_GAN(
    discriminator=discriminator_replace, generator=aa_mutator_phase_2,
    binding_affinity_predictor=binding_affinity_predictor,
    clip_value=0.1, clip_replace_value=0.1, discriminator_unchange_training=2, select = 'increase',
    replace_index = True, generate_seq_num = 1
)


# batch size should be set at 1
# N_EPOCHS = 2
batch_size = 1

##############################################################
old_weights = AA_Mutation_Gan.get_weights()
# decrease/ increase in binding affinity
# def build_and_compile_AA_Mutation_Gan(dlr, glr):
#     AA_Mutation_Gan = AA_Mutation_GAN(
#         discriminator=discriminator_replace, generator=aa_mutator_phase_2,
#         binding_affinity_predictor=binding_affinity_predictor,
#         clip_value=0.05, clip_replace_value=0.1, discriminator_unchange_training=2, select='decrease',
#         replace_index=True, generate_seq_num=1
#     )
#     AA_Mutation_Gan.set_weights(old_weights)
#
#     AA_Mutation_Gan.compile(
#         d_optimizer=keras.optimizers.Adam(learning_rate=dlr),
#         g_optimizer=keras.optimizers.Adam(learning_rate=glr),
#         g_loss_fn=keras.losses.BinaryCrossentropy(from_logits=False),
#         d_loss_fn=keras.losses.BinaryCrossentropy(from_logits=False),
#     )
#     return AA_Mutation_Gan

# increase/ decrease in binding affinity
def build_and_compile_AA_Mutation_Gan_increase(dlr, glr):
    AA_Mutation_Gan = AA_Mutation_GAN(
        discriminator=discriminator_replace, generator=aa_mutator_phase_2,
        binding_affinity_predictor=binding_affinity_predictor,
        clip_value=0.05, clip_replace_value=0.1, discriminator_unchange_training=2, select='increase',
        replace_index=True, generate_seq_num=1
    )
    AA_Mutation_Gan.set_weights(old_weights)

    AA_Mutation_Gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=dlr),
        g_optimizer=keras.optimizers.Adam(learning_rate=glr),
        g_loss_fn=keras.losses.BinaryCrossentropy(from_logits=False),
        d_loss_fn=keras.losses.BinaryCrossentropy(from_logits=False),
    )
    return AA_Mutation_Gan

# es = EarlyStopping(monitor='g_ratio', mode='max', baseline=0.9)
random_num = random.randint(1,100000)
print('random_num')
print(random_num) #2343 84263

# decrease/ increase in binding affinity
# for num in range(1):
#     for i in [0.000005, 0.0000075, 0.0000085, 0.00001]: #d_optimizer
#         for j in [0.000005, 0.0000075, 0.0000085, 0.00001]: #g_optimizer
#             for N_EPOCHS in [1]:
#                 total = 0
#                 AA_Mutation_Gan = build_and_compile_AA_Mutation_Gan(i, j)
#                 csv_logger = tf.keras.callbacks.CSVLogger('metric/{}-training-num-{}-dlr-{:.6f}-glr-{:.6f}-epoch-{}.log'.format(random_num,num,i,j, N_EPOCHS), separator=',', append=True)
#                 for epoch in range(N_EPOCHS):
#                     for a, b, c, d, x, y, z, e in get_batch():
#                         history_1 = AA_Mutation_Gan.fit([a, b, c, d, x, y, z], batch_size=batch_size, shuffle=True,
#                                                         callbacks=[csv_logger])
#                         total = total + 1
#                         print('total')
#                         print(total)
#                         if AA_Mutation_Gan.g_similarity_ratio_tracker.result() >= 0.93 and total >= 12 and AA_Mutation_Gan.g_binding_aff_tracker.result() > 0:
#                             break
#
#
#
#                 d_result = AA_Mutation_Gan.d_similarity_ratio_tracker.result()
#                 g_result = AA_Mutation_Gan.g_similarity_ratio_tracker.result()
#                 d_aff_result = AA_Mutation_Gan.d_binding_aff_tracker.result()
#                 g_aff_result = AA_Mutation_Gan.g_binding_aff_tracker.result()
#                 aa_mutator = AA_Mutation_Gan.generator
#                 aa_mutator.save_weights('models/{}-aa_mutator-num-{}-dlr-{:.6f}-glr-{:.6f}-drt-{:.3f}-grt-{:.3f}-daff-{:.3f}-gaff-{:.3f}-epoch-{}_weights.h5'.format(random_num,num,i,j,d_result,g_result, d_aff_result, g_aff_result, N_EPOCHS))
#
# aa_mutator_final = build_aa_mutator()

# increase/ decrease in binding affinity
for num in range(1):
    for i in [0.000005, 0.0000075, 0.0000085, 0.00001]: #d_optimizer
        for j in [0.000005, 0.0000075, 0.0000085, 0.00001]: #g_optimizer
            for N_EPOCHS in [1]:
                total = 0
                AA_Mutation_Gan = build_and_compile_AA_Mutation_Gan_increase(i, j)
                csv_logger = tf.keras.callbacks.CSVLogger('metric/increase_{}-training-num-{}-dlr-{:.6f}-glr-{:.6f}-epoch-{}.log'.format(random_num,num,i,j, N_EPOCHS), separator=',', append=True)
                for epoch in range(N_EPOCHS):
                    for a, b, c, d, x, y, z, e in get_batch():
                        history_1 = AA_Mutation_Gan.fit([a, b, c, d, x, y, z], batch_size=batch_size, shuffle=True,
                                                        callbacks=[csv_logger])
                        total = total + 1
                        print('total')
                        print(total)
                        if AA_Mutation_Gan.g_similarity_ratio_tracker.result() >= 0.93 and total >= 12 and AA_Mutation_Gan.g_binding_aff_tracker.result() < 0:
                            break



                d_result = AA_Mutation_Gan.d_similarity_ratio_tracker.result()
                g_result = AA_Mutation_Gan.g_similarity_ratio_tracker.result()
                d_aff_result = AA_Mutation_Gan.d_binding_aff_tracker.result()
                g_aff_result = AA_Mutation_Gan.g_binding_aff_tracker.result()
                aa_mutator = AA_Mutation_Gan.generator
                aa_mutator.save_weights('models/increase-{}-aa_mutator-num-{}-dlr-{:.6f}-glr-{:.6f}-drt-{:.3f}-grt-{:.3f}-daff-{:.3f}-gaff-{:.3f}-epoch-{}_weights.h5'.format(random_num,num,i,j,d_result,g_result, d_aff_result, g_aff_result, N_EPOCHS))

aa_mutator_final = build_aa_mutator()







#
