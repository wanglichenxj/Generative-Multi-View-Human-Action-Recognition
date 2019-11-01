# =====================
# Adaptive Graph Guided Embedding for Multi-label Annotation (AG2E)
# =====================
# Author: Lichen Wang, Yunyu Liu
# Date: Mar, 2019
# E-mail: wanglichenxj@gmail.com, liu.yuny@husky.neu.edu

# @inproceedings{GMVAR_lichen,
#   title={Generative Multi-View Human Action Recognition},
#   author={Wang, Lichen and Ding, Zhengming and Tao, Zhiqiang and Liu, Yunyu and Fu, Yun},
#   booktitle={Proc. IEEE International Conference on Computer Vision},
#   year={2019},
#   organization={}
# }
# =====================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data
import random
import cv2
import os
import dataset_loader
import scipy.io

# assign the GPU number
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main(times):
    database = 'UCB' # assign evaluation dataset 'UWA30' and 'DHA'
    train_data = dataset_loader.data_loader(database)
    train_data.read_train()

    # assign action number for different datasets
    if database == 'UWA30':
        class_num = 30
    elif database == 'UCB':
        class_num = 11
    elif database == 'DHA':
        class_num = 23

    # training parameters
    mb_size = 64
    x_dim = 3*2048 # RGB feature dimension
    y_dim = 110 # depth feature dimension
    hx_dim = 800 # RGB encoder E1 hidden layer dimension
    hy_dim = 100 # Depth encoder E2 hidden layer dimension

    space_dim_1 = 300 # RGB subspace representation dimension
    space_dim_2 = 100 # Depth subspace representation dimension

    hg1_dim = 600 # generator 1 (RGB->depth) hidden layer dimension
    hg2_dim = 600 # generator 2 (depth->RGB) hidden layer dimension

    hd1_dim = 400 # discriminator 1 hidden layer dimension
    hd2_dim = 400 # discriminator 2 hidden layer dimension

    noise_dim = 300 # noise dimension for the conditional generator
    h_g_dim = 500 
    epsilon = 1e-6

    G_updates = 5
    c_rf = 0.99  # weights between real and fake
    cg_rf = 0.9 # the weights of the Cg() model (looks like =1 is more reasonable)
    hcg_dim = 100 # graph classifier Cg() hidden layer dimension
    lamda_g_smi = 0.1  # similiarity weights
    lbd_t = 0.001 # triplet loss weight    
    
    output_m=[] # final output results from each config    

    # Leaky Relu
    def leak_relu(x, alpha):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

    def sigmoid_conv(x):
        x=np.asarray(x)
        x_out = 1 / (1 + np.exp(-x))
        return x_out

    # network weights initialize
    def xavier_init(size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    # noise generation for conditional generator input
    def sample_Noise(m, n):
        return np.random.uniform(0., 1., size=[m, n])

    x = tf.placeholder(tf.float32, shape=[None, x_dim]) # RGB feature
    y = tf.placeholder(tf.float32, shape=[None, y_dim])  # depth feature
    z = tf.placeholder(tf.float32, shape=[None, class_num]) # label
    noise = tf.placeholder(tf.float32, shape=[None, noise_dim]) # random noise for generator

    # =======================
    # structure of encoder E_1(.) for RGB view
    # =======================
    E_x_W1 = tf.Variable(xavier_init([x_dim, hx_dim]))
    E_x_b1 = tf.Variable(tf.zeros(shape=[hx_dim]))
    E_x_W2 = tf.Variable(xavier_init([hx_dim, space_dim_1]))
    E_x_b2 = tf.Variable(tf.zeros(shape=[space_dim_1]))
    theta_E_1 = [E_x_W1, E_x_W2, E_x_b1, E_x_b2]

    def E_1(x):
        x_norm = tf.nn.sigmoid(x) # we consider the sigmoid is a normalization function to improve the performance
        E_x_h1 = leak_relu(tf.matmul(x_norm, E_x_W1) + E_x_b1, 0.25)  
        E_x_logit = leak_relu(tf.matmul(E_x_h1, E_x_W2) + E_x_b2, 0.25)   
        return E_x_logit

    # =======================
    # structure of encoder E_2(.) for depth view
    # =======================
    E_y_W1 = tf.Variable(xavier_init([y_dim, hy_dim]))
    E_y_b1 = tf.Variable(tf.zeros(shape=[hy_dim]))
    E_y_W2 = tf.Variable(xavier_init([hy_dim, space_dim_2]))
    E_y_b2 = tf.Variable(tf.zeros(shape=[space_dim_2]))
    theta_E_2 = [E_y_W1, E_y_W2, E_y_b1, E_y_b2]

    def E_2(x):
        E_y_h1 = leak_relu(tf.matmul(x, E_y_W1) + E_y_b1, 0.25)
        E_y_logit = leak_relu(tf.matmul(E_y_h1, E_y_W2) + E_y_b2,0.25) 
        return E_y_logit

    
    # =======================
    # structure of classifier C_1(.)
    # =======================
    Classifier_1_W1 = tf.Variable(xavier_init([space_dim_1, class_num]))
    Classifier_1_b1 = tf.Variable(tf.zeros(shape=[class_num]))    
    theta_C_1 = [Classifier_1_W1, Classifier_1_b1]

    def classifier_1(x):        
        Classifier_logit = tf.matmul(x, Classifier_1_W1) + Classifier_1_b1        
        Classifier_prob = tf.nn.sigmoid(Classifier_logit)
        return Classifier_logit, Classifier_prob

    # =======================
    # structure of classifier C_2(.)
    # =======================
    Classifier_2_W1 = tf.Variable(xavier_init([space_dim_2, class_num]))
    Classifier_2_b1 = tf.Variable(tf.zeros(shape=[class_num]))    
    theta_C_2 = [Classifier_2_W1, Classifier_2_b1]

    def classifier_2(x):
        Classifier_logit = tf.matmul(x, Classifier_2_W1) + Classifier_2_b1        
        Classifier_prob = tf.nn.sigmoid(Classifier_logit)
        return Classifier_logit, Classifier_prob


    # =======================
    # structure of discriminator D_1(.), input is the generated depth representation
    # =======================
    D1_W1 = tf.Variable(xavier_init([y_dim + class_num, hd1_dim]), name='D_W1')
    D1_b1 = tf.Variable(tf.zeros(shape=[hd1_dim]),name='D_b1')
    D1_W2 = tf.Variable(xavier_init([hd1_dim, hd1_dim]),name='D_W2')
    D1_b2 = tf.Variable(tf.zeros(shape=[hd1_dim]),name='D_b2')
    D1_M1 = tf.Variable(xavier_init([hd1_dim, 15]),name='D_M1')
    D1_Mb1 = tf.Variable(xavier_init([15]),name='D_Mb1')
    D1_W3 = tf.Variable(xavier_init([405, 1]),name='D_W3')
    D1_b3 = tf.Variable(tf.zeros(shape=[1]),name='D_b3')
    theta_D1 = [D1_W1, D1_W2, D1_b1, D1_b2, D1_W3, D1_b3,D1_M1,D1_Mb1]

    # mini batch to stablize the training procedure
    def minibatch1(input, num_kernels=5, kernel_dim=3):
        x = tf.matmul(input, D1_M1) + D1_Mb1
        activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
        diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
        return tf.concat(axis=1, values=[input, minibatch_features])

    def discriminator1(y, z):
        inputs = tf.concat(axis=1, values=[y, z])        
        D_h1 = leak_relu(tf.matmul(inputs, D1_W1) + D1_b1,0.2)
        D_h2 = leak_relu(tf.matmul(D_h1, D1_W2) + D1_b2,0.2)
        D_h3 = minibatch1(D_h2)
        D_h3 = tf.matmul(D_h3, D1_W3) + D1_b3
        D_logit = D_h3
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob, D_logit


    # =======================
    # structure of discriminator D_2(.), input is the generated RGB representation
    # =======================
    D2_W1 = tf.Variable(xavier_init([x_dim + class_num, hd2_dim]), name='D_W1')
    D2_b1 = tf.Variable(tf.zeros(shape=[hd2_dim]),name='D_b1')
    D2_W2 = tf.Variable(xavier_init([hd2_dim, hd2_dim]),name='D_W2')
    D2_b2 = tf.Variable(tf.zeros(shape=[hd2_dim]),name='D_b2')
    D2_M1 = tf.Variable(xavier_init([hd2_dim, 15]),name='D_M1')
    D2_Mb1 = tf.Variable(xavier_init([15]),name='D_Mb1')
    D2_W3 = tf.Variable(xavier_init([405, 1]),name='D_W3')
    D2_b3 = tf.Variable(tf.zeros(shape=[1]),name='D_b3')
    theta_D2 = [D2_W1, D2_W2, D2_b1, D2_b2, D2_W3, D2_b3,D2_M1,D2_Mb1]

    # mini batch to stablize the training procedure
    def minibatch2(input, num_kernels=5, kernel_dim=3):
        x = tf.matmul(input, D2_M1) + D2_Mb1
        activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
        diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
        return tf.concat(axis=1, values=[input, minibatch_features])

    def discriminator2(y, z):
        inputs = tf.concat(axis=1, values=[y, z])
        D_h1 = leak_relu(tf.matmul(inputs, D2_W1) + D2_b1,0.2)
        D_h2 = leak_relu(tf.matmul(D_h1, D2_W2) + D2_b2,0.2)
        D_h3 = minibatch2(D_h2)
        D_h3 = tf.matmul(D_h3, D2_W3) + D2_b3
        D_logit = D_h3
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob, D_logit


    # =======================
    # structure of discriminator G_1(.), from RGB generates depth representation
    # =======================
    G_w1_BN = tf.Variable(xavier_init([x_dim + noise_dim, h_g_dim]), name='G_W1')
    G_scale1 = tf.Variable(tf.ones([h_g_dim]))
    G_beta1 = tf.Variable(tf.zeros([h_g_dim]))
    G_w2_BN = tf.Variable(xavier_init([h_g_dim, h_g_dim]),name='G_W2')
    G_scale2 = tf.Variable(tf.ones([h_g_dim]))
    G_beta2 = tf.Variable(tf.zeros([h_g_dim]))
    G_w3 = tf.Variable(xavier_init([h_g_dim, h_g_dim]),name='G_W3')
    G_b3 = tf.Variable(tf.zeros([h_g_dim]))
    G_w4 = tf.Variable(xavier_init([h_g_dim, y_dim]), name='G_W3')
    G_b4 = tf.Variable(tf.zeros([y_dim]))
    theta_G1 = [G_w1_BN, G_scale1, G_beta1, G_w2_BN, G_scale2, G_beta2, G_w3, G_b3, G_w4, G_b4]

    def generator1(x, noise):
        noise = noise * 60 
        inputs = tf.concat(axis=1, values=[x, noise])
        G_z1_BN = tf.matmul(inputs,G_w1_BN)
        G_batch_mean1, G_batch_var1 = tf.nn.moments(G_z1_BN,[0])
        G_z1_hat = (G_z1_BN - G_batch_mean1) / tf.sqrt(G_batch_var1 + epsilon)
        G_BN1 = G_scale1 * G_z1_hat + G_beta1
        G_l1_BN = leak_relu(G_BN1,0.2)
        G_z2_BN = tf.matmul(G_l1_BN,G_w2_BN)
        G_batch_mean2, G_batch_var2 = tf.nn.moments(G_z2_BN,[0])
        G_BN2 = tf.nn.batch_normalization(G_z2_BN,G_batch_mean2,G_batch_var2,G_beta2,G_scale2,epsilon)
        G_BN3 = leak_relu(tf.matmul(G_BN2,G_w3) + G_b3,0.2)
        G_prob = tf.matmul(G_BN3,G_w4) + G_b4
        return G_prob


    # =======================
    # structure of discriminator G_2(.), from depth generate RGB representation
    # =======================
    G2_w1_BN = tf.Variable(xavier_init([y_dim + noise_dim, h_g_dim]), name='G_W1') 
    G2_scale1 = tf.Variable(tf.ones([h_g_dim]))
    G2_beta1 = tf.Variable(tf.zeros([h_g_dim]))
    G2_w2_BN = tf.Variable(xavier_init([h_g_dim, h_g_dim]),name='G_W2')
    G2_scale2 = tf.Variable(tf.ones([h_g_dim]))
    G2_beta2 = tf.Variable(tf.zeros([h_g_dim]))
    G2_w3 = tf.Variable(xavier_init([h_g_dim, h_g_dim]),name='G_W3')
    G2_b3 = tf.Variable(tf.zeros([h_g_dim]))
    G2_w4 = tf.Variable(xavier_init([h_g_dim, x_dim]), name='G_W3')
    G2_b4 = tf.Variable(tf.zeros([x_dim]))
    theta_G2 = [G2_w1_BN, G2_scale1, G2_beta1, G2_w2_BN, G2_scale2, G2_beta2, G2_w3, G2_b3, G2_w4, G2_b4]

    def generator2(x, noise):  
        inputs = tf.concat(axis=1, values=[x, noise])
        G_z1_BN = tf.matmul(inputs,G2_w1_BN)
        G_batch_mean1, G_batch_var1 = tf.nn.moments(G_z1_BN,[0])
        G_z1_hat = (G_z1_BN - G_batch_mean1) / tf.sqrt(G_batch_var1 + epsilon)
        G_BN1 = G_scale1 * G_z1_hat + G_beta1
        G_l1_BN = leak_relu(G_BN1,0.2)
        G_z2_BN = tf.matmul(G_l1_BN,G2_w2_BN)
        G_batch_mean2, G_batch_var2 = tf.nn.moments(G_z2_BN,[0])
        G_BN2 = tf.nn.batch_normalization(G_z2_BN,G_batch_mean2,G_batch_var2,G2_beta2,G2_scale2,epsilon)
        G_BN3 = leak_relu(tf.matmul(G_BN2,G2_w3) + G2_b3,0.2)
        G_prob = tf.matmul(G_BN3,G2_w4) + G2_b4
        return G_prob


    # ====================
    # view-correlation discovery network
    # Input: the predicted labels from C_1(.) and C_2(.) (The RGB and depth view)
    # Output: the final predicted label
    # ====================    
    Classifier_g_W1 = tf.Variable(xavier_init([class_num * class_num, hcg_dim]))
    Classifier_g_b1 = tf.Variable(tf.zeros(shape=[hcg_dim]))
    Classifier_g_W2 = tf.Variable(xavier_init([hcg_dim, class_num]))
    Classifier_g_b2 = tf.Variable(tf.zeros(shape=[class_num]))
    theta_g_C = [Classifier_g_W1, Classifier_g_W2, Classifier_g_b1, Classifier_g_b2]  # double layer

    def classifier_g(pr, pd):
        pr_in = pr 
        pd_in = pd
        C_prob_1 = tf.expand_dims(pr_in, -1)
        C_prob_2 = tf.expand_dims(pd_in, 1)
        W_feature = tf.matmul(C_prob_1, C_prob_2) # cross-view discovery matrix
        C_hw = tf.reshape(W_feature, [-1, class_num * class_num])
        C_h1 = leak_relu(tf.matmul(C_hw, Classifier_g_W1) + Classifier_g_b1, 0.25)
        Classifier_g_logit = tf.matmul(C_h1, Classifier_g_W2) + Classifier_g_b2        
        Classifier_g_prob = tf.nn.softmax(Classifier_g_logit)
        return Classifier_g_logit, Classifier_g_prob


    # =======================
    # model training function
    # =======================
    def train_total():     
        generate_fake_z2 = generator1(x, noise) # generate RGB -> depth
        generate_fake_z1 = generator2(y, noise) # generate depth -> RGB

        C_1_real_logit, C_1_real_prob = classifier_1(E_1(x))
        C_1_fake_logit, C_1_fake_prob = classifier_1(E_1(generate_fake_z1))

        C_2_real_logit, C_2_real_prob = classifier_2(E_2(y))
        C_2_fake_logit, C_2_fake_prob = classifier_2(E_2(generate_fake_z2))

        C_g_logit_real, C_g_prob_real = classifier_g(C_1_real_prob, C_2_real_prob)
        # C_g_logit_fake, C_g_prob_fake = classifier_g(C_1_fake_prob, C_2_fake_prob)
        C_g_logit_rf, C_g_prob_rf = classifier_g(C_1_real_prob, C_2_fake_prob)
        C_g_logit_fr, C_g_prob_fr = classifier_g(C_1_fake_prob, C_2_real_prob)

        acc_c1_real = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(C_1_real_logit, 1), tf.argmax(z, 1)), "float"))
        acc_c1_fake = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(C_1_fake_logit, 1), tf.argmax(z, 1)), "float"))
        acc_c2_real = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(C_2_real_logit, 1), tf.argmax(z, 1)), "float"))
        acc_c2_fake = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(C_2_fake_logit, 1), tf.argmax(z, 1)), "float"))

        C_1_loss = c_rf * tf.reduce_mean(tf.square(C_1_real_logit - z)) + (1.0 - c_rf) * tf.reduce_mean(tf.square(C_1_fake_logit - z))
        C_2_loss = c_rf * tf.reduce_mean(tf.square(C_2_real_logit - z)) + (1.0 - c_rf) * tf.reduce_mean(tf.square(C_2_fake_logit - z))

        C_g_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=C_g_logit_real, labels=z))
        C_g_loss_rf = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=C_g_logit_rf, labels=z))
        C_g_loss_fr = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=C_g_logit_fr, labels=z))
                
        C_g_loss_sum = cg_rf * C_g_loss_real + (1.0 - cg_rf)/2 * (C_g_loss_rf + C_g_loss_fr)
        C_g_solver = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(C_g_loss_sum, var_list=[theta_g_C])

        triplet_loss_1 = tf.reduce_mean(tf.contrib.losses.metric_learning.triplet_semihard_loss(labels=tf.argmax(z, 1),embeddings=tf.nn.l2_normalize(E_1(x)), margin=1.0))
        triplet_loss_2 = tf.reduce_mean(tf.contrib.losses.metric_learning.triplet_semihard_loss(labels=tf.argmax(z, 1),embeddings=tf.nn.l2_normalize(E_2(y)), margin=1.0))

        E_1_loss = lbd_t * triplet_loss_1 + C_1_loss
        E_2_loss = lbd_t * triplet_loss_2 + C_2_loss
        E_1_solver = tf.train.AdamOptimizer(0.0001).minimize(E_1_loss, var_list=[theta_E_1, theta_C_1])
        E_2_solver = tf.train.AdamOptimizer(0.0001).minimize(E_2_loss, var_list=[theta_E_2, theta_C_2])

        # discriminator 1 loss
        D1_prob_real, D1_logit_real = discriminator1(y, z)
        D1_prob_fake, D1_logit_fake = discriminator1(generate_fake_z2, z)
        D1_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D1_logit_real, labels=tf.ones_like(D1_logit_real)))
        D1_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D1_logit_fake, labels=tf.zeros_like(D1_logit_fake)))
        D1_loss = D1_loss_real + D1_loss_fake

        # discriminator 2 loss
        D2_prob_real, D2_logit_real = discriminator2(x, z)  
        D2_prob_fake, D2_logit_fake = discriminator2(generate_fake_z1, z)
        D2_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_logit_real, labels=tf.ones_like(D2_logit_real)))
        D2_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_logit_fake, labels=tf.zeros_like(D2_logit_fake)))
        D2_loss = D2_loss_real + D2_loss_fake
        
        D1_real_ave = tf.reduce_mean(D1_prob_real)
        D1_fake_ave = tf.reduce_mean(D1_prob_fake)
        D2_real_ave = tf.reduce_mean(D2_prob_real)
        D2_fake_ave = tf.reduce_mean(D2_prob_fake)

        # generator 1 and generator 2 loss
        G1_loss_dis = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D1_logit_fake, labels=tf.ones_like(D1_logit_fake)))
        G2_loss_dis = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_logit_fake, labels=tf.ones_like(D2_logit_fake)))        
        G1_loss_fea = tf.reduce_mean(tf.norm(generate_fake_z2 - y, ord='euclidean')) # generated feature similiarity
        G2_loss_fea = tf.reduce_mean(tf.norm(generate_fake_z1 - x, ord='euclidean')) # generated feature similiarity        
        G1_loss = G1_loss_dis + lamda_g_smi * G1_loss_fea
        G2_loss = G2_loss_dis + lamda_g_smi * G2_loss_fea


        D1_solver = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(D1_loss, var_list=[theta_D1])
        D2_solver = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(D2_loss, var_list=[theta_D2])
        G1_solver = tf.train.AdamOptimizer(learning_rate=0.00003).minimize(G1_loss, var_list=[theta_G1])
        G2_solver = tf.train.AdamOptimizer(learning_rate=0.00004).minimize(G2_loss, var_list=[theta_G2])

        # Evaluate on testing stage in 3 settings: (1) real Z1 real Z2 (2) real Z1 fake Z2 (3) fake Z1 real Z2
        C_r1r2_logit, C_r1r2_prob = classifier_g(C_1_real_prob, C_2_real_prob) # real z1 + real z2
        acc_te_r1r2 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(C_r1r2_prob, 1), tf.argmax(z, 1)), "float"))
        C_r1f2_logit, C_r1f2_prob = classifier_g(C_1_real_prob, C_2_fake_prob) # real z1 + fake z2
        acc_te_r1f2 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(C_r1f2_prob, 1), tf.argmax(z, 1)), "float"))
        C_f1r2_logit, C_f1r2_prob = classifier_g(C_1_fake_prob, C_2_real_prob) # fake z1 + real z2
        acc_te_f1r2 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(C_f1r2_prob, 1), tf.argmax(z, 1)), "float"))

        # ============= Set the GPU usage
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        # start training
        for i in range(30001):

            # ===================
            #  Training stage
            # ===================
            x_mb, y_mb, z_mb = train_data.train_next_batch(mb_size)            
            noise_sample = sample_Noise(mb_size, noise_dim)
            for upCE in range(1):
                _, _, _= sess.run([C_g_solver, E_1_solver, E_2_solver], feed_dict={x:x_mb, y: y_mb, z: z_mb, noise:noise_sample})            
            for upD in range(1):               
                _, _ = sess.run([D1_solver, D2_solver], feed_dict={x:x_mb, y: y_mb, z: z_mb, noise:noise_sample})            
            for upG in range(G_updates):
                x_mb, y_mb, z_mb = train_data.train_next_batch(mb_size)                
                noise_sample = sample_Noise(mb_size, noise_dim)
                _, _ = sess.run([G1_solver, G2_solver], feed_dict={x:x_mb, y: y_mb,z: z_mb, noise:noise_sample})

            # =====================
            #  Testing stage
            # =====================
            if i % 50 == 0:
                # ===== get training results
                x_mb, y_mb, z_mb = train_data.train_next_batch(mb_size) # load batch-size training      
                noise_sample = sample_Noise(mb_size, noise_dim)
                acc_tr_c1r, acc_tr_c1f, acc_tr_c2r, acc_tr_c2f = sess.run([acc_c1_real, acc_c1_fake, acc_c2_real, acc_c2_fake], feed_dict={x:x_mb, y: y_mb, z: z_mb, noise:noise_sample})
                D1_tr_r, D1_tr_f, D2_tr_r, D2_tr_f = sess.run([D1_real_ave, D1_fake_ave, D2_real_ave, D2_fake_ave], feed_dict={x:x_mb, y: y_mb, z: z_mb, noise:noise_sample})

                # ===== get the testing results
                x_mb, y_mb, z_mb = train_data.test_next_batch(train_data.sample_test_num) # load all test samples                
                noise_sample = sample_Noise(train_data.sample_test_num, noise_dim)                
                acc_te_r1r2_num, acc_te_r1f2_num, acc_te_f1r2_num, acc_te_c1r, acc_te_c1f, acc_te_c2r, acc_te_c2f = sess.run([acc_te_r1r2, acc_te_r1f2, acc_te_f1r2, acc_c1_real, acc_c1_fake, acc_c2_real, acc_c2_fake], feed_dict={x: x_mb, y: y_mb, z: z_mb, noise: noise_sample})
                print('Iteration = ',i, '  Accuracy:  RGB = %.4f'% acc_te_r1f2_num, '  Depth = %.4f'% acc_te_f1r2_num, '  RGB-D = %.4f'% acc_te_r1r2_num)
   
        # release GPU memory after training
        tf.reset_default_graph()

    # ======================
    # run the train function
    # ======================
    train_total()

main(times=1)

