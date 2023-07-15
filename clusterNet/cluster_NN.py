import  tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import os
os.environ["OMP_NUM_THREADS"] = '4'
import scipy.io as scio
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import ConvLSTM2D, Dense, LeakyReLU, Multiply
from tensorflow.keras.layers import BatchNormalization, Lambda, Multiply, Add
from sklearn.metrics import roc_curve, auc
import numpy as np
import random
from tools import *

tf.set_random_seed(1)
np.random.seed(1)
random.seed(1)
N_CLUSTER = 2
encodede_dim = 4
cluster_dim = 4
def _test_cluster(clusternn,hazard_id):
    '''Test with 2021 data'''
    pilot_encoded_test, pilot_pred_test, cluster_features_test = sess.run([clusternn.encoded,clusternn.pred, clusternn.cluster_features],
                                                      feed_dict={clusternn.input: pilot_test, \
                                                                 clusternn.input_batch_size:
                                                                     pilot_test.shape[
                                                                         0]})
    si_acc = np.sum(pilot_pred_test == hazard_id) / pilot_pred_test.shape[0]
    return pilot_encoded_test,pilot_pred_test, cluster_features_test
class cluster_NN(object):
    def __init__(self, N_CLUSTER,cluster_dim):
        self.n_cluster = N_CLUSTER
        self.kmeans = KMeans(n_clusters=N_CLUSTER, n_init=20)
        # model setup
        self.input = tf.placeholder(tf.float32, shape=[None, 115])
        self.input_batch_size = tf.placeholder(tf.int32, shape=())

        self.w1 = tf.Variable(tf.random_normal(shape=(115, 64), stddev=0.01,seed=10), name='w1')
        self.b1 = tf.Variable(tf.zeros(shape=(64,)), name='b1')
        self.w2 = tf.Variable(tf.random_normal(shape=(64, 32), stddev=0.01,seed=10), name='w2')
        self.b2 = tf.Variable(tf.zeros(shape=(32,)), name='b2')
        self.w3 = tf.Variable(tf.random_normal(shape=(32, encodede_dim), stddev=0.01,seed=10), name='w3')
        self.b3 = tf.Variable(tf.zeros(shape=(encodede_dim,)), name='b3')
        self.w4 = tf.Variable(tf.random_normal(shape=(encodede_dim, 32), stddev=0.01,seed=10), name='w4')
        self.b4 = tf.Variable(tf.zeros(shape=(32,)), name='b4')
        self.w5 = tf.Variable(tf.random_normal(shape=(32, 64), stddev=0.01,seed=10), name='w5')
        self.b5 = tf.Variable(tf.zeros(shape=(64,)), name='b5')
        self.w6 = tf.Variable(tf.random_normal(shape=(64, 115), stddev=0.01,seed=10), name='w6')
        self.b6 = tf.Variable(tf.zeros(shape=(115,)), name='b6')

        self.encoded, self.decoded = self.ae_setup(self.input)
        self.w7 = tf.Variable(tf.random_normal(shape=(encodede_dim, cluster_dim), stddev=1, seed=10), name='w7')
        self.b7 = tf.Variable(tf.zeros(shape=(cluster_dim,)), name='b7')
        self.cluster_features_data = self.clusnn_setup(self.encoded)
        self.cluster_features = self.cluster_features_data
        self.cluster_feature_dim = cluster_dim
        self.B_update = tf.Variable(tf.zeros(shape=(N_CLUSTER, self.cluster_feature_dim)), name="B_update")
        self.dist = self._pairwise_euclidean_distance(self.cluster_features, self.B_update, self.input_batch_size,
                                                      N_CLUSTER)
        self.pred = tf.argmin(self.dist, 1)
        self.T = (tf.one_hot(self.pred, N_CLUSTER))
        q = 1.0 / (1.0 + self.dist ** 2)  # REF: dec
        self.q = (q / tf.reduce_sum(q, axis=1, keepdims=True))
        self.p = tf.placeholder(tf.float32, shape=(None, N_CLUSTER))
        self.loss_dec = self._kl_divergence(self.p, self.q)
        self.reconstruct_loss = tf.losses.mean_squared_error(self.decoded, self.input)

        # introduce some hazardous wind reports by pilots as prior information
        self.pilot_input = tf.placeholder(tf.float32, shape=[None, 115])
        self.pilot_size = tf.placeholder(tf.int32, shape=())
        self.pilot_encoded, self.pilot_decoded = self.ae_setup(self.pilot_input)
        self.pilot_cluster_features_data = self.clusnn_setup(self.pilot_encoded)
        self.pilot_center = tf.reshape(self.pilot_input[0, ::],
                                       [1, -1])  # tf.reshape(tf.reduce_mean(self.pilot_input,axis = 0),[1,-1])
        center_encoded, center_decoded = self.ae_setup(self.pilot_center)
        self.pilot_center_cluster_features = self.clusnn_setup(center_encoded)
        self.pilot_center_dist = (self.pilot_cluster_features_data - self.pilot_center_cluster_features) ** 2
        self.pilot_among_dists = self._pairwise_euclidean_distance(self.pilot_cluster_features_data,
                                                                   self.pilot_cluster_features_data, self.pilot_size,
                                                                   self.pilot_size)
        self.pilot_loss = tf.reduce_mean((tf.reduce_sum(self.pilot_center_dist, axis=1)))  # 使飞行员分类在第一类

        # introduce some calm winds as prior information
        self.true_input = tf.placeholder(tf.float32, shape=[None, 115])
        self.true_size = tf.placeholder(tf.int32, shape=())
        self.true_encoded, self.true_decoded = self.ae_setup(self.true_input)
        self.true_cluster_features_data = self.clusnn_setup(self.true_encoded)
        self.true_center = tf.zeros(shape=(1, 115))
        center_encoded, center_decoded = self.ae_setup(self.true_center)
        self.true_center_cluster_features = self.clusnn_setup(center_encoded)  # self.clusnn_setup(center_encoded)
        self.true_center_dist = (self.true_cluster_features_data - self.true_center_cluster_features) ** 2
        self.true_among_dists = self._pairwise_euclidean_distance(self.true_cluster_features_data,
                                                                  self.true_cluster_features_data, self.true_size,
                                                                  self.true_size)
        self.true_loss = tf.reduce_mean((tf.reduce_sum(self.true_center_dist, axis=1)))
        true_pilot_interval = self._pairwise_euclidean_distance(self.true_cluster_features_data,
                                                                self.pilot_cluster_features_data, self.true_size,
                                                                self.pilot_size)
        self.max_interval = -(tf.reduce_mean(true_pilot_interval) * 1)
        pilot_true_center = tf.concat([self.pilot_center_cluster_features, self.true_center_cluster_features], axis=0)
        dist2 = self._pairwise_euclidean_distance(self.cluster_features, pilot_true_center, self.input_batch_size, N_CLUSTER)
        q = 1.0 / (1.0 + dist2)  # REF: dec
        q = (q / tf.reduce_sum(q, axis=1, keepdims=True))
        self.class_loss = tf.reduce_mean(q[:, 0])
        self.w_loss = tf.reduce_mean((self.w7) ** 2)
        self.loss_supervise = 1 * self.reconstruct_loss + 1 * self.loss_dec+0.005* self.pilot_loss+ 0.02*self.true_loss\
        + 0.04 * self.max_interval+ 1 * self.class_loss
        self.optimizer_dec = tf.train.AdamOptimizer(0.001).minimize(self.loss_supervise)
        self.loss_pretrain = self.reconstruct_loss  # +self.w_loss*0.01
        self.optimizer_pretrain = tf.train.AdamOptimizer(0.005).minimize(self.loss_pretrain)

    def clusnn_setup(self, encoded):
        cluster_features = (tf.matmul(encoded, self.w7) + self.b7)
        return cluster_features
    def svm_para_cal(self,pred_,cluster_features_):
        a = np.zeros(shape=(2, 1))
        b = np.zeros(shape=(1, ))
        if np.unique(pred_).shape[0]>1:
            svc = self.svc.fit(cluster_features_,pred_)
            a = svc.coef_.T
            b = svc.intercept_
        return tf.assign(self.w8, a),tf.assign(self.b8, b)
    def ae_setup(self,input):
        x = tf.nn.relu(tf.matmul(input, self.w1) + self.b1)
        x = tf.nn.relu(tf.matmul(x, self.w2) + self.b2)
        encoded = (tf.matmul(x, self.w3) + self.b3)
        x = tf.nn.relu(tf.matmul(encoded, self.w4) + self.b4)
        x = tf.nn.relu(tf.matmul(x, self.w5) + self.b5)
        decoded = (tf.matmul(x, self.w6) + self.b6)
        return encoded,decoded
    def _pairwise_euclidean_distance(self, a, b, input_batch_size,center_nums):
        p1 = tf.matmul(
            tf.expand_dims(tf.reduce_sum(tf.square(a), 1), 1),
            tf.ones(shape=(1, center_nums))
        )
        p2 = tf.transpose(tf.matmul(
            tf.reshape(tf.reduce_sum(tf.square(b), 1), shape=[-1, 1]),
            tf.ones(shape=(input_batch_size, 1)),
            transpose_b=True
        ))
        res = tf.sqrt(tf.abs(tf.add(p1, p2) - 2 * tf.matmul(a, b, transpose_b=True))+1e-4)
        return res

    def _kl_divergence(self, target, pred):
        return tf.reduce_mean(tf.reduce_sum(target * tf.log(target / (pred)), axis=1))

    def target_distribution(self, q):
        p = q**2 / q.sum(axis=0)
        p = p / p.sum(axis=1, keepdims=True)
        return p
    def get_assign_cluster_centers_op(self, features):
        kmeans = self.kmeans.fit(features)
        kmeans.cluster_centers_ = kmeans.cluster_centers_[np.argsort(np.sum(kmeans.cluster_centers_**2,1))]
        return tf.assign(self.B_update, kmeans.cluster_centers_)

    def get_assign_B_undate_op(self,features,T_):
        B_update = np.zeros((N_CLUSTER,self.cluster_feature_dim))
        for i in range(N_CLUSTER):
            B_update[i,::] = np.mean(features[np.where(T_[:,i]==1)[0],::],axis = 0).reshape(1,-1)
            if np.isnan(B_update[i,0]):
                idx = random.randint(0, BATCH_SIZE-1)
                B_update[i,::] = features[idx,::]
        B_update = B_update[np.argsort(np.sum(B_update ** 2, 1))]
        return tf.assign(self.B_update, B_update)