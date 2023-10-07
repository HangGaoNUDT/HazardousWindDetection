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
from sklearn.metrics import roc_curve, auc
import numpy as np
import random
from tools import *
from sklearn.manifold import TSNE
from Data_import import data_frame
from model_evaluation import *
from ClusterNN import cluster_NN,svm_classifier
from ClusterNN_without_linear_layer import *
tf.set_random_seed(1)
np.random.seed(1)
random.seed(1)

Data = data_frame()

# clusternn = cluster_NN()
# pilot_train_ = Data.pilot_semi_learning_generator(semi_ratio = 1)
# metric_values = clusternn.train(Data,pilot_train_,train_steps = 1001, BATCH_SIZE = 1024,
#                                                imbalance_beta_values = np.array([1,10,100,1000]),If_cal_AUC=False,
#                                 If_plot_tSNE_encoded=True)
clusternn = cluster_NN_without_linear_layer()
pilot_train_ = Data.pilot_semi_learning_generator(semi_ratio = 1)
metric_values = clusternn.train(Data,pilot_train_,train_steps = 1001, BATCH_SIZE = 1024,
                                               imbalance_beta_values = np.array([1,10,100,1000]),If_cal_AUC=True,
                                If_plot_tSNE=True,tsne_num_list = list(np.arange(0,1000,100)))
scio.savemat(f'./results_save/metric_values_without_linear_layer_1000.mat',
             {'metric_values': metric_values})
