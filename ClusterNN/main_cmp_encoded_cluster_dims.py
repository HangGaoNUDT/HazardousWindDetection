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
# tf.set_random_seed(1)
# np.random.seed(1)
# random.seed(1)
#
Data = data_frame()
encoded_dims = [1,2,3]#4,4,4,4,4,4,7,10,20,
cluster_dims = [1,2,3]#1,2,3,4,5,6,7,10,20,
for dim_iter in range(len(encoded_dims)):
    tf.set_random_seed(1)
    np.random.seed(1)
    random.seed(1)
    clusternn = cluster_NN(encoded_dim = encoded_dims[dim_iter], cluster_dim = cluster_dims[dim_iter])
    pilot_train_ = Data.pilot_semi_learning_generator(semi_ratio = 1)
    metric_values = clusternn.train(Data,pilot_train_,train_steps = 1001, BATCH_SIZE = 1024,
                                    imbalance_beta_values = np.array([1,10,100,1000]),If_cal_AUC=True)
    scio.savemat(f'./results_save/metric_values_iterations_encodedDim{encoded_dims[dim_iter]}_clusterDim{cluster_dims[dim_iter]}_1000.mat',{'metric_values':metric_values})
