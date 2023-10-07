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
semi_ratios = np.arange(0.1,1.1,0.1)
for id_ratio in range(len(semi_ratios)):
    tf.set_random_seed(1)
    np.random.seed(1)
    random.seed(1)
    clusternn = cluster_NN()
    pilot_train_ = Data.pilot_semi_learning_generator(semi_ratio = semi_ratios[id_ratio])
    metric_values = clusternn.train(Data,pilot_train_,train_steps = 1001, BATCH_SIZE = 1024,
                                                   imbalance_beta_values = np.array([1,10,100,1000]),If_cal_AUC=True)
    scio.savemat(f'./results_save/metric_values_hazrd_labels_num10_{semi_ratios[id_ratio]*10}_1000.mat',{'metric_values':metric_values})
