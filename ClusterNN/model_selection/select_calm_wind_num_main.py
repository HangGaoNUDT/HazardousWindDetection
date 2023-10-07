import  tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import os
os.environ["OMP_NUM_THREADS"] = '4'
import random
from tools import *
from Data_import import data_frame
from model_evaluation import *
from ClusterNN import cluster_NN,svm_classifier
tf.set_random_seed(1)
np.random.seed(1)
random.seed(1)

clusternn = cluster_NN()
# calm_train_num = 50
calm_range_set = 5
for calm_train_num in [10]:#np.arange(10,16,1):1,5,10,30,50,70,90,110,130,150
    tf.set_random_seed(1)
    np.random.seed(1)
    random.seed(1)
    Data = data_frame(calm_train_num=calm_train_num,calm_range_set=calm_range_set)
    pilot_train_ = Data.pilot_semi_learning_generator(semi_ratio = 1)
    metric_values = clusternn.train(Data,pilot_train_,train_steps = 5001, BATCH_SIZE = 1024,
                                                   imbalance_beta_values = np.array([1,10,100,1000]),If_cal_AUC=True)
    # scio.savemat(f'./results_save/metric_values_00001.mat',{'metric_values':metric_values})
    scio.savemat(f'./results_save/metric_values_CalmRange{calm_range_set}_CalmTrainNum{calm_train_num}_5000.mat',{'metric_values':metric_values})
