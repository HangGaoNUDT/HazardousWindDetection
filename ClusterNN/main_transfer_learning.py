import  tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import os
os.environ["OMP_NUM_THREADS"] = '4'
from Data_import import data_frame
from model_evaluation import *
from ClusterNN import cluster_NN,svm_classifier
import scipy.io as scio
# tf.set_random_seed(1)
# np.random.seed(1)
# random.seed(1)
#
Data = data_frame()
clusternn = cluster_NN()
# # First train the network with data from the southern runway
# pilot_train_ = Data.pilot_semi_learning_generator(semi_ratio = 1)
# clusternn.train(Data,pilot_train_,train_steps = 500, BATCH_SIZE = 1024)

Data.read_data_for_transfer_learning()
metric_values = clusternn.transfer_learning(Data,Data.pilot_train_transfer,train_steps = 1001, BATCH_SIZE = 1024,
                                            imbalance_beta_values = np.array([1,10,100,1000]),
                                            If_cal_AUC =False, step_cal_AUC = [400],
                                            If_plot_tSNE = True, tsne_num_list=[400,800,1000],
                                            If_plot_hazard_value = False,If_plot_ROC = False)#, tsne_num_list=[0,50,100,200,400,500,1000]list(np.arange(0,501,100)),
# scio.savemat(
#     f'./results_save/metric_values_transferlearning.mat',
#     {'metric_values': metric_values})
