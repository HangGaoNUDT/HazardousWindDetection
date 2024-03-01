import  tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from scipy.stats import pearsonr
import os
os.environ["OMP_NUM_THREADS"] = '4'
import random
from sklearn.svm import SVC
from tools import *
from model_evaluation import *
import sklearn.svm as svm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.patches as mpatches
from mpl_toolkits import mplot3d
tf.set_random_seed(1)
np.random.seed(1)
random.seed(1)


class autoencoder_NN(object):
    def __init__(self, encoded_dim):
        self.n_cluster = 2
        self.encoded_dim = encoded_dim
        # model setup
        tf.reset_default_graph()
        self.input = tf.placeholder(tf.float32, shape=[None, 115])
        self.input_batch_size = tf.placeholder(tf.int32, shape=())
        self.w1 = tf.Variable(tf.random_normal(shape=(115, 64), stddev=0.01, seed=10), name='w1')
        self.b1 = tf.Variable(tf.zeros(shape=(64,)), name='b1')
        self.w2 = tf.Variable(tf.random_normal(shape=(64, 32), stddev=0.01, seed=10), name='w2')
        self.b2 = tf.Variable(tf.zeros(shape=(32,)), name='b2')
        self.w3 = tf.Variable(tf.random_normal(shape=(32, self.encoded_dim), stddev=0.01, seed=10), name='w3')
        self.b3 = tf.Variable(tf.zeros(shape=(self.encoded_dim,)), name='b3')
        self.w4 = tf.Variable(tf.random_normal(shape=(self.encoded_dim, 32), stddev=0.01, seed=10), name='w4')
        self.b4 = tf.Variable(tf.zeros(shape=(32,)), name='b4')
        self.w5 = tf.Variable(tf.random_normal(shape=(32, 64), stddev=0.01, seed=10), name='w5')
        self.b5 = tf.Variable(tf.zeros(shape=(64,)), name='b5')
        self.w6 = tf.Variable(tf.random_normal(shape=(64, 115), stddev=0.01, seed=10), name='w6')
        self.b6 = tf.Variable(tf.zeros(shape=(115,)), name='b6')

        self.encoded, self.decoded = self.ae_setup(self.input)

        self.reconstruct_loss = tf.losses.mean_squared_error(self.decoded, self.input)
        self.loss_pretrain = self.reconstruct_loss  # +self.w_loss*0.01
        self.optimizer_pretrain = tf.train.AdamOptimizer(0.005).minimize(self.loss_pretrain)

    def ae_setup(self, input):
        x = tf.nn.relu(tf.matmul(input, self.w1) + self.b1)
        x = tf.nn.relu(tf.matmul(x, self.w2) + self.b2)
        encoded = (tf.matmul(x, self.w3) + self.b3)
        x = tf.nn.relu(tf.matmul(encoded, self.w4) + self.b4)
        x = tf.nn.relu(tf.matmul(x, self.w5) + self.b5)
        decoded = (tf.matmul(x, self.w6) + self.b6)
        return encoded, decoded

    def train(self, Data, BATCH_SIZE, pre_train_steps=20000):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=None)
        # '''# pre-train'''
        ae_ckpt_path = os.path.join('pre_ae_ckpt', f'model_autoencoder_{self.encoded_dim}.ckpt')
        if 1 - os.path.exists(ae_ckpt_path + '.index'):
            # pre_train_steps = 20000
            for i_step_pre in range(pre_train_steps):
                index_list = [i for i in range(Data.DATA_NUM)]
                idxs = random.sample(index_list, BATCH_SIZE)
                train_x_batch = Data.unlabeled_data_v_r[idxs]
                encoded_, decoded_, _, loss_pretrain_ = sess.run([self.encoded, self.decoded,
                                                                  self.optimizer_pretrain,
                                                                  self.loss_pretrain],
                                                                 feed_dict={self.input: train_x_batch,
                                                                            self.input_batch_size: BATCH_SIZE})
                if i_step_pre % 1000 == 0:
                    print(f'encoded_dim = {self.encoded_dim}, steps={i_step_pre}, reconstruct loss = {loss_pretrain_}')
            saver.save(sess, ae_ckpt_path)
        else:
            saver.restore(sess, ae_ckpt_path)
            index_list = [i for i in range(Data.DATA_NUM)]
            idxs = random.sample(index_list, BATCH_SIZE)
            train_x_batch = Data.unlabeled_data_v_r[idxs]
            encoded_, decoded_, _, loss_pretrain_ = sess.run([self.encoded, self.decoded,
                                                              self.optimizer_pretrain,
                                                              self.loss_pretrain],
                                                             feed_dict={self.input: train_x_batch,
                                                                        self.input_batch_size: BATCH_SIZE})

            print(f'encoded_dim = {self.encoded_dim}, reconstruct loss = {loss_pretrain_}')

        sess.close()
        return loss_pretrain_


class svm_classifier(object):
    def __init__(self):
        self.clf = svm.SVC(kernel='linear')
    def train(self, train_x, train_y):
        if np.unique(train_y).size > 1:
            self.clf.fit(train_x, train_y)
        else:
            self.clf.fit(train_x[0:2, ::], [1,0])
    def decision_function(self,x):
        predict_label = self.clf.decision_function(x)
        return predict_label

class cluster_NN(object):
    def __init__(self, encoded_dim=4,cluster_dim=4):
        self.n_cluster = 2
        self.encoded_dim = encoded_dim
        self.cluster_dim = cluster_dim
        self.kmeans = KMeans(n_clusters=self.n_cluster, n_init=20)
        self.Classifier = svm_classifier()
        self.Classifier_std = preprocessing.StandardScaler()
        self.all_colors = ['#2E94B9', "#fa625f", '#62C8A5', '#f29c2b']
        # model setup
        tf.reset_default_graph()
        self.input = tf.placeholder(tf.float32, shape=[None, 115])
        self.input_batch_size = tf.placeholder(tf.int32, shape=())
        self.w1 = tf.Variable(tf.random_normal(shape=(115, 64), stddev=0.01,seed=10), name='w1')
        self.b1 = tf.Variable(tf.zeros(shape=(64,)), name='b1')
        self.w2 = tf.Variable(tf.random_normal(shape=(64, 32), stddev=0.01,seed=10), name='w2')
        self.b2 = tf.Variable(tf.zeros(shape=(32,)), name='b2')
        self.w3 = tf.Variable(tf.random_normal(shape=(32, self.encoded_dim), stddev=0.01,seed=10), name='w3')
        self.b3 = tf.Variable(tf.zeros(shape=(self.encoded_dim,)), name='b3')
        self.w4 = tf.Variable(tf.random_normal(shape=(self.encoded_dim, 32), stddev=0.01,seed=10), name='w4')
        self.b4 = tf.Variable(tf.zeros(shape=(32,)), name='b4')
        self.w5 = tf.Variable(tf.random_normal(shape=(32, 64), stddev=0.01,seed=10), name='w5')
        self.b5 = tf.Variable(tf.zeros(shape=(64,)), name='b5')
        self.w6 = tf.Variable(tf.random_normal(shape=(64, 115), stddev=0.01,seed=10), name='w6')
        self.b6 = tf.Variable(tf.zeros(shape=(115,)), name='b6')

        self.encoded, self.decoded = self.ae_setup(self.input)
        self.w7 = tf.Variable(tf.random_normal(shape=(self.encoded_dim, self.cluster_dim), stddev=1, seed=10), name='w7')
        self.b7 = tf.Variable(tf.zeros(shape=(self.cluster_dim,)), name='b7')
        self.cluster_features_data = self.clusnn_setup(self.encoded)
        self.cluster_features = self.cluster_features_data

        self.reconstruct_loss = tf.losses.mean_squared_error(self.decoded, self.input)

        # self.w_loss = -tf.reduce_sum(tf.abs(self.w7))
        # 增加一些飞行员观测作为known information
        self.pilot_input = tf.placeholder(tf.float32, shape=[None, 115])
        self.pilot_size = tf.placeholder(tf.int32, shape=())
        self.pilot_encoded, self.pilot_decoded = self.ae_setup(self.pilot_input)
        self.pilot_cluster_features_data = self.clusnn_setup(self.pilot_encoded)
        self.pilot_center = tf.reshape(self.pilot_input[0, ::],
                                       [1, -1])  # tf.reshape(tf.reduce_mean(self.pilot_input,axis = 0),[1,-1])
        center_encoded, center_decoded = self.ae_setup(self.pilot_center)
        self.pilot_center_cluster_features = self.clusnn_setup(center_encoded)
        self.pilot_center_dist = (self.pilot_cluster_features_data - self.pilot_center_cluster_features) ** 2
        self.pilot_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(self.pilot_center_dist, axis=1)+1e-4))  # 使飞行员分类在第一类

        # 增加一些均匀风场作为known information
        self.calm_input = tf.placeholder(tf.float32, shape=[None, 115])
        self.calm_size = tf.placeholder(tf.int32, shape=())
        self.calm_encoded, self.calm_decoded = self.ae_setup(self.calm_input)
        self.calm_cluster_features_data = self.clusnn_setup(self.calm_encoded)
        self.calm_center = tf.reduce_mean(self.calm_input,axis = 0,keepdims=True)
        center_encoded, center_decoded = self.ae_setup(self.calm_center)
        self.calm_center_cluster_features = self.clusnn_setup(center_encoded)  # self.clusnn_setup(center_encoded)
        self.calm_center_dist = (self.calm_cluster_features_data - self.calm_center_cluster_features) ** 2
        self.calm_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(self.calm_center_dist, axis=1)+1e-4))

        self.B_update = tf.concat([self.pilot_center_cluster_features, self.calm_center_cluster_features], axis=0)
        self.dist = self._pairwise_euclidean_distance(self.cluster_features, self.B_update, self.input_batch_size,
                                                      self.n_cluster)
        self.pred = tf.argmin(self.dist, 1)
        self.T = (tf.one_hot(self.pred, self.n_cluster))
        q = 1.0 / (1.0 + self.dist ** 2)  # REF: dec
        self.q = (q / tf.reduce_sum(q, axis=1, keepdims=True))
        self.p = tf.placeholder(tf.float32, shape=(None, self.n_cluster))
        self.loss_dec = self._kl_divergence(self.p, self.q)

        calm_pilot_interval = self._pairwise_euclidean_distance(self.calm_cluster_features_data,
                                                                self.pilot_cluster_features_data, self.calm_size,
                                                                self.pilot_size)
        self.max_interval = -(tf.reduce_mean(calm_pilot_interval) * 1)#tf.log
        self.w_loss = tf.reduce_mean(tf.abs(self.w1)**2)+tf.reduce_mean(tf.abs(self.w2)**2)+\
                      tf.reduce_mean(tf.abs(self.w3)**2)+tf.reduce_mean(tf.abs(self.w4)**2)+\
                      tf.reduce_mean(tf.abs(self.w5)**2)+tf.reduce_mean(tf.abs(self.w6)**2)+tf.reduce_mean(tf.abs(self.w7)**2)

        self.loss_supervise = 1 * self.reconstruct_loss + 1 * self.loss_dec+1* self.pilot_loss+ 1*self.calm_loss+ 0.7 * self.max_interval+1*self.w_loss
        self.optimizer_dec = tf.train.AdamOptimizer(0.0001).minimize(self.loss_supervise)
        self.loss_pretrain = self.reconstruct_loss  # +self.w_loss*0.01
        self.optimizer_pretrain = tf.train.AdamOptimizer(0.005).minimize(self.loss_pretrain)

    def clusnn_setup(self, encoded):
        # temp_feature = tf.nn.relu(tf.matmul(encoded, self.w7) + self.b7)#做非线性变换
        cluster_features = (tf.matmul(encoded, self.w7) + self.b7)  # 做线性变换tf.nn.sigmoid
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
        # init mu
        kmeans = self.kmeans.fit(features)
        kmeans.cluster_centers_ = kmeans.cluster_centers_[np.argsort(np.sum(kmeans.cluster_centers_**2,1))]#按距离原点的距离将两个类簇划分为1和2，避免每次随机指定1，2类簇
        return tf.assign(self.B_update, kmeans.cluster_centers_)

    def train(self, Data, pilot_semi_, train_steps, BATCH_SIZE, imbalance_beta_values=np.array([1,10,100,1000]),
              If_cal_AUC = False, step_cal_AUC = None, If_plot_hazard_factor_distribution = False,
              If_plot_tSNE = False, tsne_num_list=[0,10,25,100,1000], If_tsne_3d = False,
                          If_plot_hazard_value = False, If_plot_ROC = False, If_plot_tSNE_encoded = False,
                          If_other_airports = False, other_airports_v = None, If_pred = False, 
                          calm_test = None, pilot_test = None):
        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=None)
        # '''# pre-train'''
        ae_ckpt_path = os.path.join('ae_ckpt',f'model_encoder_encodedDim{self.encoded_dim}_cluaterDim{self.cluster_dim}.ckpt')
        if 1-os.path.exists(ae_ckpt_path+'.index'):
            pre_train_steps = 20000
            for i_step_pre in range(pre_train_steps):
                index_list = [i for i in range(Data.DATA_NUM)]
                idxs = random.sample(index_list, BATCH_SIZE)
                train_x_batch = Data.unlabeled_data_v_r[idxs]
                encoded_, decoded_, _, loss_pretrain_ = sess.run([self.encoded, self.decoded,
                                                                  self.optimizer_pretrain,
                                                                  self.loss_pretrain],
                                                                 feed_dict={self.input: train_x_batch,
                                                                            self.input_batch_size: BATCH_SIZE})
                if i_step_pre % 1000 == 0:
                    print(f'{i_step_pre}, reconstruct loss = {loss_pretrain_}')
            saver.save(sess, ae_ckpt_path)
        else:
            saver.restore(sess, ae_ckpt_path)

        '''# semi-supervise '''
        sdec_ckpt_path = os.path.join('sdec_ckpt', 'model_supervise.ckpt')
        # saver.restore(sess,sdec_ckpt_path)

        metric_values = {'ACC_TRAIN':np.zeros((train_steps,1)),
                        'PTA':np.zeros((train_steps,1)),
                        'ACC_TEST':np.zeros((train_steps,1)),
                        'AUC':np.zeros((train_steps,imbalance_beta_values.size)),
                        'CSI':np.zeros((train_steps,imbalance_beta_values.size))}

        # encoded_, decoded_, cluster_features_, q_, pred_, T_, dist_ = sess.run([self.encoded, self.decoded,
        #                                                                         self.cluster_features, self.q,
        #                                                                         self.pred, self.T,
        #                                                                         self.dist],
        #                                                                        feed_dict={
        #                                                                            self.input: Data.unlabeled_data_v_r,
        #                                                                            self.input_batch_size: Data.DATA_NUM
        #                                                                            })  # clusternn.cluster_features_phys:v_diff_phy
        # sess.run(self.get_assign_cluster_centers_op(cluster_features_))


        for i_step in range(train_steps):
            pilot_train_encoded_, _, pilot_train_cluster_features_ , pilot_train_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,pilot_semi_)
            calm_train_encoded_, _, calm_train_cluster_features_, calm_train_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,Data.calm_train)
            pilot_cluster_nums = [np.sum(pilot_train_pred_ == i) for i in range(self.n_cluster)]
            hazard_id = np.argmax(pilot_cluster_nums)

            if If_other_airports&(i_step == 500):
                _, _, cluster_features_all_, pred_all_ = self.predict_in_sess(pilot_semi_,Data,sess, Data.unlabeled_data_v_r)
                all_hazard_factor = self.hazard_factor_calculate(pilot_semi_,Data,sess,hazard_id,cluster_features_all_)
                _, _, other_cluster_features_all_, other_pred_all_ = self.predict_in_sess(pilot_semi_,Data,sess, other_airports_v)
                other_airports_hazard_factor = self.hazard_factor_calculate(pilot_semi_,Data,sess,hazard_id,other_cluster_features_all_)
                _, _, pilot_test_cluster_features_, pilot_test_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,Data.pilot_test_v_r)
                pilot_test_hazard_factor = self.hazard_factor_calculate(pilot_semi_,Data,sess,hazard_id,pilot_test_cluster_features_)

                mu = np.mean(all_hazard_factor)
                sigma = np.std(all_hazard_factor)
                all_hazard_factor_norm = (all_hazard_factor - mu) / sigma
                other_airports_hazard_factor_norm = (other_airports_hazard_factor - mu) / sigma
                pilot_test_hazard_factor_norm = (pilot_test_hazard_factor - mu) / sigma
                self.cluster_visualization_tsne_test_other_airports(hazard_id, 
                                                    cluster_features_all_, pilot_test_cluster_features_,other_cluster_features_all_, 
                                                    pred_all_, pilot_test_pred_,other_pred_all_,
                                                    all_hazard_factor_norm,pilot_test_hazard_factor_norm,
                                                    other_airports_hazard_factor_norm)
                # scio.savemat('pilot_interval_hazard_factors_2h.mat',{'other_airports_hazard_factor':other_airports_hazard_factor,
                # 
                #                                                   'other_pred_all_':other_pred_all_})
                AUC, _, fpr, tpr = self.roc_calculate(sess, hazard_id, Data,[800],pilot_semi_)
                return AUC,fpr, tpr, pilot_test_hazard_factor,all_hazard_factor
            if If_pred&(i_step == train_steps-1):
                if calm_test is None:
                    AUCs_pred = []
                    AUCs_truth = []
                    corr_ = []
                    corr_cos_ = []
                    ahead_time = []
                    time_bin = 10
                    for interval_i in range(1,6):#[1,3]:
                        for ahead_frame in range(1,11):#:[10]
                            data_path = f'D:\\HKG1\\revise2\\program\\predict\\prediction_models\prediction_models_{interval_i}_INTERVAL\\prediction_results_{ahead_frame}_ahead'
                            pilot_save_path = os.path.join(data_path,'predictions_nn_pilots.mat')
                            calm_save_path = os.path.join(data_path,'predictions_nn_calms.mat')
                            pilot_save_ = scio.loadmat(pilot_save_path)
                            pilot_pred_corridors = pilot_save_['pred_corridors']
                            print(pilot_pred_corridors.shape[0])
                            pilot_truth_corridors = pilot_save_['truth_corridors']
                            pilot_pred_time_aheads = pilot_save_['pred_time_aheads'][0]
                            calm_save_ = scio.loadmat(calm_save_path)
                            calm_pred_corridors = calm_save_['pred_corridors']
                            calm_truth_corridors = calm_save_['truth_corridors']
                            calm_pred_time_aheads = calm_save_['pred_time_aheads'][0]

                            ahead_time.append(list(pilot_pred_time_aheads)+list(calm_pred_time_aheads))

                            pilots_headwinds = Data.std.transform(pilot_pred_corridors)
                            calm_headwinds = Data.std.transform(calm_pred_corridors)
                            calm_test = calm_headwinds
                            pilot_test = pilots_headwinds
                            _, _, calm_test_cluster_features_, calm_test_pred_ = self.predict_in_sess(pilot_semi_,Data,sess, calm_test)
                            _, _, pilot_test_cluster_features_, pilot_test_pred_ = self.predict_in_sess(pilot_semi_,Data,sess, pilot_test)
                            calm_test_hazard_factor_ = self.hazard_factor_calculate(pilot_semi_,Data,sess,hazard_id,calm_test_cluster_features_)
                            pilot_test_hazard_factor_ = self.hazard_factor_calculate(pilot_semi_,Data,sess,hazard_id,pilot_test_cluster_features_)
                            AUC,CSI = self.roc_calculate_pred(sess, hazard_id, Data,pilot_semi_,i_step,
                                    calm_test_cluster_features_,calm_test_pred_,
                                    pilot_test_cluster_features_,pilot_test_pred_,
                                If_plot_hazard_value)
                            AUCs_pred.append(AUC)
                            _, _, cluster_features_all_, pred_all_ = self.predict_in_sess(pilot_semi_,Data,sess, Data.unlabeled_data_v_r)
                            all_hazard_factor = self.hazard_factor_calculate(pilot_semi_,Data,sess,hazard_id,cluster_features_all_)
                            mu = np.mean(all_hazard_factor)
                            sigma = np.std(all_hazard_factor)
                            all_hazard_factor_norm = (all_hazard_factor - mu) / sigma
                            pilot_test_hazard_factor_norm = (pilot_test_hazard_factor_ - mu) / sigma
                            # thre_norm = (thre_ - mu) / sigma
                            pred_hazard_factor_ = pilot_test_hazard_factor_norm
                            if ahead_frame == 10:
                                # self.cluster_visualization_tsne_test_pred(cluster_features_all_,pilot_test_cluster_features_,
                                #              pred_all_, all_hazard_factor_norm,pilot_test_hazard_factor_norm,interval_i)
                                temp_a = 2
                            pilots_headwinds = Data.std.transform(pilot_truth_corridors)
                            calm_headwinds = Data.std.transform(calm_truth_corridors)
                            calm_test = calm_headwinds
                            pilot_test = pilots_headwinds
                            _, _, calm_test_cluster_features_, calm_test_pred_ = self.predict_in_sess(pilot_semi_,Data,sess, calm_test)
                            _, _, pilot_test_cluster_features_, pilot_test_pred_ = self.predict_in_sess(pilot_semi_,Data,sess, pilot_test)
                            calm_test_hazard_factor_ = self.hazard_factor_calculate(pilot_semi_,Data,sess,hazard_id,calm_test_cluster_features_)
                            pilot_test_hazard_factor_ = self.hazard_factor_calculate(pilot_semi_,Data,sess,hazard_id,pilot_test_cluster_features_)
                            # from sklearn.metrics.pairwise import cosine_similarity
                            
                            from scipy.stats import pearsonr
                            r_cos_ = cosine_similarity(pred_hazard_factor_.reshape(1, -1),pilot_test_hazard_factor_.reshape(1, -1))
                            r_ = pearsonr(pred_hazard_factor_,pilot_test_hazard_factor_)
                            corr_.append(r_)
                            corr_cos_.append(r_cos_)
                            AUC,CSI = self.roc_calculate_pred(sess, hazard_id, Data,pilot_semi_,i_step,
                                    calm_test_cluster_features_,calm_test_pred_,
                                    pilot_test_cluster_features_,pilot_test_pred_,
                                If_plot_hazard_value)
                            AUCs_truth.append(AUC)
                            pilot_test_hazard_factor_norm = (pilot_test_hazard_factor_ - mu) / sigma
                            # plt.plot(pred_hazard_factor_,label='prediction')
                            # plt.plot(pilot_test_hazard_factor_norm,label='ground truth')
                            # plt.legend()
                            # # plt.show(block=True)
                            # plt.savefig(f'../figures/predict_cluster_performance_hazard_factor_3.pdf')
                            # if (interval_i == 1)&(ahead_frame == 1): # model is=0表示truth  
                            #     self.cluster_visualization_tsne_test_pred(hazard_id, cluster_features_all_,pilot_test_cluster_features_,calm_test_cluster_features_,
                            #                             pred_all_, pilot_test_pred_,calm_test_pred_,interval_i-1)
                           
                    ahead_time_mean = ([int(np.mean(ahead_time[i])) for i in range(len(ahead_time))])
                    time_intervals = np.sort(np.unique(ahead_time_mean))
                    x_s = []
                    y_s = []
                    z_s = []
                    w_s = []
                    for i_time in time_intervals:
                        idxs = np.where(np.array(ahead_time_mean)==i_time)[0]
                        # plt.plot(np.array(ahead_time_mean)[idxs],np.array(AUCs_pred)[idxs],'.')
                        x_s.append(np.array(ahead_time_mean)[idxs])
                        y_s.append(np.array(AUCs_pred)[idxs])
                        z_s.append(np.array(corr_).reshape(50,2)[idxs,0])
                        w_s.append(np.array(corr_cos_))
                    from scipy.optimize import curve_fit
                    def curv_func(x, a, b, c,d,e):
                        return a*x**4+b*x**3+c*x**2+d*x+e
                    scio.savemat('plot_predict_cluster_performance_h.mat',{'x_s':x_s,'y_s':y_s,'z_s':z_s,'w_s':w_s})


                else:
                    rams_path = 'D:\\HKG1\\revise2\\wrf_hk\\'
                    pilot_path = os.path.join(rams_path,'rams_data_pilots_in_2h.mat')
                    calm_path = os.path.join(rams_path,'rams_data_calms_in_2h.mat')
                    calm_ram = scio.loadmat(calm_path)
                    calm_ahead_seconds = calm_ram['calm_ahead_seconds']
                    calm_headwinds = calm_ram['calm_headwinds']
                    calm_time = calm_ram['calm_time']
                    calm_index_valid = np.where(calm_ahead_seconds<=3600)[1]
                    calm_headwinds = calm_headwinds[calm_index_valid,::]
                    calm_time = calm_time[calm_index_valid]
                    pilot_ram = scio.loadmat(pilot_path)
                    pilots_ahead_seconds = pilot_ram['pilots_ahead_seconds']
                    pilots_headwinds = pilot_ram['pilots_headwinds']
                    pilots_time = pilot_ram['pilots_time']
                    pilots_index_valid = np.where(pilots_ahead_seconds<=3600)[1]
                    pilots_headwinds = pilots_headwinds[pilots_index_valid,::]
                    pilots_time = pilots_time[pilots_index_valid]
                    pilots_headwinds = Data.std.transform(pilots_headwinds.squeeze())
                    calm_headwinds = Data.std.transform(calm_headwinds.squeeze())
                    pilot_test = pilots_headwinds
                    calm_test = calm_headwinds
                    _, _, calm_test_cluster_features_, calm_test_pred_ = self.predict_in_sess(pilot_semi_,Data,sess, calm_test)
                    _, _, pilot_test_cluster_features_, pilot_test_pred_ = self.predict_in_sess(pilot_semi_,Data,sess, pilot_test)
                    calm_test_hazard_factor_ = self.hazard_factor_calculate(pilot_semi_,Data,sess,hazard_id,calm_test_cluster_features_)
                    pilot_test_hazard_factor_ = self.hazard_factor_calculate(pilot_semi_,Data,sess,hazard_id,pilot_test_cluster_features_)
                    _, _, cluster_features_all_, pred_all_ = self.predict_in_sess(pilot_semi_,Data,sess, Data.unlabeled_data_v_r)
                    all_hazard_factor = self.hazard_factor_calculate(pilot_semi_,Data,sess,hazard_id,cluster_features_all_)
                    mu = np.mean(all_hazard_factor)
                    sigma = np.std(all_hazard_factor)
                    all_hazard_factor_norm = (all_hazard_factor - mu) / sigma
                    pilot_test_hazard_factor_norm = (pilot_test_hazard_factor_ - mu) / sigma
                    calm_test_hazard_factor_norm = (calm_test_hazard_factor_ - mu) / sigma
                    AUCs,CSIs = self.roc_calculate_pred(sess, hazard_id, Data,pilot_semi_,i_step,
                            calm_test_cluster_features_,calm_test_pred_,
                            pilot_test_cluster_features_,pilot_test_pred_,
                        If_plot_hazard_value)
                    
                    import matplotlib.gridspec as gridspec
                    fig = plt.figure()
                    gs = gridspec.GridSpec(2, 5, height_ratios=[2, 1])
                    ax1 = plt.subplot(gs[1,0:5])
                    ax1.plot(pilot_test_hazard_factor_norm,'o-')
                    ax1.set_xticks(np.arange(3,30,6))
                    for i_ax in range(5):
                        ax2 = plt.subplot(gs[0,i_ax])
                        cf = ax2.pcolor(pilots_headwinds.squeeze().T[::,i_ax*6:(i_ax+1)*6],cmap='Spectral_r')
                        ax2.set_yticks(())
                        ax2.set_xticks([3])
                        # ax2.vlines(np.arange(0,30,1),0,115)
                        fig.colorbar(cf, orientation='horizontal', extend='both')
                    fig.subplots_adjust(wspace=0)
                    plt.savefig(f'../figures/Pred_wrf.png',dpi=600)  #
                    plt.savefig(f'../figures/Pred_wrf.pdf')
                    self.cluster_visualization_tsne_test_pred_wrf(cluster_features_all_,pilot_test_cluster_features_,
                                             pred_all_, all_hazard_factor_norm,pilot_test_hazard_factor_norm,pilots_time)
                    pilot_ram = scio.loadmat('D:\\HKG1\\revise2\\wrf_hk\\rams_data_pilots_in_2h.mat')
                    pilots_ahead_seconds = pilot_ram['pilots_ahead_seconds']
                    pilots_headwinds = pilot_ram['pilots_headwinds']
                    pilots_time = pilot_ram['pilots_time']
                    plt.plot(1-pilot_test_pred_)
                    plt.plot(pilot_test_hazard_factor_)
                    plt.xticks(np.arange(pilot_test_pred_.size),pilots_ahead_seconds[0])
                    
            index_list = [i for i in range(Data.DATA_NUM)]
            idxs = random.sample(index_list, BATCH_SIZE)
            train_x_batch = Data.unlabeled_data_v_r[idxs]
            encoded_, decoded_, cluster_features_, q_, pred_, T_, dist_ = sess.run([self.encoded, self.decoded,
                                                                             self.cluster_features, self.q,
                                                                             self.pred, self.T,self.dist],
                                                                            feed_dict={self.input: train_x_batch,
                                                                                       self.input_batch_size: BATCH_SIZE,
                                    self.calm_input: Data.calm_train, self.calm_size: Data.calm_train.shape[0],
                                    self.pilot_input: pilot_semi_,
                                    self.pilot_size: pilot_semi_.shape[0]
                                                                                       })
            p_ = self.target_distribution(q_)
            encoded_, decoded_, pred_, q, reconstruct_loss_, _,\
            cluster_loss0, loss0_, calm_loss_, pilot_loss_, \
            max_interval_, pilot_cluster_features_data_, pilot_center_dist_, pilot_center_cluster_features_, B_update_ = \
                sess.run([self.encoded, self.decoded, self.pred, self.q,
                          self.reconstruct_loss,self.optimizer_dec,
                          self.loss_dec, self.loss_supervise, self.calm_loss,
                          self.pilot_loss, self.max_interval, self.pilot_cluster_features_data,
                          self.pilot_center_dist,self.pilot_center_cluster_features,self.B_update],
                         feed_dict={self.input: train_x_batch, self.input_batch_size: BATCH_SIZE,
                                    self.calm_input: Data.calm_train, self.calm_size: Data.calm_train.shape[0],
                                    self.pilot_input: pilot_semi_,
                                    self.pilot_size: pilot_semi_.shape[0],self.p:p_})#
            if i_step % 1 == 0:
                encoded_, decoded_, cluster_features_, pred_= self.predict_in_sess(pilot_semi_,Data, sess,train_x_batch)
                pilot_train_encoded_, pilot_train_decoded_, pilot_train_cluster_features_ , pilot_train_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,pilot_semi_)
                calm_train_encoded_,calm_train_decoded_, calm_train_cluster_features_, calm_train_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,Data.calm_train)
                pilot_cluster_nums = [np.sum(pilot_train_pred_ == i) for i in range(self.n_cluster)]
                hazard_id = np.argmax(pilot_cluster_nums)
                train_acc_ = np.max(pilot_cluster_nums) / pilot_train_pred_.shape[0]
                train_pta_ = (np.sum(pred_ == hazard_id) / pred_.shape[0])

                pilot_test_encoded_, pilot_test_decoded_, test_cluster_features_, pilot_test_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,Data.pilot_test_v_r)
                test_acc = np.sum(pilot_test_pred_ == hazard_id) / pilot_test_pred_.shape[0]
                metric_values['ACC_TRAIN'][i_step, 0] = train_acc_
                metric_values['PTA'][i_step, 0] = train_pta_
                metric_values['ACC_TEST'][i_step, 0] = test_acc
                print(f'{i_step}, total loss = {loss0_}, reconstruct loss = {reconstruct_loss_}, '
                      f'cluster loss = {cluster_loss0}, calm loss = {calm_loss_}, pilot loss = {pilot_loss_}, max interval = {max_interval_}, ')
                print(f'Train ACC: {train_acc_}, PTA: {train_pta_}, Test ACC: {test_acc}')
                # print(B_update_)
        saver.save(sess, sdec_ckpt_path)
        _, _, cluster_features_all_, pred_all_ = self.predict_in_sess(pilot_semi_,Data,sess,Data.unlabeled_data_v_r)
        sess.close()
        return  metric_values
    def hazard_factor_calculate(self,pilot_semi_,Data,sess,hazard_id,other_cluster_features_all_):
        _, _, cluster_features_all_, pred_all_ = self.predict_in_sess(pilot_semi_,Data, sess, Data.unlabeled_data_v_r)
        _, _, pilot_train_cluster_features_ , pilot_train_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,pilot_semi_)
        _,_, calm_train_cluster_features_, calm_train_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,Data.calm_train)

        ratio_used_for_svm = 0.05
        index_list = [i for i in range(pred_all_.shape[0])]
        num_used_for_svm = int(ratio_used_for_svm * pred_all_.shape[0])
        idxs = random.sample(index_list, num_used_for_svm)
        if hazard_id == 0:
            pred_all_ = 1 - pred_all_
        add_train_x = np.vstack((pilot_train_cluster_features_, calm_train_cluster_features_))
        add_train_y = np.hstack((np.zeros(pilot_train_pred_.shape) + 1, np.zeros(calm_train_pred_.shape)))
        train_x = np.vstack((add_train_x,cluster_features_all_[idxs,::]))
        train_y = np.hstack((add_train_y,pred_all_[idxs]))
        train_x_std = self.Classifier_std.fit_transform(train_x)
        self.Classifier.train(train_x=train_x_std, train_y=train_y)
        other_airports_x_std = self.Classifier_std.transform(other_cluster_features_all_)
        other_airports_hazard_factor = self.Classifier.decision_function(other_airports_x_std)
        return other_airports_hazard_factor
    def predict_in_sess(self,pilot_semi_,Data,sess,x):
        encoded_, decoded_, cluster_features_, pred_ = sess.run([self.encoded, self.decoded,
                                                                 self.cluster_features, self.pred],
                                                                feed_dict={self.input: x,
                                                                           self.input_batch_size: x.shape[0],
                                    self.calm_input: Data.calm_train, self.calm_size: Data.calm_train.shape[0],
                                    self.pilot_input: pilot_semi_,
                                    self.pilot_size: pilot_semi_.shape[0]})
        return encoded_, decoded_, cluster_features_, pred_
    def predict(self,x):
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=None)
            sdec_ckpt_path = os.path.join('sdec_ckpt', 'model_supervise.ckpt')
            saver.restore(sess, sdec_ckpt_path)
            encoded_, decoded_, cluster_features_, pred_ = sess.run([self.encoded, self.decoded,
                                                                     self.cluster_features, self.pred],
                                                                    feed_dict={self.input: x,
                                                                               self.input_batch_size: x.shape[0]})
        return encoded_, decoded_, cluster_features_, pred_
    def roc_calculate_pred(self, sess, hazard_id, Data,pilot_semi_,i_step,
                           calm_test_cluster_features_,calm_test_pred_,
                           pilot_test_cluster_features_,pilot_test_pred_,
                      If_plot_hazard_value):
        _, _, cluster_features_all_, pred_all_ = self.predict_in_sess(pilot_semi_,Data, sess, Data.unlabeled_data_v_r)
        _, _, pilot_train_cluster_features_ , pilot_train_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,pilot_semi_)
        _,_, calm_train_cluster_features_, calm_train_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,Data.calm_train)

        ratio_used_for_svm = 0.05
        index_list = [i for i in range(pred_all_.shape[0])]
        num_used_for_svm = int(ratio_used_for_svm * pred_all_.shape[0])
        idxs = random.sample(index_list, num_used_for_svm)
        if hazard_id == 0:
            pred_all_ = 1 - pred_all_
        add_train_x = np.vstack((pilot_train_cluster_features_, calm_train_cluster_features_))
        add_train_y = np.hstack((np.zeros(pilot_train_pred_.shape) + 1, np.zeros(calm_train_pred_.shape)))
        train_x = np.vstack((add_train_x,cluster_features_all_[idxs,::]))
        train_y = np.hstack((add_train_y,pred_all_[idxs]))
        train_x_std = self.Classifier_std.fit_transform(train_x)
        self.Classifier.train(train_x=train_x_std, train_y=train_y)
        test_dataset_x = np.vstack((calm_test_cluster_features_, pilot_test_cluster_features_))
        test_dataset_y = np.hstack((np.zeros(calm_test_pred_.shape), np.zeros(pilot_test_pred_.shape) + 1))
        test_dataset_x_std = self.Classifier_std.transform(test_dataset_x)
        test_predict_label = self.Classifier.decision_function(test_dataset_x_std)
        if If_plot_hazard_value:
            self.plot_hazard_factor_value(test_predict_label,i_step)
        fpr, tpr, threshold = roc_curve(test_dataset_y, test_predict_label, drop_intermediate=False,
                                            pos_label=1)
        maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))

        AUC, CSI = calculate_metrics(decision_values=test_predict_label, truth=test_dataset_y, hazard_id = 1)
        # scio.savemat('pred_clusternn.mat',{'decision_values':test_predict_label, 'truth':test_dataset_y, 'AUC': AUC})
        print(f'AUC={AUC}, CSI = {CSI}')
        return AUC,CSI
    def roc_calculate(self, sess, hazard_id, Data,imbalance_beta_values,pilot_semi_):
        _, _, cluster_features_all_, pred_all_ = self.predict_in_sess(pilot_semi_,Data, sess, Data.unlabeled_data_v_r)
        _, _, pilot_test_cluster_features_, pilot_test_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,Data.pilot_test_v_r)
        _, _, pilot_train_cluster_features_ , pilot_train_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,pilot_semi_)
        _,_, calm_train_cluster_features_, calm_train_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,Data.calm_train)

        ratio_used_for_svm = 0.05
        index_list = [i for i in range(pred_all_.shape[0])]
        num_used_for_svm = int(ratio_used_for_svm * pred_all_.shape[0])
        idxs = random.sample(index_list, num_used_for_svm)
        if hazard_id == 0:
            pred_all_ = 1 - pred_all_
        add_train_x = np.vstack((pilot_train_cluster_features_, calm_train_cluster_features_))
        add_train_y = np.hstack((np.zeros(pilot_train_pred_.shape) + 1, np.zeros(calm_train_pred_.shape)))
        train_x = np.vstack((add_train_x,cluster_features_all_[idxs,::]))
        train_y = np.hstack((add_train_y,pred_all_[idxs]))
        train_x_std = self.Classifier_std.fit_transform(train_x)
        self.Classifier.train(train_x=train_x_std, train_y=train_y)
        # AUCs = np.zeros((1,imbalance_beta_values.size))
        # CSIs = np.zeros((1,1))
        FPR= []
        TPR = []
        Thresholds = []
        for count_imbalance, beta_imbalance in enumerate(imbalance_beta_values):
            index = Data.calm_ratios.index(beta_imbalance)
            calm_test = Data.calm_test_generator(calm_ratio_index=index)
            calm_encoded_, calm_decoded_, calm_cluster_features_, calm_pred_ = self.predict_in_sess(pilot_semi_,Data,sess, calm_test)
            test_dataset_x = np.vstack((calm_cluster_features_, pilot_test_cluster_features_))
            test_dataset_y = np.hstack((np.zeros(calm_pred_.shape), np.zeros(pilot_test_pred_.shape) + 1))
            test_dataset_x_std = self.Classifier_std.transform(test_dataset_x)
            test_predict_label = self.Classifier.decision_function(test_dataset_x_std)
            fpr, tpr, threshold = roc_curve(test_dataset_y, test_predict_label, drop_intermediate=False,
                                            pos_label=1)
            maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
            Thresholds.append(threshold[maxindex])
            AUC, CSI = calculate_metrics(decision_values=test_predict_label, truth=test_dataset_y, hazard_id = 1)
            print(f'AUC={AUC}, CSI = {CSI}')

        return AUC, CSI, fpr, tpr


    def cluster_visualization_tsne_test(self, hazard_id, cluster_features_all_, 
                                        pilot_train_cluster_features_, calm_train_cluster_features_,
                                  pred_all_, pilot_train_pred_,calm_train_pred_,
                                  all_hazard_factor,pilot_train_hazard_factor,calm_train_hazard_factor):
        fig = plt.figure(figsize=(5, 5))
        pca = TSNE(n_components=2)
        ratio_used_for_tsne = 0.04
        index_list = [i for i in range(pred_all_.shape[0])]
        num_used_for_tsne = int(ratio_used_for_tsne * pred_all_.shape[0])
        random.seed(2)
        idxs = random.sample(index_list, num_used_for_tsne)
        all_embedded = pca.fit_transform(np.vstack((cluster_features_all_[idxs,::], pilot_train_cluster_features_, calm_train_cluster_features_)))
        X_unlabel_embedded = all_embedded[:num_used_for_tsne, ::]
        pilot_train_embedded = all_embedded[num_used_for_tsne:-calm_train_pred_.shape[0], ::]
        calm_train_embedded = all_embedded[-calm_train_pred_.shape[0]:, ::]
        # centers_embedded = all_embedded[-2:,::]

        all_hazard_factor_idxs = all_hazard_factor[idxs]
        idx_y_0 = np.argsort(np.abs(all_hazard_factor_idxs))[:20]
        from scipy.optimize import curve_fit
        def curv_func_rmse(x,  c,d,e):
            return c*x**2+d*x+e
        popt, pcov = curve_fit(curv_func_rmse, X_unlabel_embedded[idx_y_0,0], X_unlabel_embedded[idx_y_0,1])
        x_interp = np.arange(np.min(X_unlabel_embedded[idx_y_0,0]),np.max(X_unlabel_embedded[idx_y_0,0]))

        plt.plot(x_interp, curv_func_rmse(x_interp, *popt),'C7',label='y=0')
        if hazard_id == 0:
            pred_all_ = 1 - pred_all_
            pilot_train_pred_ = 1 - pilot_train_pred_
            calm_train_pred_ = 1 - calm_train_pred_
        all_labels = ['All data in cluster 1', 'All data in cluster 2']
        pilot_labels = ['Hazardous wind in cluster 1', 'Hazardous wind in cluster 2']
        calm_labels = ['Calm wind in cluster 1', 'Calm wind in cluster 2']
        clim = (-1,4)
        cf1_1 = plt.scatter(X_unlabel_embedded[:, 0], X_unlabel_embedded[:, 1], marker='o',
                    c=all_hazard_factor_idxs[:],cmap='Spectral_r',alpha=0.5,s = 5,label='Unlabeled trainning data (HongKong)')
        # plt.plot(X_unlabel_embedded[idx_y_0,0],X_unlabel_embedded[idx_y_0,1],'.')

        cf2 = plt.scatter(pilot_train_embedded[::, 0], pilot_train_embedded[::, 1],
                    marker='o', c=pilot_train_hazard_factor[::],cmap='Spectral_r',
                    alpha=1, edgecolors='k', linewidth=0.5, s=25,label='Train hazardous data (HongKong)')
        cf3 = plt.scatter(calm_train_embedded[::, 0], calm_train_embedded[::, 1],
                    marker='s', c=calm_train_hazard_factor[::],cmap='Spectral_r',label='Train calm data (Generated)', alpha=1, edgecolors='k', linewidth=0.5, s=25)
        cbar = fig.colorbar(mappable=cf1_1)
        cbar.mappable.set_clim(clim)
        cbar = fig.colorbar(mappable=cf2)
        cbar.mappable.set_clim(clim)
        cbar = fig.colorbar(mappable=cf3)
        cbar.mappable.set_clim(clim)
        plt.colorbar()
        plt.clim(clim)
        # plt.legend()
        plt.xticks(())
        plt.yticks(())
        plt.savefig(f'../figures/with_linear_layer.png',dpi=600)  #
        plt.savefig(f'../figures/with_linear_layer.pdf')
    def cluster_visualization_tsne_test_pred_wrf(self, cluster_features_all_,pilot_test_cluster_features_,
                                             pred_all_, all_hazard_factor,pilot_test_hazard_factor,pilots_time):
        fig = plt.figure(figsize=(5, 5))
        pca = TSNE(n_components=2)
        ratio_used_for_tsne = 0.04
        index_list = [i for i in range(pred_all_.shape[0])]
        num_used_for_tsne = int(ratio_used_for_tsne * pred_all_.shape[0])
        random.seed(0)
        idxs = random.sample(index_list, num_used_for_tsne)
        all_embedded = pca.fit_transform(np.vstack((cluster_features_all_[idxs,::], pilot_test_cluster_features_)))
        X_unlabel_embedded = all_embedded[:num_used_for_tsne, ::]
        pilot_test_embedded = all_embedded[num_used_for_tsne:, ::]
        all_hazard_factor_idxs = all_hazard_factor[idxs]
        idx_y_0 = np.argsort(np.abs(all_hazard_factor_idxs))[:20]

        from scipy.optimize import curve_fit
        def curv_func_rmse(x,  c,d,e):
            return c*x**2+d*x+e
        popt, pcov = curve_fit(curv_func_rmse, X_unlabel_embedded[idx_y_0,0], X_unlabel_embedded[idx_y_0,1])
        x_interp = np.arange(np.min(X_unlabel_embedded[idx_y_0,0]),np.max(X_unlabel_embedded[idx_y_0,0]))
        plt.plot(x_interp, curv_func_rmse(x_interp, *popt),'C7',label='y=0')

        clim = (-1,4)
        cf1_1 = plt.scatter(X_unlabel_embedded[:, 0], X_unlabel_embedded[:, 1], marker='o',
                    c=all_hazard_factor_idxs[:],cmap='Spectral_r',alpha=0.5,s = 5,label='Unlabeled trainning data')
        # plt.plot(X_unlabel_embedded[idx_y_0,0],X_unlabel_embedded[idx_y_0,1],'.')
        cbar = fig.colorbar(mappable=cf1_1)
        cbar.mappable.set_clim(clim)
        markers = ['o','X','D','^','*']
        size_s = [55,55,55,55,125]
        for i_case in range(5):
            cf2 = plt.scatter(pilot_test_embedded[i_case*6:(i_case+1)*6, 0], pilot_test_embedded[i_case*6:(i_case+1)*6, 1],
                    marker=markers[i_case], c=pilot_test_hazard_factor[i_case*6:(i_case+1)*6],cmap='Spectral_r',
                    alpha=1, edgecolors='k', linewidth=0.5, s=size_s[i_case],label=f'Predicted hazardous data {pilots_time[i_case*6]}')        
            cbar = fig.colorbar(mappable=cf2)
            cbar.mappable.set_clim(clim)
        # plt.legend()
        plt.colorbar(extend = 'both')
        plt.clim(clim)
        plt.xlim((-58.7,-4.2))
        plt.ylim((-83.1,-44.2))
        plt.xticks(())
        plt.yticks(())      
        plt.show()  
        plt.title(f'wrf pred model')
        plt.savefig(f'../figures/PredModel_wrf_zoomin.png',dpi=600)  #
        plt.savefig(f'../figures/PredModel_wrf_zoomin.pdf')
    def cluster_visualization_tsne_test_pred(self, cluster_features_all_,pilot_test_cluster_features_,
                                             pred_all_, all_hazard_factor,pilot_test_hazard_factor,model_id):
        fig = plt.figure(figsize=(5, 5))
        pca = TSNE(n_components=2)
        ratio_used_for_tsne = 0.04
        index_list = [i for i in range(pred_all_.shape[0])]
        num_used_for_tsne = int(ratio_used_for_tsne * pred_all_.shape[0])
        random.seed(0)
        idxs = random.sample(index_list, num_used_for_tsne)
        all_embedded = pca.fit_transform(np.vstack((cluster_features_all_[idxs,::], pilot_test_cluster_features_)))
        X_unlabel_embedded = all_embedded[:num_used_for_tsne, ::]
        pilot_test_embedded = all_embedded[num_used_for_tsne:, ::]
        all_hazard_factor_idxs = all_hazard_factor[idxs]
        idx_y_0 = np.argsort(np.abs(all_hazard_factor_idxs))[:20]

        from scipy.optimize import curve_fit
        def curv_func_rmse(x,  c,d,e):
            return c*x**2+d*x+e
        popt, pcov = curve_fit(curv_func_rmse, X_unlabel_embedded[idx_y_0,0], X_unlabel_embedded[idx_y_0,1])
        x_interp = np.arange(np.min(X_unlabel_embedded[idx_y_0,0]),np.max(X_unlabel_embedded[idx_y_0,0]))
        plt.plot(x_interp, curv_func_rmse(x_interp, *popt),'C7',label='y=0')

        clim = (-1,4)
        cf1_1 = plt.scatter(X_unlabel_embedded[:, 0], X_unlabel_embedded[:, 1], marker='o',
                    c=all_hazard_factor_idxs[:],cmap='Spectral_r',alpha=0.5,s = 5,label='Unlabeled trainning data')
        # plt.plot(X_unlabel_embedded[idx_y_0,0],X_unlabel_embedded[idx_y_0,1],'.')

        cf2 = plt.scatter(pilot_test_embedded[::, 0], pilot_test_embedded[::, 1],
                    marker='o', c=pilot_test_hazard_factor[::],cmap='Spectral_r',
                    alpha=1, edgecolors='k', linewidth=0.5, s=25,label='Predicted hazardous data')        
        cbar = fig.colorbar(mappable=cf1_1)
        cbar.mappable.set_clim(clim)
        cbar = fig.colorbar(mappable=cf2)
        cbar.mappable.set_clim(clim)
        plt.colorbar()
        plt.clim(clim)
        plt.xticks(())
        plt.yticks(())      
        plt.show()  
        plt.title(f'pred model={model_id}')
        plt.savefig(f'../figures/PredModel_{model_id}.png',dpi=600)  #
        plt.savefig(f'../figures/PredModel_{model_id}.pdf')
    def cluster_visualization_tsne_test_other_airports(self,hazard_id, cluster_features_all_, pilot_test_cluster_features_,other_cluster_features_all_, 
                                                    pred_all_, pilot_test_pred_,other_pred_all_,
                                                    all_hazard_factor,pilot_test_hazard_factor,other_airports_hazard_factor):
        fig = plt.figure(figsize=(5, 5))
        pca = TSNE(n_components=2)
        ratio_used_for_tsne = 0.04
        index_list = [i for i in range(pred_all_.shape[0])]
        num_used_for_tsne = int(ratio_used_for_tsne * pred_all_.shape[0])
        random.seed(2)
        idxs = random.sample(index_list, num_used_for_tsne)
        all_embedded = pca.fit_transform(np.vstack((cluster_features_all_[idxs,::], pilot_test_cluster_features_, other_cluster_features_all_)))
        X_unlabel_embedded = all_embedded[:num_used_for_tsne, ::]
        pilot_test_embedded = all_embedded[num_used_for_tsne:-other_pred_all_.shape[0], ::]
        others_embedded = all_embedded[-other_pred_all_.shape[0]:,::]
        all_hazard_factor_idxs = all_hazard_factor[idxs]
        idx_y_0 = np.argsort(np.abs(all_hazard_factor_idxs))[:20]


        from scipy.optimize import curve_fit
        def curv_func_rmse(x,  c,d,e):
            return c*x**2+d*x+e
        popt, pcov = curve_fit(curv_func_rmse, X_unlabel_embedded[idx_y_0,0], X_unlabel_embedded[idx_y_0,1])
        x_interp = np.arange(np.min(X_unlabel_embedded[idx_y_0,0]),np.max(X_unlabel_embedded[idx_y_0,0]))
        plt.plot(x_interp, curv_func_rmse(x_interp, *popt),'C7',label='y=0')

        if hazard_id == 0:
            pred_all_ = 1 - pred_all_
            pilot_test_pred_ = 1 - pilot_test_pred_
            other_pred_all_ = 1 - other_pred_all_
        all_labels = ['All data in cluster 1', 'All data in cluster 2']
        pilot_labels = ['Hazardous wind in cluster 1', 'Hazardous wind in cluster 2']
        calm_labels = ['Calm wind in cluster 1', 'Calm wind in cluster 2']
        clim = (-1,4)
        cf1_1 = plt.scatter(X_unlabel_embedded[:, 0], X_unlabel_embedded[:, 1], marker='o',
                    c=all_hazard_factor_idxs[:],cmap='Spectral_r',alpha=0.5,s = 5,label='Unlabeled trainning data (HongKong)',
                    vmin=-1, vmax=4)
        # plt.plot(X_unlabel_embedded[idx_y_0,0],X_unlabel_embedded[idx_y_0,1],'.')

        cf2 = plt.scatter(pilot_test_embedded[::, 0], pilot_test_embedded[::, 1],
                    marker='o', c=pilot_test_hazard_factor[::],cmap='Spectral_r',
                    alpha=1, edgecolors='k', linewidth=0.5, s=25,label='Test hazardous data (HongKong)',
                    vmin=-1, vmax=4)
        cf3 = plt.scatter(others_embedded[::, 0], others_embedded[::, 1],
                    marker='*', c=other_airports_hazard_factor[::],cmap='Spectral_r',
                    label='Test hazardous data (Other airports)', alpha=1, edgecolors='k', linewidth=0.5, s=125,
                    vmin=-1, vmax=4)
        plt.colorbar()
        plt.show(block=True)
        plt.savefig(f'../figures/other_airports_feature.pdf')

    def plot_ROC(self,FPR,TPR,AUCs,i_step):
        plt.figure(figsize=(5, 5))
        Nu = [1, 10, 100, 1000]  # ,5000
        for i in range(4):
            plt.plot(FPR[i], TPR[i], label=fr'$\beta={Nu[i]}$ (auc = {np.round(AUCs[0, i], 2)})',
                     c=self.all_colors[-i - 1], linewidth=2)

        plt.plot([0, 2], [0, 2], linewidth=0.5, color='C7')  # ,label='Random choice'
        plt.hlines(0.68, 0, 2, colors='C7', linestyles='--')
        plt.text(0.2, 0.69, ('WTWS (TPR=0.68)'))  # ,color = all_colors[1]
        plt.hlines(0.18,0,2,colors='C7',linestyles='-.')
        plt.text(0.05,0.20,('AWARE (TPR=0.18)'))
        plt.xlim((-0.01, 1.01))
        plt.ylim((-0.01, 1.01))
        # plt.xlim((-10.01,-1.01))
        # plt.ylim((-10.01,-1.01))
        plt.legend(frameon=False, loc='lower right')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.show(block=True)
        plt.savefig(f'../figures/{i_step}steps_ROC.png',dpi=600)  #
        plt.savefig(f'../figures/{i_step}steps_ROC.pdf')
        plt.close()

    def seasonal_statistic_fit(self,Data,corridor_months,pilot_semi_):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=None)
        # '''# pre-train'''
        ae_ckpt_path = os.path.join('ae_ckpt',
                                    f'model_encoder_encodedDim{self.encoded_dim}_cluaterDim{self.cluster_dim}.ckpt')
        saver.restore(sess, ae_ckpt_path)

        '''# semi-supervise '''
        sdec_ckpt_path = os.path.join('sdec_ckpt', 'model_supervise.ckpt')
        saver.restore(sess, sdec_ckpt_path)
        _, _, cluster_features_all_, pred_all_ = self.predict_in_sess(pilot_semi_, Data, sess, Data.unlabeled_data_v_r)
        _, _, pilot_train_cluster_features_, pilot_train_pred_ = self.predict_in_sess(pilot_semi_, Data, sess,
                                                                                      pilot_semi_)
        _, _, calm_train_cluster_features_, calm_train_pred_ = self.predict_in_sess(pilot_semi_, Data, sess,
                                                                                    Data.calm_train)

        ratio_used_for_svm = 0.05
        index_list = [i for i in range(pred_all_.shape[0])]
        num_used_for_svm = int(ratio_used_for_svm * pred_all_.shape[0])
        idxs = random.sample(index_list, num_used_for_svm)
        pilot_cluster_nums = [np.sum(pilot_train_pred_ == i) for i in range(self.n_cluster)]
        hazard_id = np.argmax(pilot_cluster_nums)
        if hazard_id == 0:
            pred_all_ = 1 - pred_all_
        add_train_x = np.vstack((pilot_train_cluster_features_, calm_train_cluster_features_))
        add_train_y = np.hstack((np.zeros(pilot_train_pred_.shape) + 1, np.zeros(calm_train_pred_.shape)))
        train_x = np.vstack((add_train_x, cluster_features_all_[idxs, ::]))
        train_y = np.hstack((add_train_y, pred_all_[idxs]))
        train_x_std = self.Classifier_std.fit_transform(train_x)
        self.Classifier.train(train_x=train_x_std, train_y=train_y)
        hazard_factors_months = []
        for month_i in range(12):  # 2018年缺失6月的数据[0,1,2,3,4,6,7,8,9,10,11]:#
            corridor_month_i = corridor_months[month_i]
            _, _, cluster_features_month_i, pred_month_i = self.predict_in_sess(pilot_semi_,Data, sess, corridor_month_i)
            cluster_features_month_i_std = self.Classifier_std.transform(cluster_features_month_i)
            hazard_factors_month_i = self.Classifier.decision_function(cluster_features_month_i_std)
            hazard_factors_months.append(hazard_factors_month_i)
        return hazard_factors_months
    def transfer_learning(self, Data, pilot_semi_, train_steps, BATCH_SIZE, imbalance_beta_values=np.array([1,10,100,1000]),
              If_cal_AUC = False, step_cal_AUC = None, If_plot_tSNE = False,tsne_num_list=np.arange(0,501,50),
                          If_plot_hazard_value = False, If_plot_ROC = False):
        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=None)
        # '''# pre-train'''
        ae_ckpt_path = os.path.join('ae_ckpt',f'model_encoder_encodedDim{self.encoded_dim}_cluaterDim{self.cluster_dim}.ckpt')
        saver.restore(sess, ae_ckpt_path)

        '''# semi-supervise '''
        sdec_ckpt_path = os.path.join('sdec_ckpt', 'model_supervise.ckpt')
        saver.restore(sess,sdec_ckpt_path)
        transfer_sdec_ckpt_path = os.path.join('sdec_ckpt', 'model_transfer.ckpt')
        metric_values = {'ACC_TRAIN':np.zeros((train_steps,1)),
                        'PTA':np.zeros((train_steps,1)),
                        'ACC_TEST':np.zeros((train_steps,1)),
                        'AUC':np.zeros((train_steps,imbalance_beta_values.size))}



        for i_step in range(train_steps):
            _, _, pilot_train_cluster_features_, pilot_train_pred_ = self.predict_in_sess(pilot_semi_, Data, sess,
                                                                                      pilot_semi_)
            _, _, calm_train_cluster_features_, calm_train_pred_ = self.predict_in_sess(pilot_semi_, Data, sess,
                                                                                        Data.calm_train)
            _, _, cluster_features_all_, pred_all_ = self.predict_in_sess(pilot_semi_, Data, sess,
                                                                          Data.unlabeled_data_transfer)
            _, _, pilot_test_cluster_features_, pilot_test_pred_ = self.predict_in_sess(pilot_semi_, Data, sess,
                                                                                        Data.pilot_test_transfer)
            pilot_cluster_nums = [np.sum(pilot_train_pred_ == i) for i in range(self.n_cluster)]
            hazard_id = np.argmax(pilot_cluster_nums)
            if step_cal_AUC == None:
                if (If_cal_AUC) & (i_step % 100 == 0):
                    AUCs = self.roc_calculate_transfer(sess, hazard_id, Data, imbalance_beta_values, pilot_semi_,i_step,If_plot_hazard_value,If_plot_ROC)
                    metric_values['AUC'][i_step, ::] = AUCs
            if step_cal_AUC != None:
                if (If_cal_AUC) & (i_step in step_cal_AUC):
                    AUCs = self.roc_calculate_transfer(sess, hazard_id, Data, imbalance_beta_values, pilot_semi_,i_step,If_plot_hazard_value,If_plot_ROC)
                    metric_values['AUC'][i_step, ::] = AUCs
            if (If_plot_tSNE) & (i_step in tsne_num_list):
                _, _, cluster_features_all_, pred_all_ = self.predict_in_sess(pilot_semi_, Data, sess,
                                                                              Data.unlabeled_data_transfer)
                _, _, pilot_test_cluster_features_, pilot_test_pred_ = self.predict_in_sess(pilot_semi_, Data, sess,
                                                                                            Data.pilot_test_transfer)
                self.cluster_visualization_tsne_transfer(hazard_id, cluster_features_all_, pilot_train_cluster_features_,
                                                         calm_train_cluster_features_, pilot_test_cluster_features_,
                                                         pred_all_, pilot_train_pred_, calm_train_pred_, pilot_test_pred_,
                                                         i_step)
            index_list = [i for i in range(Data.unlabeled_data_transfer.shape[0])]
            idxs = random.sample(index_list, BATCH_SIZE)
            train_x_batch = Data.unlabeled_data_transfer[idxs]

            encoded_, decoded_, cluster_features_, q_, pred_, T_, dist_ = sess.run([self.encoded, self.decoded,
                                                                             self.cluster_features, self.q,
                                                                             self.pred, self.T,self.dist],
                                                                            feed_dict={self.input: train_x_batch,
                                                                                       self.input_batch_size: BATCH_SIZE,
                                    self.calm_input: Data.calm_train, self.calm_size: Data.calm_train.shape[0],
                                    self.pilot_input: pilot_semi_,
                                    self.pilot_size: pilot_semi_.shape[0]
                                                                                       })
            p_ = self.target_distribution(q_)
            encoded_, decoded_, pred_, q, reconstruct_loss_, _,\
            cluster_loss0, loss0_, calm_loss_, pilot_loss_, \
            max_interval_, pilot_cluster_features_data_, pilot_center_dist_, pilot_center_cluster_features_, B_update_ = \
                sess.run([self.encoded, self.decoded, self.pred, self.q,
                          self.reconstruct_loss,self.optimizer_dec,
                          self.loss_dec, self.loss_supervise, self.calm_loss,
                          self.pilot_loss, self.max_interval, self.pilot_cluster_features_data,
                          self.pilot_center_dist,self.pilot_center_cluster_features,self.B_update],
                         feed_dict={self.input: train_x_batch, self.input_batch_size: BATCH_SIZE,
                                    self.calm_input: Data.calm_train, self.calm_size: Data.calm_train.shape[0],
                                    self.pilot_input: pilot_semi_,
                                    self.pilot_size: pilot_semi_.shape[0],self.p:p_})#
            if i_step % 1 == 0:
                encoded_, decoded_, cluster_features_, pred_= self.predict_in_sess(pilot_semi_,Data, sess,train_x_batch)
                pilot_train_encoded_, pilot_train_decoded_, pilot_train_cluster_features_ , pilot_train_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,pilot_semi_)
                calm_train_encoded_,calm_train_decoded_, calm_train_cluster_features_, calm_train_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,Data.calm_train)
                pilot_cluster_nums = [np.sum(pilot_train_pred_ == i) for i in range(self.n_cluster)]
                hazard_id = np.argmax(pilot_cluster_nums)
                train_acc_ = np.max(pilot_cluster_nums) / pilot_train_pred_.shape[0]
                train_pta_ = (np.sum(pred_ == hazard_id) / pred_.shape[0])

                pilot_test_encoded_, pilot_test_decoded_, test_cluster_features_, pilot_test_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,Data.pilot_test_transfer)
                test_acc = np.sum(pilot_test_pred_ == hazard_id) / pilot_test_pred_.shape[0]
                metric_values['ACC_TRAIN'][i_step, 0] = train_acc_
                metric_values['PTA'][i_step, 0] = train_pta_
                metric_values['ACC_TEST'][i_step, 0] = test_acc
                print(f'{i_step}, total loss = {loss0_}, reconstruct loss = {reconstruct_loss_}, '
                      f'cluster loss = {cluster_loss0}, calm loss = {calm_loss_}, pilot loss = {pilot_loss_}, max interval = {max_interval_}, ')
                print(f'Train ACC: {train_acc_}, PTA: {train_pta_}, Test ACC: {test_acc}')
                # print(B_update_)
        saver.save(sess, transfer_sdec_ckpt_path)
        _, _, cluster_features_all_, pred_all_ = self.predict_in_sess(pilot_semi_,Data,sess,Data.unlabeled_data_transfer)
        sess.close()
        return  metric_values
    def roc_calculate_transfer(self, sess, hazard_id, Data,imbalance_beta_values,pilot_semi_,i_step,If_plot_hazard_value,If_plot_ROC):
        _, _, cluster_features_all_, pred_all_ = self.predict_in_sess(pilot_semi_,Data, sess, Data.unlabeled_data_transfer)
        _, _, pilot_test_cluster_features_, pilot_test_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,Data.pilot_test_transfer)
        _, _, pilot_train_cluster_features_ , pilot_train_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,pilot_semi_)
        _,_, calm_train_cluster_features_, calm_train_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,Data.calm_train)

        ratio_used_for_svm = 1
        index_list = [i for i in range(pred_all_.shape[0])]
        num_used_for_svm = int(ratio_used_for_svm * pred_all_.shape[0])
        idxs = random.sample(index_list, num_used_for_svm)
        if hazard_id == 0:
            pred_all_ = 1 - pred_all_
        add_train_x = np.vstack((pilot_train_cluster_features_, calm_train_cluster_features_))
        add_train_y = np.hstack((np.zeros(pilot_train_pred_.shape) + 1, np.zeros(calm_train_pred_.shape)))
        train_x = np.vstack((add_train_x,cluster_features_all_[idxs,::]))
        train_y = np.hstack((add_train_y,pred_all_[idxs]))
        train_x_std = self.Classifier_std.fit_transform(train_x)
        self.Classifier.train(train_x=train_x_std, train_y=train_y)
        AUCs = np.zeros((1,imbalance_beta_values.size))
        CSIs = np.zeros((1,imbalance_beta_values.size))
        FPR = []
        TPR = []
        Thresholds = []
        for count_imbalance, beta_imbalance in enumerate(imbalance_beta_values):
            calm_test = Data.calm_test_transfer_generator(calm_ratio=beta_imbalance)
            calm_encoded_, calm_decoded_, calm_cluster_features_, calm_pred_ = self.predict_in_sess(pilot_semi_,Data,sess, calm_test)
            test_dataset_x = np.vstack((calm_cluster_features_, pilot_test_cluster_features_))
            test_dataset_y = np.hstack((np.zeros(calm_pred_.shape), np.zeros(pilot_test_pred_.shape) + 1))
            test_dataset_x_std = self.Classifier_std.transform(test_dataset_x)
            test_predict_label = self.Classifier.decision_function(test_dataset_x_std)
            if If_plot_hazard_value&((beta_imbalance==1)|(beta_imbalance==100)):
                self.plot_hazard_factor_value_transfer(test_predict_label, beta_imbalance,i_step)
            fpr, tpr, threshold = roc_curve(test_dataset_y, test_predict_label, drop_intermediate=False,
                                                pos_label=1)
            FPR.append(fpr)
            TPR.append(tpr)
            maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
            Thresholds.append(threshold[maxindex])
            AUC, CSI = calculate_metrics(decision_values=test_predict_label, truth=test_dataset_y, hazard_id = 1)
            print(f'AUC={AUC}')#, CSI = {CSI}
            AUCs[0,count_imbalance] = AUC
            CSIs[0,count_imbalance] = CSI
        if If_plot_ROC:
            self.plot_ROC_transfer(FPR,TPR,AUCs,i_step)
        return AUCs
    def cluster_visualization_tsne_transfer(self, hazard_id, cluster_features_all_, pilot_train_cluster_features_, calm_train_cluster_features_,pilot_test_cluster_features_,
                                  pred_all_, pilot_train_pred_,calm_train_pred_, pilot_test_pred_, i_step):
        plt.figure(figsize=(5, 5))
        pca = TSNE(n_components=2)
        ratio_used_for_tsne = 1
        index_list = [i for i in range(pred_all_.shape[0])]
        num_used_for_tsne = int(ratio_used_for_tsne * pred_all_.shape[0])
        idxs = random.sample(index_list, num_used_for_tsne)
        all_embedded = pca.fit_transform(np.vstack((cluster_features_all_[idxs,::], pilot_train_cluster_features_, 
                                                    calm_train_cluster_features_, pilot_test_cluster_features_)))
        X_unlabel_embedded = all_embedded[:num_used_for_tsne, ::]
        pilot_train_embedded = all_embedded[num_used_for_tsne:-calm_train_pred_.shape[0]-62, ::]
        calm_train_embedded = all_embedded[-calm_train_pred_.shape[0]-62:-62, ::]
        # centers_embedded = all_embedded[-64:-62,::]
        pilot_test_embedded = all_embedded[-62:,::]
        if hazard_id == 0:
            pred_all_ = 1 - pred_all_
            pilot_train_pred_ = 1 - pilot_train_pred_
            calm_train_pred_ = 1 - calm_train_pred_
            pilot_test_pred_ = 1 - pilot_test_pred_
        all_labels = ['All data in cluster 1', 'All data in cluster 2']
        pilot_labels = ['Hazardous wind in cluster 1', 'Hazardous wind in cluster 2']
        calm_labels = ['Calm wind in cluster 1', 'Calm wind in cluster 2']
        plt.scatter(X_unlabel_embedded[pred_all_[idxs] == 0, 0], X_unlabel_embedded[pred_all_[idxs] == 0, 1], marker='.',
                    c=self.all_colors[0], alpha=0.2)  # ,label=all_labels[0]
        plt.scatter(X_unlabel_embedded[pred_all_[idxs] == 1, 0], X_unlabel_embedded[pred_all_[idxs] == 1, 1], marker='.',
                    c=self.all_colors[1], alpha=0.2)  # ,label=all_labels[1]
        plt.scatter(pilot_train_embedded[pilot_train_pred_ == 0, 0], pilot_train_embedded[pilot_train_pred_ == 0, 1],
                    marker='o', c=self.all_colors[0], label=pilot_labels[0], edgecolors='k', linewidth=0.5, s=55)
        plt.scatter(pilot_train_embedded[pilot_train_pred_ == 1, 0], pilot_train_embedded[pilot_train_pred_ == 1, 1],
                    marker='o', c=self.all_colors[1], label=pilot_labels[1], edgecolors='k', linewidth=0.5, s=55)
        plt.scatter(calm_train_embedded[calm_train_pred_ == 0, 0], calm_train_embedded[calm_train_pred_ == 0, 1],
                    marker='s', c=self.all_colors[3], label=calm_labels[0], alpha=1, edgecolors='k', linewidth=0.5)  # 0.2,0.5 s=25
        plt.scatter(calm_train_embedded[calm_train_pred_ == 1, 0], calm_train_embedded[calm_train_pred_ == 1, 1],
                    marker='s', c=self.all_colors[3], label=calm_labels[1], alpha=1, edgecolors='k', linewidth=0.5)#, s=25
        # plt.scatter(centers_embedded[::, 0], centers_embedded[::, 1],
        #             marker='^', c='k', alpha=1, edgecolors='k', linewidth=0.5, s=45)
        # plt.scatter(pilot_test_embedded[pilot_test_pred_ == 0, 0], pilot_test_embedded[pilot_test_pred_ == 0, 1],
        #             marker='*', c=all_colors[3], label=calm_labels[0], alpha=1, edgecolors='k', linewidth=0.5,
        #             s=55)  # 0.2,0.5
        plt.scatter(pilot_test_embedded[::, 0], pilot_test_embedded[::, 1],
                    marker='*', c=self.all_colors[2], label=calm_labels[1], alpha=1, edgecolors='k', linewidth=0.5, s=105)
        plt.title(f'step={i_step}')
        plt.xticks(())
        plt.yticks(())
        plt.savefig(f'../figures/transfer_{i_step}steps.png',dpi=600)  #
        plt.savefig(f'../figures/transfer_{i_step}steps.pdf')
        plt.close()
    def plot_hazard_factor_value_transfer(self,test_predict_label,beta_imbalance,i_step):
        plt.figure()
        id = 1
        plt.subplot(121)
        i = beta_imbalance
        plt.plot(test_predict_label[::id], c="#2E94B9", linewidth=2)
        # plt.hlines(0, 0, 62 * (i + 1) / id, linestyles='--', colors='C7')
        # plt.hlines(0,0,62*(i+1)/id,colors='C7')
        left, bottom, width, height = (0, np.min(test_predict_label)-2, 62 * i / id, np.max(test_predict_label)-np.min(test_predict_label)+4)
        rect = mpatches.Rectangle((left, bottom), width, height,
                                  # fill=False,
                                  alpha=0.05,
                                  facecolor=self.all_colors[0])
        plt.gca().add_patch(rect)
        left, bottom, width, height = (62 * i / id, np.min(test_predict_label)-2, 62 / id, np.max(test_predict_label)-np.min(test_predict_label)+4)
        rect = mpatches.Rectangle((left, bottom), width, height,
                                  # fill=False,
                                  alpha=0.1,
                                  facecolor=self.all_colors[1])
        plt.gca().add_patch(rect)
        plt.xlim((0, 62 * (i + 1) / id))
        plt.ylim((np.min(test_predict_label)-2,np.max(test_predict_label)+2))
        plt.xticks([34 * i / id, (62 * i + 34) / id], ['Calm', 'Hazardous'], family='Arial')
        # plt.yticks([0, 1000], ['0', '1000'])
        plt.subplot(122)
        import pandas as pd
        from scipy.stats import norm
        data = pd.Series(test_predict_label[:-62])  # 将数据由数组转换成series形式
        y_density, x_bins, patches_pilot_dir = plt.hist(data, 6, stacked=True, density=True, facecolor='#2E94B9',
                                                        edgecolor='w', alpha=0.5, orientation='horizontal')  #
        mu = np.mean(data)  # 计算均值
        sigma = np.std(data)  # 计算标准差
        y = norm.pdf(x_bins, mu, sigma)
        plt.plot(y, x_bins, color='#2E94B9', linewidth=2, label='Calm')
        plt.yticks(())
        plt.xticks(())
        data = pd.Series(test_predict_label[-62:])  # 将数据由数组转换成series形式
        y_density, x_bins, patches_pilot_dir = plt.hist(data, 15, stacked=True, density=True, facecolor='#fa625f',
                                                        edgecolor='w', alpha=0.5, orientation='horizontal')
        mu = np.mean(data)  # 计算均值
        sigma = np.std(data)  # 计算标准差
        y = norm.pdf(x_bins, mu, sigma)
        plt.plot(y, x_bins, color='#fa625f', linewidth=4, label='Hazardous')
        plt.yticks(())
        plt.xticks(())
        # plt.hlines(0, 0, 0.028, linestyles='--', colors='C7')
        # plt.xlim((0,0.025))
        if i == 1:
            plt.legend(frameon=False)
        # plt.show(block=True)
        plt.savefig(f'../figures/transfer_{i_step}steps_beta_{beta_imbalance}.png',dpi=600)  #
        plt.savefig(f'../figures/transfer_{i_step}steps_beta_{beta_imbalance}.pdf')
        plt.close()
    def plot_ROC_transfer(self,FPR,TPR,AUCs,i_step):
        plt.figure(figsize=(5, 5))
        Nu = [1, 10, 100, 1000]  # ,5000
        for i in range(4):
            plt.plot(FPR[i], TPR[i], label=fr'$\beta={Nu[i]}$ (auc = {np.round(AUCs[0, i], 2)})',
                     c=self.all_colors[-i - 1], linewidth=2)

        plt.plot([0, 2], [0, 2], linewidth=0.5, color='C7')  # ,label='Random choice'
        plt.hlines(0.709, 0, 2, colors='C7', linestyles='--')
        plt.text(0.3, 0.72, ('WTWS (TPR=0.71)'))  # ,color = all_colors[1]
        # plt.hlines(0.18,0,2,colors=all_colors[2],linestyles='-.')
        # plt.text(0.2,0.20,('AWARE (TPR=0.18)'),color = all_colors[2])
        plt.xlim((-0.01, 1.01))
        plt.ylim((-0.01, 1.01))
        # plt.xlim((-10.01,-1.01))
        # plt.ylim((-10.01,-1.01))
        plt.legend(frameon=False, loc='lower right')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.show(block=True)
        plt.savefig(f'../figures/transfer_{i_step}steps_ROC.png',dpi=600)  #
        plt.savefig(f'../figures/transfer_{i_step}steps_ROC.pdf')
        plt.close()
    def seasonal_statistic_fit_transfer(self,Data,corridor_months,pilot_semi_):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=None)
        # # '''# pre-train'''
        # ae_ckpt_path = os.path.join('ae_ckpt',
        #                             f'model_encoder_encodedDim{self.encoded_dim}_cluaterDim{self.cluster_dim}.ckpt')
        # saver.restore(sess, ae_ckpt_path)

        '''# semi-supervise '''
        transfer_sdec_ckpt_path = os.path.join('sdec_ckpt', 'model_transfer.ckpt')
        saver.restore(sess, transfer_sdec_ckpt_path)
        _, _, cluster_features_all_, pred_all_ = self.predict_in_sess(pilot_semi_, Data, sess, Data.unlabeled_data_transfer)
        _, _, pilot_train_cluster_features_, pilot_train_pred_ = self.predict_in_sess(pilot_semi_, Data, sess,
                                                                                      pilot_semi_)
        _, _, calm_train_cluster_features_, calm_train_pred_ = self.predict_in_sess(pilot_semi_, Data, sess,
                                                                                    Data.calm_train)

        ratio_used_for_svm = 1
        index_list = [i for i in range(pred_all_.shape[0])]
        num_used_for_svm = int(ratio_used_for_svm * pred_all_.shape[0])
        idxs = random.sample(index_list, num_used_for_svm)
        pilot_cluster_nums = [np.sum(pilot_train_pred_ == i) for i in range(self.n_cluster)]
        hazard_id = np.argmax(pilot_cluster_nums)
        if hazard_id == 0:
            pred_all_ = 1 - pred_all_
        add_train_x = np.vstack((pilot_train_cluster_features_, calm_train_cluster_features_))
        add_train_y = np.hstack((np.zeros(pilot_train_pred_.shape) + 1, np.zeros(calm_train_pred_.shape)))
        train_x = np.vstack((add_train_x, cluster_features_all_[idxs, ::]))
        train_y = np.hstack((add_train_y, pred_all_[idxs]))
        train_x_std = self.Classifier_std.fit_transform(train_x)
        self.Classifier.train(train_x=train_x_std, train_y=train_y)
        hazard_factors_months = []

        for month_i in range(len(corridor_months)):  # 2018年缺失6月的数据[0,1,2,3,4,6,7,8,9,10,11]:#
            corridor_month_i = corridor_months[month_i]
            _, _, cluster_features_month_i, pred_month_i = self.predict_in_sess(pilot_semi_,Data, sess, corridor_month_i)
            cluster_features_month_i_std = self.Classifier_std.transform(cluster_features_month_i)
            hazard_factors_month_i = self.Classifier.decision_function(cluster_features_month_i_std)
            hazard_factors_months.append(hazard_factors_month_i)
        # if len(corridor_months)<12:
        #     hazard_factors_months.append(hazard_factors_months[-1])
        return hazard_factors_months