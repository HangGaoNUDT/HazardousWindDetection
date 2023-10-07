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
        # self.B_update = tf.Variable(tf.zeros(shape=(self.n_cluster, self.cluster_dim)), name="B_update")
        # self.dist = self._pairwise_euclidean_distance(self.cluster_features, self.B_update, self.input_batch_size,
        #                                               self.n_cluster)
        # self.pred = tf.argmin(self.dist, 1)
        # self.T = (tf.one_hot(self.pred, self.n_cluster))
        # q = 1.0 / (1.0 + self.dist ** 2)  # REF: dec
        # self.q = (q / tf.reduce_sum(q, axis=1, keepdims=True))
        # self.p = tf.placeholder(tf.float32, shape=(None, self.n_cluster))
        # self.loss_dec = self._kl_divergence(self.p, self.q)

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
                          If_plot_hazard_value = False, If_plot_ROC = False, If_plot_tSNE_encoded = False):
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
            if step_cal_AUC == None:
                if (If_cal_AUC)&(i_step % 100==0):
                    AUCs,CSIs = self.roc_calculate(sess, hazard_id, Data,imbalance_beta_values,pilot_semi_,i_step,
                                                   If_plot_hazard_value,If_plot_ROC,If_plot_hazard_factor_distribution)
                    metric_values['AUC'][i_step, ::] = AUCs
                    metric_values['CSI'][i_step, ::] = CSIs
            if step_cal_AUC != None:
                if (If_cal_AUC)&(i_step in step_cal_AUC):
                    AUCs,CSIs = self.roc_calculate(sess, hazard_id, Data,imbalance_beta_values,pilot_semi_,i_step,
                                                   If_plot_hazard_value,If_plot_ROC,If_plot_hazard_factor_distribution)
                    metric_values['AUC'][i_step, ::] = AUCs
                    metric_values['CSI'][i_step, ::] = CSIs
            if (If_plot_tSNE)&(i_step in tsne_num_list):
                _, _, cluster_features_all_, pred_all_ = self.predict_in_sess(pilot_semi_,Data,sess, Data.unlabeled_data_v_r)
                if If_tsne_3d:
                    self.cluster_visualization_tsne_test_3d(hazard_id, cluster_features_all_, pilot_train_cluster_features_,
                                                   calm_train_cluster_features_,
                                                   pred_all_, pilot_train_pred_, calm_train_pred_, i_step)

                else:
                    self.cluster_visualization_tsne_test(hazard_id, cluster_features_all_, pilot_train_cluster_features_,
                                                   calm_train_cluster_features_,
                                                   pred_all_, pilot_train_pred_, calm_train_pred_, i_step)
                    # self.cluster_visualization_pca_test(cluster_features_all_, pilot_train_cluster_features_,
                    #                                calm_train_cluster_features_,
                    #                                pred_all_, pilot_train_pred_, calm_train_pred_, i_step,B_update_)
            if If_plot_tSNE_encoded&(i_step == 1000):
                encoded_all_, decoded_all_, cluster_features_all_, pred_all_ = self.predict_in_sess(pilot_semi_,Data,sess, Data.unlabeled_data_v_r)
                self.cluster_visualization_tsne_test(hazard_id, encoded_all_, pilot_train_encoded_,
                                                     calm_train_encoded_,
                                                     pred_all_, pilot_train_pred_, calm_train_pred_, i_step)
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
    def roc_calculate(self, sess, hazard_id, Data,imbalance_beta_values,pilot_semi_,i_step,
                      If_plot_hazard_value,If_plot_ROC,If_plot_hazard_factor_distribution):
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
        AUCs = np.zeros((1,imbalance_beta_values.size))
        CSIs = np.zeros((1,imbalance_beta_values.size))
        FPR= []
        TPR = []
        Thresholds = []
        for count_imbalance, beta_imbalance in enumerate(imbalance_beta_values):
            calm_test = Data.calm_test_generator(calm_ratio=beta_imbalance)
            calm_encoded_, calm_decoded_, calm_cluster_features_, calm_pred_ = self.predict_in_sess(pilot_semi_,Data,sess, calm_test)
            test_dataset_x = np.vstack((calm_cluster_features_, pilot_test_cluster_features_))
            test_dataset_y = np.hstack((np.zeros(calm_pred_.shape), np.zeros(pilot_test_pred_.shape) + 1))
            test_dataset_x_std = self.Classifier_std.transform(test_dataset_x)
            test_predict_label = self.Classifier.decision_function(test_dataset_x_std)

            if If_plot_hazard_value&((beta_imbalance==1)|(beta_imbalance==100)):
                self.plot_hazard_factor_value(test_predict_label, beta_imbalance,i_step)

            fpr, tpr, threshold = roc_curve(test_dataset_y, test_predict_label, drop_intermediate=False,
                                            pos_label=1)
            maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
            Thresholds.append(threshold[maxindex])
            FPR.append(fpr)
            TPR.append(tpr)
            AUC, CSI = calculate_metrics(decision_values=test_predict_label, truth=test_dataset_y, hazard_id = 1)
            print(f'AUC={AUC}, CSI = {CSI}')
            AUCs[0,count_imbalance] = AUC
            CSIs[0,count_imbalance] = CSI
        if If_plot_hazard_factor_distribution:
            cluster_features_all_std = self.Classifier_std.transform(cluster_features_all_)
            hazard_factor_values_all = self.Classifier.decision_function(cluster_features_all_std)
            self.plot_violin_intensities(hazard_factor_values_all, Data, test_predict_label,optimal_threshold = Thresholds)
            self.plot_hazard_factor_distribution(hazard_factor_values_all,optimal_threshold = Thresholds)
        if If_plot_ROC:
            self.plot_ROC(FPR,TPR,AUCs,i_step)
        return AUCs,CSIs
    def plot_hazard_factor_distribution(self,all_features,optimal_threshold):
        mu = np.mean(all_features)
        sigma = np.std(all_features)
        all_features = (all_features - mu) / sigma
        optimal_threshold = (optimal_threshold- mu) / sigma
        plt.figure()
        import pandas as pd
        data = pd.Series(all_features)  # 将数据由数组转换成series形式
        n, bins, patches = plt.hist(data, 50, density=True, edgecolor='w',facecolor=self.all_colors[0])
        plt.vlines(optimal_threshold,0,np.max(n))
        data.plot(kind='kde', c="#fa625f", label='Feature distribution', linewidth=2)
        mu = np.mean(all_features)
        sigma = np.std(all_features)
        from scipy.stats import norm
        x = np.arange(-6,6,0.01)
        y = norm.pdf(x, mu, sigma)
        plt.plot(x, y, linestyle='--', color='#ffad60', label='Normal distribution', linewidth=2)

        plt.legend(frameon=False)
        plt.xlabel('Hazard Factor Value')
        plt.xlim((-6, 6))
        # plt.show(block=True)
        plt.savefig(f'../figures/distribution_hazard_factors.png',dpi=600)  #
        plt.savefig(f'../figures/distribution_hazard_factors.pdf')
    def plot_violin_intensities(self,all_features,Data, test_predict_label,optimal_threshold):
        mu = np.mean(all_features)
        sigma = np.std(all_features)
        all_features = (all_features - mu) / sigma
        test_predict_label = (test_predict_label - mu) / sigma
        pilot_turb_mags = Data.pilot_turb_mags_test
        pilot_ws_mags = Data.pilot_ws_mags_test
        pilot_turb_mags[np.where(np.isnan(pilot_turb_mags))] = 0
        pilot_mags = np.vstack((np.abs(pilot_ws_mags.T), np.abs(pilot_turb_mags.T)))
        pilot_mags_unique = np.unique(pilot_mags, axis=1)
        pilot_feature_mean = []
        optimal_threshold = (optimal_threshold- mu) / sigma
        feature = test_predict_label[-280:]
        feature[np.where((feature < np.max(optimal_threshold))|(feature>6))] = np.nan
        # plt.subplot(121)
        pilot_mag_names = []
        ratio_mags = []
        plt.subplot(211)
        for i_unique_mag in range(pilot_mags_unique.shape[1]):
            index_ws_mag = np.where((np.abs(pilot_ws_mags) == pilot_mags_unique[0, i_unique_mag])
                                    & (np.abs(pilot_turb_mags) == pilot_mags_unique[1, i_unique_mag])
                                    & ~np.isnan(feature.reshape(-1, 1)))

            if index_ws_mag[0].shape[0] > 6:
                # plt.plot(feature[index_ws_mag[0]], label=f'ws={pilot_mags_unique[0, i_unique_mag]}, '
                #                                                    f'turb={pilot_mags_unique[1, i_unique_mag]}')
                # plt.hlines(np.nanmean(feature[index_ws_mag[0]]), 0, 280, colors=f'C{i_unique_mag}')
                # plt.legend()
                pilot_feature_mean.append(np.nanmean(feature[index_ws_mag[0]]))
                pilot_mag_names.append(
                    f'ws={pilot_mags_unique[0, i_unique_mag]},turb={pilot_mags_unique[1, i_unique_mag]}')
                plt.violinplot(feature[index_ws_mag[0]], [len(pilot_mag_names) - 1], showmedians=True)  #
                # ratio_mags.append(
                #     np.sum(feature[index_ws_mag[0]] > np.percentile(all_features,90)) / index_ws_mag[0].size)  # correspond to 90th percentilenp.percentile(all_features,90)
                ratio_mags.append(np.median(feature[index_ws_mag[0]]))
                # plt.plot(len(pilot_mag_names)*np.ones_like(feature[index_ws_mag[0]]),feature[index_ws_mag[0]],'o')
            # plt.hlines(1.5,-1,7,colors='C7',linestyles='--',linewidth=0.5)
            plt.xticks(())
            plt.xlim((-0.5, len(pilot_mag_names) - 0.5))
            # plt.yticks(())
            plt.ylabel('Value')
        plt.subplot(212)
        plt.plot(np.arange(len(pilot_mag_names)), ratio_mags, '-', marker='.')
        plt.xticks(np.arange(len(pilot_mag_names)), (pilot_mag_names), rotation=20)
        plt.ylabel('Median Value')
        plt.xlim((-0.5, len(pilot_mag_names) - 0.5))
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(f'../figures/violin_hazard_factors.png',dpi=600)  #
        plt.savefig(f'../figures/violin_hazard_factors.pdf')
    def cluster_visualization_pca_test(self, hazard_id, cluster_features_all_, pilot_train_cluster_features_, calm_train_cluster_features_,
                                  pred_all_, pilot_train_pred_,calm_train_pred_, i_step, centers):
        plt.figure(figsize=(5, 5))
        pca = PCA(n_components=2)
        ratio_used_for_tsne = 0.05
        index_list = [i for i in range(pred_all_.shape[0])]
        num_used_for_tsne = int(ratio_used_for_tsne * pred_all_.shape[0])
        idxs = random.sample(index_list, num_used_for_tsne)
        all_embedded = pca.fit_transform(np.vstack((cluster_features_all_[idxs,::], pilot_train_cluster_features_, calm_train_cluster_features_)))
        X_unlabel_embedded = all_embedded[:num_used_for_tsne, ::]
        pilot_train_embedded = all_embedded[num_used_for_tsne:-calm_train_pred_.shape[0], ::]
        calm_train_embedded = all_embedded[-calm_train_pred_.shape[0]:, ::]
        centers_embedded = pca.transform(centers)
        all_colors = ['#2E94B9', "#fa625f", '#62C8A5', '#f29c2b']
        all_labels = ['All data in cluster 1', 'All data in cluster 2']
        pilot_labels = ['Hazardous wind in cluster 1', 'Hazardous wind in cluster 2']
        calm_labels = ['Calm wind in cluster 1', 'Calm wind in cluster 2']
        plt.scatter(X_unlabel_embedded[pred_all_[idxs] == 0, 0], X_unlabel_embedded[pred_all_[idxs] == 0, 1], marker='.',
                    c=all_colors[0], alpha=0.04)  # ,label=all_labels[0]
        plt.scatter(X_unlabel_embedded[pred_all_[idxs] == 1, 0], X_unlabel_embedded[pred_all_[idxs] == 1, 1], marker='.',
                    c=all_colors[1], alpha=0.04)  # ,label=all_labels[1]
        plt.scatter(pilot_train_embedded[pilot_train_pred_ == 0, 0], pilot_train_embedded[pilot_train_pred_ == 0, 1],
                    marker='o', c=all_colors[0], label=pilot_labels[0], edgecolors='k', linewidth=0.5, s=25)
        plt.scatter(pilot_train_embedded[pilot_train_pred_ == 1, 0], pilot_train_embedded[pilot_train_pred_ == 1, 1],
                    marker='o', c=all_colors[1], label=pilot_labels[1], edgecolors='k', linewidth=0.5, s=25)
        plt.scatter(calm_train_embedded[calm_train_pred_ == 0, 0], calm_train_embedded[calm_train_pred_ == 0, 1],
                    marker='s', c=all_colors[3], label=calm_labels[0], alpha=1, edgecolors='k', linewidth=0.5,
                    s=25)  # 0.2,0.5
        plt.scatter(calm_train_embedded[calm_train_pred_ == 1, 0], calm_train_embedded[calm_train_pred_ == 1, 1],
                    marker='s', c=all_colors[2], label=calm_labels[1], alpha=1, edgecolors='k', linewidth=0.5, s=25)
        plt.scatter(centers_embedded[::, 0], centers_embedded[::, 1],
                    marker='s', c='k', alpha=1, edgecolors='k', linewidth=0.5, s=45)

        plt.title(f'step={i_step}')
        # plt.xticks(())
        # plt.yticks(())
        plt.savefig(f'{i_step}steps.png',dpi=600)  #
        # plt.savefig(f'{i_step}steps.pdf')

    def cluster_visualization_tsne_test(self, hazard_id, cluster_features_all_, pilot_train_cluster_features_, calm_train_cluster_features_,
                                  pred_all_, pilot_train_pred_,calm_train_pred_, i_step):
        plt.figure(figsize=(5, 5))
        pca = TSNE(n_components=2)
        ratio_used_for_tsne = 0.05
        index_list = [i for i in range(pred_all_.shape[0])]
        num_used_for_tsne = int(ratio_used_for_tsne * pred_all_.shape[0])
        idxs = random.sample(index_list, num_used_for_tsne)
        all_embedded = pca.fit_transform(np.vstack((cluster_features_all_[idxs,::], pilot_train_cluster_features_, calm_train_cluster_features_)))
        X_unlabel_embedded = all_embedded[:num_used_for_tsne, ::]
        pilot_train_embedded = all_embedded[num_used_for_tsne:-calm_train_pred_.shape[0], ::]
        calm_train_embedded = all_embedded[-calm_train_pred_.shape[0]:, ::]
        # centers_embedded = all_embedded[-2:,::]

        if hazard_id == 0:
            pred_all_ = 1 - pred_all_
            pilot_train_pred_ = 1 - pilot_train_pred_
            calm_train_pred_ = 1 - calm_train_pred_
        all_labels = ['All data in cluster 1', 'All data in cluster 2']
        pilot_labels = ['Hazardous wind in cluster 1', 'Hazardous wind in cluster 2']
        calm_labels = ['Calm wind in cluster 1', 'Calm wind in cluster 2']
        plt.scatter(X_unlabel_embedded[pred_all_[idxs] == 0, 0], X_unlabel_embedded[pred_all_[idxs] == 0, 1], marker='.',
                    c=self.all_colors[0], alpha=0.04)  # ,label=all_labels[0]
        plt.scatter(X_unlabel_embedded[pred_all_[idxs] == 1, 0], X_unlabel_embedded[pred_all_[idxs] == 1, 1], marker='.',
                    c=self.all_colors[1], alpha=0.04)  # ,label=all_labels[1]
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
        # plt.scatter(centers_embedded[::, 0], centers_embedded[::, 1],
        #             marker='^', c='k', alpha=1, edgecolors='k', linewidth=0.5, s=45)
        plt.title(f'step={i_step}')
        plt.xticks(())
        plt.yticks(())
        plt.savefig(f'../figures/{i_step}steps_encoded_feature.png',dpi=600)  #
        plt.savefig(f'../figures/{i_step}steps_encoded_feature.pdf')


    def cluster_visualization_tsne_test_3d(self, hazard_id, cluster_features_all_, pilot_train_cluster_features_, calm_train_cluster_features_,
                                  pred_all_, pilot_train_pred_,calm_train_pred_, i_step):
        plt.figure(figsize=(5, 5))
        ax = plt.axes(projection='3d')
        pca = TSNE(n_components=3)
        ratio_used_for_tsne = 0.05
        index_list = [i for i in range(pred_all_.shape[0])]
        num_used_for_tsne = int(ratio_used_for_tsne * pred_all_.shape[0])
        idxs = random.sample(index_list, num_used_for_tsne)
        all_embedded = pca.fit_transform(np.vstack((cluster_features_all_[idxs,::], pilot_train_cluster_features_, calm_train_cluster_features_)))
        X_unlabel_embedded = all_embedded[:num_used_for_tsne, ...]
        pilot_train_embedded = all_embedded[num_used_for_tsne:-calm_train_pred_.shape[0], ...]
        calm_train_embedded = all_embedded[-calm_train_pred_.shape[0]:, ...]
        all_colors = ['#2E94B9', "#fa625f", '#62C8A5', '#f29c2b']

        if hazard_id == 0:
            pred_all_ = 1 - pred_all_
            pilot_train_pred_ = 1 - pilot_train_pred_
            calm_train_pred_ = 1 - calm_train_pred_
        all_labels = ['All data in cluster 1', 'All data in cluster 2']
        pilot_labels = ['Hazardous wind in cluster 1', 'Hazardous wind in cluster 2']
        calm_labels = ['Calm wind in cluster 1', 'Calm wind in cluster 2']
        ax.scatter(X_unlabel_embedded[pred_all_[idxs] == 0, 0], X_unlabel_embedded[pred_all_[idxs] == 0, 1], X_unlabel_embedded[pred_all_[idxs] == 0, 2], marker='.',
                   c=all_colors[0], alpha=0.04)  # ,label=all_labels[0]
        ax.scatter(X_unlabel_embedded[pred_all_[idxs] == 1, 0], X_unlabel_embedded[pred_all_[idxs] == 1, 1], X_unlabel_embedded[pred_all_[idxs] == 1, 2], marker='.',
                   c=all_colors[1], alpha=0.04)  # ,label=all_labels[1]
        ax.scatter(pilot_train_embedded[pilot_train_pred_ == 0, 0], pilot_train_embedded[pilot_train_pred_ == 0, 1], pilot_train_embedded[pilot_train_pred_ == 0, 2],
                   marker='o', c=all_colors[0], label=pilot_labels[0], edgecolors='k', linewidth=0.5, s=25)
        ax.scatter(pilot_train_embedded[pilot_train_pred_ == 1, 0], pilot_train_embedded[pilot_train_pred_ == 1, 1], pilot_train_embedded[pilot_train_pred_ == 1, 2],
                   marker='o', c=all_colors[1], label=pilot_labels[1], edgecolors='k', linewidth=0.5, s=25)
        ax.scatter(calm_train_embedded[calm_train_pred_ == 0, 0], calm_train_embedded[calm_train_pred_ == 0, 1], calm_train_embedded[calm_train_pred_ == 0, 2],
                   marker='s', c=all_colors[3], label=calm_labels[0], alpha=1, edgecolors='k', linewidth=0.5,
                   s=25)  # 0.2,0.5
        ax.scatter(calm_train_embedded[calm_train_pred_ == 1, 0], calm_train_embedded[calm_train_pred_ == 1, 1], calm_train_embedded[calm_train_pred_ == 1, 2],
                    marker='s', c=all_colors[2], label=calm_labels[1], alpha=1, edgecolors='k', linewidth=0.5, s=25)
        ax.set_title(f'step={i_step}')
        plt.xticks(())
        plt.yticks(())
        ax.set_zticks(())
        # plt.show(block=True)
        plt.savefig(f'{i_step}steps.png',dpi=600)  #
        plt.savefig(f'{i_step}steps.pdf')

    def plot_hazard_factor_value(self,test_predict_label,beta_imbalance,i_step):
        plt.figure()
        id = 1
        plt.subplot(121)
        i = beta_imbalance
        plt.plot(test_predict_label[::id], c="#2E94B9", linewidth=2)
        # plt.hlines(0, 0, 62 * (i + 1) / id, linestyles='--', colors='C7')
        # plt.hlines(0,0,62*(i+1)/id,colors='C7')
        left, bottom, width, height = (0, np.min(test_predict_label)-2, 280 * i / id, np.max(test_predict_label)-np.min(test_predict_label)+4)
        rect = mpatches.Rectangle((left, bottom), width, height,
                                  # fill=False,
                                  alpha=0.05,
                                  facecolor=self.all_colors[0])
        plt.gca().add_patch(rect)
        left, bottom, width, height = (280 * i / id, np.min(test_predict_label)-2, 280 / id, np.max(test_predict_label)-np.min(test_predict_label)+4)
        rect = mpatches.Rectangle((left, bottom), width, height,
                                  # fill=False,
                                  alpha=0.1,
                                  facecolor=self.all_colors[1])
        plt.gca().add_patch(rect)
        plt.xlim((0, 280 * (i + 1) / id))
        plt.ylim((np.min(test_predict_label)-2,np.max(test_predict_label)+2))
        plt.xticks([140 * i / id, (280 * i + 140) / id], ['Calm', 'Hazardous'], family='Arial')
        # plt.yticks([0, 1000], ['0', '1000'])
        plt.subplot(122)
        import pandas as pd
        from scipy.stats import norm
        data = pd.Series(test_predict_label[:-280])  # 将数据由数组转换成series形式
        y_density, x_bins, patches_pilot_dir = plt.hist(data, 6, stacked=True, density=True, facecolor='#2E94B9',
                                                        edgecolor='w', alpha=0.5, orientation='horizontal')  #
        mu = np.mean(data)  # 计算均值
        sigma = np.std(data)  # 计算标准差
        y = norm.pdf(x_bins, mu, sigma)
        plt.plot(y, x_bins, color='#2E94B9', linewidth=1.5, label='Calm')
        plt.yticks(())
        plt.xticks(())
        data = pd.Series(test_predict_label[-280:])  # 将数据由数组转换成series形式
        y_density, x_bins, patches_pilot_dir = plt.hist(data, 15, stacked=True, density=True, facecolor='#fa625f',
                                                        edgecolor='w', alpha=0.5, orientation='horizontal')
        mu = np.mean(data)  # 计算均值
        sigma = np.std(data)  # 计算标准差
        y = norm.pdf(x_bins, mu, sigma)
        plt.plot(y, x_bins, color='#fa625f', linewidth=1.5, label='Hazardous')
        plt.yticks(())
        plt.xticks(())
        # plt.hlines(0, 0, 0.028, linestyles='--', colors='C7')
        # plt.xlim((0,0.025))
        if i == 1:
            plt.legend(frameon=False)
        # plt.show(block=True)
        plt.savefig(f'../figures/{i_step}steps_beta_{beta_imbalance}.png',dpi=600)  #
        plt.savefig(f'../figures/{i_step}steps_beta_{beta_imbalance}.pdf')
        plt.close()
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
        all_embedded = pca.fit_transform(np.vstack((cluster_features_all_[idxs,::], pilot_train_cluster_features_, calm_train_cluster_features_, pilot_test_cluster_features_)))
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