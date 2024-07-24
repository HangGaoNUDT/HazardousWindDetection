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

turb_ratio = 8
def pilot_intensity_classification(Data):
    pilot_turb_mags = Data.pilot_turb_mag_train
    pilot_ws_mags = Data.pilot_ws_mag_train
    pilot_turb_mags[np.where(np.isnan(pilot_turb_mags))] = 0
    pilot_ws_mags[np.where((np.abs(pilot_ws_mags)==15.1))] = 15
    # pilot_turb_mags[np.where((np.abs(pilot_turb_mags)==2.5))] = 2
    # pilot_mags = np.vstack((np.abs(pilot_ws_mags.T), np.abs(pilot_turb_mags.T)))
    # pilot_mags_unique = np.unique(pilot_mags, axis=1)
    pilot_mags = np.abs(pilot_ws_mags)+np.abs(pilot_turb_mags)*turb_ratio
    idxs = np.argsort(pilot_mags,axis=0)
    
    return np.squeeze(Data.pilot_train_v_r[idxs[:90],::]),np.squeeze(Data.pilot_train_v_r[idxs[90:],::])
def turb_select(Data):
    pilot_turb_mags = Data.pilot_turb_mags_test
    idxs = np.where(pilot_turb_mags>0)[0]
    turb_v_r = np.squeeze(Data.pilot_train_v_r[idxs,::])
    calm_turb = Data.calm_test_generator(calm_ratio_index=0,standard = False)
    all_indexes = [i for i in range(calm_turb.shape[0])]
    calm_idxs = random.sample(all_indexes,turb_v_r.shape[0])
    calm_turb = calm_turb[calm_idxs,::]
    return turb_v_r,calm_turb
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
    def __init__(self, encoded_dim=4,cluster_dim=4, a = 1, b=35, c = 3, d=31, interval = 150):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = c+d
        self.interval = interval
        self.n_cluster = 2
        self.encoded_dim = encoded_dim
        self.cluster_dim = cluster_dim
        self.kmeans = KMeans(n_clusters=self.n_cluster, n_init=20)
        self.Classifier = svm_classifier()
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
        self.cluster_features_data = (self.encoded)
        self.cluster_features = self.cluster_features_data
        self.decoded_encoded, _ = self.ae_setup(self.decoded)
        self.reconstruct_loss = tf.reduce_mean((self.input- self.decoded)**2)

        # introduce pilot reports as known information
        self.pilot_input = tf.placeholder(tf.float32, shape=[None, 115])
        self.pilot_size = tf.placeholder(tf.int32, shape=())
        self.pilot_encoded, self.pilot_decoded = self.ae_setup(self.pilot_input)
        self.pilot_cluster_features_data = (self.pilot_encoded)
        self.pilot_center_cluster_features = tf.reduce_mean(self.pilot_cluster_features_data,axis = 0,keepdims=True)
        self.pilot_center_dist = (self.pilot_cluster_features_data - self.pilot_center_cluster_features) ** 2
        self.pilot_loss = (tf.reduce_mean(tf.reduce_sum(self.pilot_center_dist,axis=1)))  # 使飞行员分类在第一类

        # introduce calm winds as known information
        self.calm_input = tf.placeholder(tf.float32, shape=[None, 115])
        self.calm_size = tf.placeholder(tf.int32, shape=())
        self.calm_encoded, self.calm_decoded = self.ae_setup(self.calm_input)
        self.calm_cluster_features_data = (self.calm_encoded)
        self.calm_center_cluster_features = tf.reduce_mean(self.calm_cluster_features_data,axis = 0,keepdims=True) 
        self.all_center_cluster_features = tf.reduce_mean(self.cluster_features,axis = 0,keepdims=True) 

        self.calm_center_dist = (self.calm_cluster_features_data - self.calm_center_cluster_features) ** 2
        self.calm_loss = ((tf.reduce_mean(tf.reduce_sum(self.calm_center_dist,axis = 1))))

        self.B_update = tf.concat([self.pilot_center_cluster_features, self.calm_center_cluster_features], axis=0)
        self.dist = self._pairwise_euclidean_distance(self.cluster_features, self.B_update, self.input_batch_size,
                                                      self.n_cluster)
        self.pred = tf.argmin(self.dist, 1)
 
        # remove the hazardous winds from all the observations result in unlabel data
        self.unlabel_input = tf.placeholder(tf.float32, shape=[None, 115])
        self.unlabel_size = tf.placeholder(tf.int32, shape=())
        self.unlabel_encoded, self.unlabel_decoded = self.ae_setup(self.unlabel_input)
        self.unlabel_cluster_features_data = (self.unlabel_encoded)
        self.all_to_calm_dist = (self.unlabel_cluster_features_data-self.calm_center_cluster_features)**2
        self.loss_dec = ((tf.reduce_mean(tf.reduce_sum(self.all_to_calm_dist, axis = 1))))

        self.pilot_weak = tf.placeholder(tf.float32, shape=[None, 115])
        self.pilot_size_weak = tf.placeholder(tf.int32, shape=())
        pilot_encoded_weak,_  = self.ae_setup(self.pilot_weak)
        self.pilot_strong = tf.placeholder(tf.float32, shape=[None, 115])
        self.pilot_size_strong = tf.placeholder(tf.int32, shape=())
        pilot_encoded_strong,_  = self.ae_setup(self.pilot_strong)
        
        self.calm_pilot_interval_weak =  tf.sqrt(tf.reduce_sum((pilot_encoded_weak-self.calm_center_cluster_features)**2,axis = 1))
        self.calm_pilot_interval_strong =  tf.sqrt(tf.reduce_sum((pilot_encoded_strong-self.calm_center_cluster_features)**2,axis = 1))

        self.calm_pilot_dist_weak =  (self.calm_pilot_interval_weak -self.interval)**2-self.e*tf.log(1e-4+tf.nn.sigmoid(-self.calm_pilot_interval_weak +self.interval))
        self.calm_pilot_dist_strong =  (self.calm_pilot_interval_strong-self.interval)**2-self.e*tf.log(1e-4+tf.nn.sigmoid(self.calm_pilot_interval_strong-self.interval))
        calm_pilot_dist = tf.reduce_mean(self.calm_pilot_dist_weak)+tf.reduce_mean(self.calm_pilot_dist_strong)

        self.max_interval = calm_pilot_dist
        self.reconstruct_loss_2 = tf.reduce_mean(tf.reduce_sum((self.encoded- self.decoded_encoded)**2, axis = 1))
        self.loss_supervise = (1/self.a)* self.reconstruct_loss_2 + (1/self.b)* self.loss_dec+\
             (1/self.c)*self.calm_loss + (1/self.d)* self.pilot_loss+ (1/self.e) * self.max_interval
        self.optimizer_dec = tf.train.AdamOptimizer(0.0001).minimize(self.loss_supervise)
        self.loss_pretrain = self.reconstruct_loss  
        self.optimizer_pretrain = tf.train.AdamOptimizer(0.001).minimize(self.loss_pretrain)


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


    def train(self, Data, pilot_semi_, train_steps, BATCH_SIZE, 
              If_cal_AUC = False, 
              If_plot_intensity = False, 
            If_other_airports = False, other_airports_v = None, If_pred_nn = False, If_pred_wrf = False):
        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=None)
        '''# pre-train'''
        ae_ckpt_path = os.path.join('ae_ckpt',f'model_encoder_encodedDim{self.encoded_dim}_cluaterDim{self.cluster_dim}_unlabel.ckpt')
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
                        'AUC':np.zeros((train_steps,1)),
                        'CSI':np.zeros((train_steps,1))}


        pilot_weak, pilot_strong = pilot_intensity_classification(Data)
        for i_step in range(train_steps):
            pilot_train_encoded_, _, pilot_train_cluster_features_ , pilot_train_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,pilot_semi_)
            calm_train_encoded_, _, calm_train_cluster_features_, calm_train_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,Data.calm_train)
            pilot_cluster_nums = [np.sum(pilot_train_pred_ == i) for i in range(self.n_cluster)]
            hazard_id = np.argmax(pilot_cluster_nums)
            if (If_cal_AUC)&(i_step == 2000):
                AUCs,CSIs,optimal_threshold = self.roc_calculate(sess, hazard_id, Data,pilot_semi_)
                metric_values['AUC'][i_step, 0] = AUCs
                metric_values['CSI'][i_step, 0] = CSIs
            if (If_plot_intensity)&(i_step == 2000):
                _, _, cluster_features_all_, pred_all_ = self.predict_in_sess(pilot_semi_,Data,sess, Data.unlabeled_data_v_r_calm)
                _, _, pilot_train_cluster_features_ , pilot_train_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,pilot_semi_)
                _, _, calm_train_cluster_features_, calm_train_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,Data.calm_train)
                all_hazard_factor = self.hazard_factor_calculate(cluster_features_all_)
                pilot_train_hazard_factor = self.hazard_factor_calculate(pilot_train_cluster_features_)
                mu = np.mean(all_hazard_factor)
                sigma = np.std(all_hazard_factor)
                all_hazard_factor_norm = (all_hazard_factor - mu) / sigma
                pilot_train_hazard_factor_norm = (pilot_train_hazard_factor - mu) / sigma
                _, _, pilot_test_cluster_features_ , pilot_test_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,Data.pilot_test_v_r)
                pilot_test_hazard_factor = self.hazard_factor_calculate(pilot_test_cluster_features_)
                pilot_test_hazard_factor_norm = (pilot_test_hazard_factor - mu) / sigma
                self.plot_fenji_violin(Data,pilot_train_hazard_factor_norm,pilot_test_hazard_factor_norm)

            if If_other_airports&(i_step == 2000):
                _, _, cluster_features_all_, pred_all_ = self.predict_in_sess(pilot_semi_,Data,sess, Data.unlabeled_data_v_r_calm)
                _, _, pilot_train_cluster_features_ , pilot_train_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,pilot_semi_)
                _, _, calm_train_cluster_features_, calm_train_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,Data.calm_train)
                _, _, other_cluster_features_all_, other_pred_all_ = self.predict_in_sess(pilot_semi_,Data,sess, other_airports_v)
                all_hazard_factor = self.hazard_factor_calculate(cluster_features_all_)
                other_airports_hazard_factor = self.hazard_factor_calculate(other_cluster_features_all_)
                _, _, pilot_test_cluster_features_, pilot_test_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,Data.pilot_test_v_r)
                pilot_test_hazard_factor = self.hazard_factor_calculate(pilot_test_cluster_features_)
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
            if If_pred_nn&(i_step == 2000):
                calm_pred_times = scio.loadmat('calm_pred_times.mat')
                calm_pred = calm_pred_times['pred_times']
                calm_true = calm_pred_times['true_times']
                pilot_pred_times = scio.loadmat('pilot_pred_times.mat')
                intervals = pilot_pred_times['pred_interals']
                pilot_pred = pilot_pred_times['pred_times']
                pilot_true = pilot_pred_times['true_times']
                corr_ = np.zeros((len(intervals[0]),1))
                corr_cos_ = np.zeros((len(intervals[0]),1))
                AUCs_pred = np.zeros((len(intervals[0]),1))
                AUCs_truth = np.zeros((len(intervals[0]),1))
               
                for interval_i in intervals[0]:
                    calm_pred_i = calm_pred[0][interval_i-1]
                    calm_true_i = calm_true[0][interval_i-1]
                    pilot_pred_i = pilot_pred[0][interval_i-1]
                    pilot_true_i = pilot_true[0][interval_i-1]
                    if (pilot_true_i.shape[0]>5)&(calm_true_i.shape[0]>5):

                        calm_test = calm_pred_i
                        pilot_test = pilot_pred_i
                        _, _, calm_test_cluster_features_, calm_test_pred_ = self.predict_in_sess(pilot_semi_,Data,sess, calm_test)
                        _, _, pilot_test_cluster_features_, pilot_test_pred_ = self.predict_in_sess(pilot_semi_,Data,sess, pilot_test)
                        calm_test_hazard_factor_ = self.hazard_factor_calculate(calm_test_cluster_features_)
                        pilot_test_hazard_factor_ = self.hazard_factor_calculate(pilot_test_cluster_features_)
                        AUC,CSI = self.roc_calculate_pred( calm_test_hazard_factor_,pilot_test_hazard_factor_,optimal_threshold)
                        AUCs_pred[interval_i-1,0] = AUC
                        _, _, cluster_features_all_, pred_all_ = self.predict_in_sess(pilot_semi_,Data,sess, Data.unlabeled_data_v_r)
                        all_hazard_factor = self.hazard_factor_calculate(cluster_features_all_)
                        mu = np.mean(all_hazard_factor)
                        sigma = np.std(all_hazard_factor)
                        all_hazard_factor_norm = (all_hazard_factor - mu) / sigma
                        pilot_test_hazard_factor_norm = (pilot_test_hazard_factor_ - mu) / sigma
                        # thre_norm = (thre_ - mu) / sigma
                        pred_hazard_factor_ = pilot_test_hazard_factor_norm
                        if (interval_i == 15)|(interval_i == 45):
                            self.cluster_visualization_tsne_test_pred(cluster_features_all_, pilot_test_cluster_features_, 
                                                    all_hazard_factor_norm,pilot_test_hazard_factor_norm, pred_time = interval_i)

                        calm_test = calm_true_i
                        pilot_test = pilot_true_i
                        _, _, calm_test_cluster_features_, calm_test_pred_ = self.predict_in_sess(pilot_semi_,Data,sess, calm_test)
                        _, _, pilot_test_cluster_features_, pilot_test_pred_ = self.predict_in_sess(pilot_semi_,Data,sess, pilot_test)
                        calm_test_hazard_factor_ = self.hazard_factor_calculate(calm_test_cluster_features_)
                        pilot_test_hazard_factor_ = self.hazard_factor_calculate(pilot_test_cluster_features_)
                        pilot_test_hazard_factor_norm = (pilot_test_hazard_factor_ - mu) / sigma
                        true_hazard_factor_ = pilot_test_hazard_factor_norm
                        from scipy.stats import pearsonr
                        AUC,CSI = self.roc_calculate_pred(calm_test_hazard_factor_,pilot_test_hazard_factor_,optimal_threshold)

                        AUCs_truth[interval_i-1,0] = AUC

                        r_cos_ = cosine_similarity(pred_hazard_factor_.reshape(1, -1),true_hazard_factor_.reshape(1, -1))
                        r_,_ = pearsonr(pred_hazard_factor_,true_hazard_factor_)
                        corr_[interval_i-1,0] = r_
                        corr_cos_[interval_i-1,0] = r_cos_
                        if (interval_i == 15):
                            scio.savemat('sfig8d.mat',{'pred_hazard_factor_':pred_hazard_factor_,
                                                       'true_hazard_factor_':true_hazard_factor_})
                        if (interval_i == 45):
                            scio.savemat('sfig8e.mat',{'pred_hazard_factor_':pred_hazard_factor_,
                                                       'true_hazard_factor_':true_hazard_factor_})
            # scio.savemat('sfig8cf.mat',{'corr_cos_':corr_cos_,'AUCs_pred':AUCs_pred})
            
            if If_pred_wrf&(i_step == 2000):
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
                pilots_headwinds = (pilots_headwinds.squeeze())
                calm_headwinds = (calm_headwinds.squeeze())
                pilot_test = pilots_headwinds
                calm_test = calm_headwinds
                _, _, calm_test_cluster_features_, calm_test_pred_ = self.predict_in_sess(pilot_semi_,Data,sess, calm_test)
                _, _, pilot_test_cluster_features_, pilot_test_pred_ = self.predict_in_sess(pilot_semi_,Data,sess, pilot_test)
                calm_test_hazard_factor_ = self.hazard_factor_calculate(calm_test_cluster_features_)
                pilot_test_hazard_factor_ = self.hazard_factor_calculate(pilot_test_cluster_features_)
                _, _, cluster_features_all_, pred_all_ = self.predict_in_sess(pilot_semi_,Data,sess, Data.unlabeled_data_v_r)
                all_hazard_factor = self.hazard_factor_calculate(cluster_features_all_)
                mu = np.mean(all_hazard_factor)
                sigma = np.std(all_hazard_factor)
                all_hazard_factor_norm = (all_hazard_factor - mu) / sigma
                pilot_test_hazard_factor_norm = (pilot_test_hazard_factor_ - mu) / sigma
                calm_test_hazard_factor_norm = (calm_test_hazard_factor_ - mu) / sigma
                AUCs,CSIs = self.roc_calculate_pred(calm_test_hazard_factor_norm,pilot_test_hazard_factor_,optimal_threshold)
                
                import matplotlib.gridspec as gridspec
                fig = plt.figure()
                plt.title('WRF data (Supplementary Fig6 ac)')
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
                # scio.savemat('sfig6ac_data.mat',{'pilots_headwinds':pilots_headwinds,
                #                             'pilot_test_hazard_factor_norm':pilot_test_hazard_factor_norm})
                plt.savefig(f'../figures/Pred_wrf.png',dpi=600)  #
                plt.savefig(f'../figures/Pred_wrf.pdf')
                self.cluster_visualization_tsne_test_pred_wrf(cluster_features_all_, pilot_test_cluster_features_, 
                                                    all_hazard_factor,pilot_test_hazard_factor_norm,pilots_time)
                

            index_list = [i for i in range(Data.DATA_NUM)]
            idxs = random.sample(index_list, BATCH_SIZE)
            train_x_batch = Data.unlabeled_data_v_r[idxs]
            index_list_calm = [i for i in range(Data.DATA_NUM_calm)]
            idxs_calm = random.sample(index_list_calm, BATCH_SIZE)
            train_x_batch_calm = Data.unlabeled_data_v_r_calm[idxs_calm]
            encoded_, decoded_, pred_,  reconstruct_loss_, _,\
            cluster_loss0, loss0_, calm_loss_, pilot_loss_, \
            max_interval_, calm_pilot_dist_strong_, calm_pilot_dist_weak_,\
                 calm_pilot_interval_weak_, calm_pilot_interval_strong_  = \
                sess.run([self.encoded, self.decoded, self.pred, 
                          self.reconstruct_loss,self.optimizer_dec,
                          self.loss_dec, self.loss_supervise, self.calm_loss,
                          self.pilot_loss, self.max_interval, self.calm_pilot_dist_strong,
                          self.calm_pilot_dist_weak,self.calm_pilot_interval_weak,self.calm_pilot_interval_strong],
                         feed_dict={self.input: train_x_batch, self.input_batch_size: BATCH_SIZE,
                                    self.calm_input: Data.calm_train, self.calm_size: Data.calm_train.shape[0],
                                    self.pilot_input: pilot_semi_,
                                    self.pilot_size: pilot_semi_.shape[0],
                                    self.unlabel_input:train_x_batch_calm, self.unlabel_size:BATCH_SIZE,
                                    self.pilot_weak:pilot_weak,self.pilot_strong:pilot_strong,
                                    self.pilot_size_weak:pilot_weak.shape[0],self.pilot_size_strong:pilot_strong.shape[0]})#
            if i_step % 10 == 0:
                encoded_, decoded_, cluster_features_, pred_= self.predict_in_sess(pilot_semi_,Data, sess,train_x_batch_calm)
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
                # print(f'{i_step}, total loss = {loss0_}, reconstruct loss = {reconstruct_loss_}, '
                #       f'cluster loss = {cluster_loss0}, calm loss = {calm_loss_}, pilot loss = {pilot_loss_}, max interval = {max_interval_}, ')
                # print(f'Train ACC: {train_acc_}, PTA: {train_pta_}, Test ACC: {test_acc}')
        saver.save(sess, sdec_ckpt_path)
        _, _, cluster_features_all_, pred_all_ = self.predict_in_sess(pilot_semi_,Data,sess,Data.unlabeled_data_v_r)
        sess.close()
        return  metric_values
    def hazard_factor_calculate(self,other_cluster_features_all_):
        other_airports_x_std = other_cluster_features_all_#self.Classifier_std.transform(other_cluster_features_all_)
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
    def roc_calculate(self, sess, hazard_id, Data,pilot_semi_):
        
        self.Classifier.__init__()
        encoded_all, _, cluster_features_all_, pred_all_ = self.predict_in_sess(pilot_semi_,Data, sess, Data.unlabeled_data_v_r_calm)
        encoded_test, _, pilot_test_cluster_features_, pilot_test_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,Data.pilot_test_v_r)
        encoded_train, _, pilot_train_cluster_features_ , pilot_train_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,pilot_semi_)
        encoded_calm,_, calm_train_cluster_features_, calm_train_pred_ = self.predict_in_sess(pilot_semi_,Data, sess,Data.calm_train_svm)
        if hazard_id == 0:
            pred_all_ = 1 - pred_all_
        add_train_x = np.vstack((pilot_train_cluster_features_, calm_train_cluster_features_))
        add_train_y = np.hstack((np.zeros(pilot_train_pred_.shape) + 1, np.zeros(calm_train_pred_.shape)))
        # ratio_used_for_svm = 0.05
        # index_list = [i for i in range(pred_all_.shape[0])]
        # num_used_for_svm = int(ratio_used_for_svm * pred_all_.shape[0])
        # idxs = random.sample(index_list, num_used_for_svm)
        train_x = np.vstack((add_train_x))#,cluster_features_all_[idxs,::]
        train_y = np.hstack((add_train_y))#,pred_all_[idxs]
        train_x_std = train_x#self.Classifier_std.fit_transform(train_x)
        self.Classifier.train(train_x=train_x_std, train_y=train_y)
        train_predict_label = self.Classifier.decision_function(train_x_std)
        fpr, tpr, threshold = roc_curve(train_y, train_predict_label, drop_intermediate=False,
                                        pos_label=1)
        maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
        print(threshold[maxindex])
        # FPR= []
        # TPR = []
        # Thresholds = []
        count = 0
        beta_imbalance = 1
        index = Data.calm_ratios.index(beta_imbalance)
        calm_test = Data.calm_test_generator(calm_ratio_index=index,standard = False)
        calm_encoded_, calm_decoded_, calm_cluster_features_, calm_pred_ = self.predict_in_sess(pilot_semi_,Data,sess, calm_test)
        # calm_cluster_features_ = calm_encoded_
        test_dataset_x = np.vstack((calm_cluster_features_, pilot_test_cluster_features_))
        test_dataset_y = np.hstack((np.zeros(calm_pred_.shape), np.zeros(pilot_test_pred_.shape) + 1))
        test_predict_label = self.hazard_factor_calculate(test_dataset_x)

        pod_7, far_7, csi_7, hss_7, gss_7, podN_7 = calculate_stat(test_predict_label>=threshold[maxindex],test_dataset_y,1)
        # results = test_predict_label>=threshold[maxindex]
        # all_hazard_factor = self.hazard_factor_calculate(cluster_features_all_)
        # mu = np.mean(all_hazard_factor)
        # sigma = np.std(all_hazard_factor)
        # scio.savemat('my_test.mat',{'my_test':results[178:],'test_predict_label':test_predict_label,'threshold':threshold[maxindex]})
        fpr, tpr, _ = roc_curve(test_dataset_y, test_predict_label, drop_intermediate=False,
                                        pos_label=1)
        # maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
        # # print(threshold[maxindex])
        # # Thresholds.append(threshold[maxindex])
        # FPR.append(fpr)
        # TPR.append(tpr)
        AUC, CSI = calculate_metrics_original(decision_values=test_predict_label, truth=test_dataset_y, hazard_id = 1)
        # scio.savemat('pred_clusternn.mat',{'decision_values':test_predict_label, 'truth':test_dataset_y, 'AUC': AUC})
        print(f'AUC={AUC}, CSI = {csi_7}')
        AUCs = AUC
        CSIs = csi_7
        count = count+1

        If_plot_ROC = True
        if If_plot_ROC:
            turb_v_r, calm_turb = turb_select(Data)
            _, _, turb_cluster_features_, _ = self.predict_in_sess(pilot_semi_,Data,sess, turb_v_r)
            _, _, calm_turb_cluster_features_, _ = self.predict_in_sess(pilot_semi_,Data,sess, calm_turb)
            test_dataset_x = np.vstack((calm_turb_cluster_features_, turb_cluster_features_))
            test_dataset_y = np.hstack((np.zeros(calm_turb_cluster_features_.shape[0]), np.zeros(turb_cluster_features_.shape[0]) + 1))
            test_predict_label = self.hazard_factor_calculate(test_dataset_x)
            fpr_turb, tpr_turb, _ = roc_curve(test_dataset_y, test_predict_label, drop_intermediate=False,
                                            pos_label=1)
            AUC_turb, _ = calculate_metrics_original(decision_values=test_predict_label, truth=test_dataset_y, hazard_id = 1)

            fig = plt.figure(figsize=(5, 5))
            plt.plot(fpr, tpr, label=fr'All types (auc = {np.round(AUC, 2)})', linewidth=2)
            plt.plot(fpr_turb, tpr_turb, label=fr'Turbulence (auc = {np.round(AUC_turb, 2)})', linewidth=2)
            plt.plot([0, 2], [0, 2], linewidth=0.5, color='C7')  # ,label='Random choice'
            plt.hlines(0.6685393258426966, 0, 2, colors='C7', linestyles='--')
            plt.text(0.2, 0.6885393258426966, ('WTWS (TPR=0.67)'))  # ,color = all_colors[1]
            plt.hlines(0.16,0,2,colors='C7',linestyles='-.')
            plt.text(0.05,0.18,('AWARE (TPR=0.16)'))
            plt.xlim((0, 1.01))
            plt.ylim((0, 1.01))
            plt.plot(0.15,0.8,'o')
            plt.legend()
            plt.title('AUC (Fig3c)')
            scio.savemat('fig3c_data.mat',{'fpr':fpr,'tpr':tpr,'fpr_turb':fpr_turb,'tpr_turb':tpr_turb})
            plt.savefig(f'../figures/roc.png',dpi=600)  #
            plt.savefig(f'../figures/roc.pdf')
        If_plot_hazard_factor_distribution = True
        if If_plot_hazard_factor_distribution:
            all_hazard_factor = self.hazard_factor_calculate(cluster_features_all_)
            mu = np.mean(all_hazard_factor)
            sigma = np.std(all_hazard_factor)
            all_features = (all_hazard_factor - mu) / sigma
            # mu = np.mean(all_features)
            # sigma = np.std(all_features)
            # all_features = (all_features - mu) / sigma
            optimal_threshold = threshold[maxindex]#(optimal_threshold- mu) / sigma
            plt.figure()
            import pandas as pd
            data = pd.Series(all_features)  # 将数据由数组转换成series形式
            n, bins, patches = plt.hist(data, 50, density=True, edgecolor='w',facecolor=self.all_colors[0])
            plt.vlines(optimal_threshold,0,np.max(n),label='Optimal threshold')
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
            # scio.savemat('fig4a_data.mat',{'hazard_factors':all_features,'normal_x':x,'normal_y':y})
            plt.title('Distribution of hazardous winds (Fig4a)')
            plt.savefig(f'../figures/distribution_hazard_factors.png',dpi=600)  #
            plt.savefig(f'../figures/distribution_hazard_factors.pdf')
        return AUCs,CSIs,optimal_threshold
    
    def cluster_visualization_tsne_test_other_airports(self,hazard_id, cluster_features_all_, pilot_test_cluster_features_,other_cluster_features_all_, 
                                                    pred_all_, pilot_test_pred_,other_pred_all_,
                                                    all_hazard_factor,pilot_test_hazard_factor,other_airports_hazard_factor):
        
        pca = TSNE(n_components=2,random_state=86)
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
        # idx_y_0 = np.argsort(np.abs(all_hazard_factor_idxs))[:20]
        # from scipy.optimize import curve_fit
        # def curv_func_rmse(x,  c,d,e):
        #     return c*x**2+d*x+e
        # popt, pcov = curve_fit(curv_func_rmse, X_unlabel_embedded[idx_y_0,0], X_unlabel_embedded[idx_y_0,1])
        # x_interp = np.arange(np.min(X_unlabel_embedded[idx_y_0,0]),np.max(X_unlabel_embedded[idx_y_0,0]))
        # plt.plot(x_interp, curv_func_rmse(x_interp, *popt),'C7',label='y=0')

        if hazard_id == 0:
            pred_all_ = 1 - pred_all_
            pilot_test_pred_ = 1 - pilot_test_pred_
            other_pred_all_ = 1 - other_pred_all_
        all_labels = ['All data in cluster 1', 'All data in cluster 2']
        pilot_labels = ['Hazardous wind in cluster 1', 'Hazardous wind in cluster 2']
        calm_labels = ['Calm wind in cluster 1', 'Calm wind in cluster 2']
        fig = plt.figure(figsize=(5, 4))
        clim = (-1,4)
        theta = np.pi/12
        rotationMatrix = np.array((np.cos(theta), -np.sin(theta), np.sin(theta),  np.cos(theta))).reshape(2,2)

        X_unlabel_embedded_rotated = np.matmul(X_unlabel_embedded,rotationMatrix)
        pilot_test_embedded_rotated = np.matmul(pilot_test_embedded,rotationMatrix)
        others_embedded_rotated = np.matmul(others_embedded,rotationMatrix)

        cf1_1 = plt.scatter(X_unlabel_embedded_rotated[:, 0], -X_unlabel_embedded_rotated[:, 1], marker='o',
                    c=all_hazard_factor_idxs[:],cmap='Spectral_r',alpha=1,s = 5,label='Unlabeled trainning data (HongKong)')
        # plt.plot(X_unlabel_embedded[idx_y_0,0],X_unlabel_embedded[idx_y_0,1],'.')

        cf2 = plt.scatter(pilot_test_embedded_rotated[::, 0], -pilot_test_embedded_rotated[::, 1],
                    marker='o', c=pilot_test_hazard_factor[::],cmap='Spectral_r',
                    alpha=1, edgecolors='k', linewidth=0.5, s=25,label='Test hazardous data (HongKong)')
        cf3 = plt.scatter(others_embedded_rotated[::, 0], -others_embedded_rotated[::, 1],
                    marker='*', c=other_airports_hazard_factor[::],cmap='Spectral_r',label='Test hazardous data (Other airports)', alpha=1, edgecolors='k', linewidth=0.5, s=125)
        cbar = fig.colorbar(mappable=cf1_1)
        cbar.mappable.set_clim(clim)
        cbar.remove()
        cbar = fig.colorbar(mappable=cf2)
        cbar.mappable.set_clim(clim)
        cbar.remove()
        cbar = fig.colorbar(mappable=cf3)
        cbar.mappable.set_clim(clim)
        # plt.colorbar()
        # plt.clim(clim)
        # plt.legend()
        # plt.show(block=True)
        # scio.savemat('fig2b_data.mat',{'X_unlabel_embedded':X_unlabel_embedded,
        #                                'pilot_test_embedded':pilot_test_embedded,
        #                                'others_embedded':others_embedded})
        plt.title('Established feature space (Fig2b)')
        plt.savefig(f'../figures/other_airports_feature.png',dpi=600)  #
        plt.savefig(f'../figures/other_airports_feature.pdf')

    def plot_fenji_violin(self,Data,pilot_train_hazard_factor,pilot_test_hazard_factor):
        pilot_turb_mags = Data.pilot_turb_mag_train
        pilot_ws_mags = Data.pilot_ws_mag_train
        pilot_turb_mags[np.where(np.isnan(pilot_turb_mags))] = 0
        pilot_ws_mags[np.where((np.abs(pilot_ws_mags)==15.1))] = 15
        # pilot_turb_mags[np.where((np.abs(pilot_turb_mags)==2.5))] = 2
        pilot_mags = np.vstack((np.abs(pilot_ws_mags.T), np.abs(pilot_turb_mags.T)))
        pilot_mags_unique = np.unique(pilot_mags, axis=1)
        pilot_mags_combine = np.abs(pilot_mags_unique[0,::])+np.abs(pilot_mags_unique[1,::])*turb_ratio
        index_sort = np.argsort(pilot_mags_combine)
        pilot_mags_unique = pilot_mags_unique[::,index_sort]
        pilot_feature_mean = []
        pilot_mag_names = []
        ratio_mags = []
        plt.figure()
        plt.title('Hazard intensity estimation (Fig4c)')
        
        
        pilot_turb_mags = Data.pilot_turb_mags_test
        pilot_ws_mags = Data.pilot_ws_mags_test
        pilot_turb_mags[np.where(np.isnan(pilot_turb_mags))] = 0
        pilot_mags = np.vstack((np.abs(pilot_ws_mags.T), np.abs(pilot_turb_mags.T)))
        pilot_mags_unique = np.unique(pilot_mags, axis=1)
        pilot_mags_combine = np.abs(pilot_mags_unique[0,::])+np.abs(pilot_mags_unique[1,::])*turb_ratio
        index_sort = np.argsort(pilot_mags_combine)
        pilot_mags_unique = pilot_mags_unique[::,index_sort]
        # pilot_train_ = Data.pilot_train_v_r
        # pilot_train_fenlei = []
        pilot_feature_mean = []
        pilot_mag_names = []
        ratio_mags = []
        pilot_violin = []
        plt.subplot(211)
        for i_unique_mag in range(pilot_mags_unique.shape[1]):
            index_ws_mag = np.where((np.abs(pilot_ws_mags) == pilot_mags_unique[0, i_unique_mag])
                                    & (np.abs(pilot_turb_mags) == pilot_mags_unique[1, i_unique_mag]))
            if index_ws_mag[0].shape[0]>5:
                label_for_legend = f'shear={pilot_mags_unique[0,i_unique_mag]},turb={pilot_mags_unique[1,i_unique_mag]}'
                pilot_feature_mean.append(np.nanmean(pilot_test_hazard_factor[index_ws_mag[0]]))
                pilot_mag_names.append(label_for_legend)
                ratio_mags.append(np.median(pilot_test_hazard_factor[index_ws_mag[0]]))
                plt.violinplot(pilot_test_hazard_factor[index_ws_mag[0]], [len(pilot_mag_names) - 1], showmedians=True)  #
                pilot_violin.append(pilot_test_hazard_factor[index_ws_mag[0]])
        plt.xticks(())
        plt.xlim((-0.5, len(pilot_mag_names) - 0.5))
        plt.ylabel('Value')
        plt.subplot(212)
        plt.plot(np.arange(len(pilot_mag_names)), ratio_mags, '-', marker='.')
        plt.xticks(np.arange(len(pilot_mag_names)), (pilot_mag_names), rotation=20)
        plt.ylabel('Median Value')
        plt.xlim((-0.5, len(pilot_mag_names) - 0.5))
        plt.tight_layout()    
        # scio.savemat('fig4b_data.mat',{'pilot_violin':pilot_violin,'ratio_mags':ratio_mags})
        plt.savefig(f'../figures/intensity_a_{self.a}_b_{self.b}_c_{self.c}_d_{self.d}_interval_{self.interval}_dim_{self.cluster_dim}.png')
        plt.savefig(f'../figures/intensity_a_{self.a}_b_{self.b}_c_{self.c}_d_{self.d}_interval_{self.interval}_dim_{self.cluster_dim}.pdf')
    def roc_calculate_pred(self, calm_test_hazard_factor_,pilot_test_hazard_factor_,optimal_threshold):
        test_dataset_factor = np.hstack((calm_test_hazard_factor_, pilot_test_hazard_factor_))
        test_dataset_y = np.hstack((np.zeros(calm_test_hazard_factor_.shape), np.zeros(pilot_test_hazard_factor_.shape[0]) + 1))
        pod_7, far_7, csi_7, hss_7, gss_7, podN_7 = calculate_stat(test_dataset_factor>=optimal_threshold,test_dataset_y,1)
        AUC, CSI = calculate_metrics_original(decision_values=test_dataset_factor, truth=test_dataset_y, hazard_id = 1)
        # print(f'AUC={AUC}, CSI = {csi_7}')
        return AUC,csi_7
    def cluster_visualization_tsne_test_pred(self, cluster_features_all_, other_cluster_features_all_, 
                                                    all_hazard_factor,other_airports_hazard_factor,pred_time = 15):

        pca = TSNE(n_components=2,random_state=86)
        ratio_used_for_tsne = 0.04
        index_list = [i for i in range(all_hazard_factor.shape[0])]
        num_used_for_tsne = int(ratio_used_for_tsne * all_hazard_factor.shape[0])
        random.seed(2)
        idxs = random.sample(index_list, num_used_for_tsne)
        all_embedded = pca.fit_transform(np.vstack((cluster_features_all_[idxs,::],  other_cluster_features_all_)))
        X_unlabel_embedded = all_embedded[:num_used_for_tsne, ::]
        others_embedded = all_embedded[-other_cluster_features_all_.shape[0]:,::]
        all_hazard_factor_idxs = all_hazard_factor[idxs]

        fig = plt.figure(figsize=(5, 4))
        clim = (-1,4)        
        theta = np.pi/2
        rotationMatrix = np.array((np.cos(theta), -np.sin(theta), np.sin(theta),  np.cos(theta))).reshape(2,2)

        X_unlabel_embedded_rotated = np.matmul(X_unlabel_embedded,rotationMatrix)
        others_embedded_rotated = np.matmul(others_embedded,rotationMatrix)
        cf1_1 = plt.scatter(X_unlabel_embedded_rotated[:, 0], -X_unlabel_embedded_rotated[:, 1], marker='o',
                    c=all_hazard_factor_idxs[:],cmap='Spectral_r',alpha=1,s = 5,label='Unlabeled trainning data')

        cf3 = plt.scatter(others_embedded_rotated[::, 0], -others_embedded_rotated[::, 1],
                    marker='o', c=other_airports_hazard_factor[::],cmap='Spectral_r',label='Predicted hazardous data', alpha=1, edgecolors='k', linewidth=0.5, s=25)
        cbar = fig.colorbar(mappable=cf1_1)
        cbar.mappable.set_clim(clim)
        cbar.remove()
        cbar = fig.colorbar(mappable=cf3)
        cbar.mappable.set_clim(clim)
        # cbar.remove()
        # plt.colorbar()
        # plt.clim(clim)
        # plt.legend()
        # plt.show(block=True)
        # scio.savemat('sfig8a_data.mat',{'X_unlabel_embedded':X_unlabel_embedded,
        #                                'others_embedded':others_embedded,
        #                                'other_airports_hazard_factor':other_airports_hazard_factor})
        plt.title(f'{pred_time}min predicted feature space (Supplementary Fig8 ab)')
        plt.savefig(f'../figures/{pred_time}_pred_feature.png',dpi=600)  #
        plt.savefig(f'../figures/{pred_time}_pred_feature.pdf')
    def cluster_visualization_tsne_test_pred_wrf(self, cluster_features_all_,other_cluster_features_all_, 
                                                    all_hazard_factor,other_airports_hazard_factor,pilots_time):
        pca = TSNE(n_components=2,random_state=86)
        ratio_used_for_tsne = 0.04
        index_list = [i for i in range(all_hazard_factor.shape[0])]
        num_used_for_tsne = int(ratio_used_for_tsne * all_hazard_factor.shape[0])
        random.seed(2)
        idxs = random.sample(index_list, num_used_for_tsne)
        all_embedded = pca.fit_transform(np.vstack((cluster_features_all_[idxs,::],  other_cluster_features_all_)))
        X_unlabel_embedded = all_embedded[:num_used_for_tsne, ::]
        others_embedded = all_embedded[-other_cluster_features_all_.shape[0]:,::]
        all_hazard_factor_idxs = all_hazard_factor[idxs]
        fig = plt.figure(figsize=(5, 4))
        clim = (-1,4)       
        theta = np.pi/2
        rotationMatrix = np.array((np.cos(theta), -np.sin(theta), np.sin(theta),  np.cos(theta))).reshape(2,2)

        X_unlabel_embedded_rotated = np.matmul(X_unlabel_embedded,rotationMatrix)
        others_embedded_rotated = np.matmul(others_embedded,rotationMatrix)
        markers = ['o','X','D','^','*']
        size_s = [55,55,55,55,125]
        cf1_1 = plt.scatter(X_unlabel_embedded_rotated[:, 0], -X_unlabel_embedded_rotated[:, 1], marker='o',
                    c=all_hazard_factor_idxs[:],cmap='Spectral_r',alpha=1,s = 5,label='Unlabeled trainning data')
        cbar = fig.colorbar(mappable=cf1_1)
        cbar.mappable.set_clim(clim)
        cbar.remove()
        for i_case in range(5):
            cf2 = plt.scatter(others_embedded_rotated[i_case*6:(i_case+1)*6, 0], -others_embedded_rotated[i_case*6:(i_case+1)*6, 1],
                    marker=markers[i_case], c=other_airports_hazard_factor[i_case*6:(i_case+1)*6],cmap='Spectral_r',
                    alpha=1, edgecolors='k', linewidth=0.5, s=size_s[i_case],label=f'Predicted hazardous data {pilots_time[i_case*6]}')        
            cbar = fig.colorbar(mappable=cf2)
            cbar.mappable.set_clim(clim)
            if i_case<4:
                cbar.remove()
        # plt.legend()
        # plt.colorbar()
        # plt.clim(clim)
        # plt.xlim((-55,70))
        # plt.ylim((35,85))
        plt.xticks(())
        plt.yticks(())      
        # plt.show()  
        # plt.title(f'wrf pred model')
        plt.title('wrf predicted feature space (Supplementary Fig6 b)')
        plt.savefig(f'../figures/PredModel_wrf.png',dpi=600)  #
        plt.savefig(f'../figures/PredModel_wrf.pdf')