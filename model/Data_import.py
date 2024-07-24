import  tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import os
os.environ["OMP_NUM_THREADS"] = '4'
import numpy as np
from math import ceil
import random
from sklearn import preprocessing
from tools import *
class data_frame(object):
    def __init__(self,seed = 1, calm_train_num = None, standard = True):
        
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # self.calm_range_set = calm_range_set
        all_corridor = all_data_read(2017,loc=3)
        pilot_corridor_2017, pilot_time_2017, pilot_wtws_2017, pilot_aware_2017,pilot_ws_mags_2017,pilot_turb_mags_2017 = pilot_corridor_read(2017, loc=3)
        pilot_corridor_2018, pilot_time_2018, pilot_wtws_2018, pilot_aware_2018,pilot_ws_mags_2018,pilot_turb_mags_2018 = pilot_corridor_read(2018, loc=3)
        pilot_corridor_2019, pilot_time_2019, pilot_wtws_2019, pilot_aware_2019,pilot_ws_mags_2019,pilot_turb_mags_2019 = pilot_corridor_read(2019, loc=3)
        pilot_corridor_2020, pilot_time_2020, pilot_wtws_2020, pilot_aware_2020,pilot_ws_mags_2020,pilot_turb_mags_2020 = pilot_corridor_read(2020, loc=3)
        wtws_test = np.hstack((pilot_wtws_2019,pilot_wtws_2020))

        wtws_acc_test = np.sum(np.hstack((pilot_wtws_2020,pilot_wtws_2019)))
        aware_acc_test = np.sum(np.hstack((pilot_aware_2020,pilot_aware_2019)))
        self.unlabeled_data_v_r = all_corridor
        self.pilot_train_v_r = np.vstack((pilot_corridor_2017,pilot_corridor_2018))  #
        self.pilot_test_v_r = np.vstack((pilot_corridor_2019, pilot_corridor_2020))  #
        
        # index = [9252, 28189, 47258, 51323, 51346, 51518, 55304, 55310, 57512, 57580, \
                 60251, 60255, 61674, 61815, 66826, 66879, 94804, 104019, 104041, 104074, \
                 110114, 112101, 120912, 121586, 121613, 121618, 137931, 137933, 182797, \
                    190478, 191201, 194766, 194867, 194869, 194891, 194893, 194899, 194902, 194958]
        index = []
        for i in np.arange(pilot_corridor_2017.shape[0]):
            for j in np.arange(all_corridor.shape[0]):
                if (pilot_corridor_2017[i,::]==all_corridor[j,::]).all():
                    index.append(j)
                    break
        print(index)
        all_corridor_calm = np.delete(all_corridor,index,axis = 0)
        self.unlabeled_data_v_r_calm = all_corridor_calm

        self.pilot_train_time = np.hstack((pilot_time_2017,pilot_time_2018))
        self.pilot_test_time = np.hstack(( pilot_time_2019, pilot_time_2020))
        pilot_ws_mags_test = np.vstack((pilot_ws_mags_2019.T, pilot_ws_mags_2020.T))  #
        pilot_turb_mags_test = np.vstack(( pilot_turb_mags_2019.T, pilot_turb_mags_2020.T))  #
        pilot_ws_mag_train = np.vstack((pilot_ws_mags_2017.T,pilot_ws_mags_2018.T))
        pilot_turb_mags_train = np.vstack((pilot_turb_mags_2017.T,pilot_turb_mags_2018.T))

        all_corridor = all_data_read(2017,loc=4)
        pilot_corridor_2017, pilot_time_2017, pilot_wtws_2017, pilot_aware_2017,pilot_ws_mags_2017,pilot_turb_mags_2017 = pilot_corridor_read(2017, loc=4)
        pilot_corridor_2018, pilot_time_2018, pilot_wtws_2018, pilot_aware_2018,pilot_ws_mags_2018,pilot_turb_mags_2018 = pilot_corridor_read(2018, loc=4)
        pilot_corridor_2019, pilot_time_2019, pilot_wtws_2019, pilot_aware_2019,pilot_ws_mags_2019,pilot_turb_mags_2019 = pilot_corridor_read(2019, loc=4)
        pilot_corridor_2020, pilot_time_2020, pilot_wtws_2020, pilot_aware_2020,pilot_ws_mags_2020,pilot_turb_mags_2020 = pilot_corridor_read(2020, loc=4)
        # index = [50899, 52917, 52922, 57466, 57468, 57469, 57473, 57481, 57487, 57494, 57567,\
                  57589, 57594, 59499, 59539, 59545, 60879, 61328, 64786, 65007, 65017, 65235,\
                      68220, 74605, 74606, 74617, 74755, 84735, 84739, 84749, 84752, 84755, 84757, \
                        84761, 84768, 89925, 90814, 90826, 111674, 111730, 112537, 112562, 112614, \
                            112628, 117885, 119556, 128684, 131695, 137191, 156028, 156030, 164379,\
                                  186482, 194004, 227853, 228066]
        for i in np.arange(pilot_corridor_2017.shape[0]):
            for j in np.arange(all_corridor.shape[0]):
                if (pilot_corridor_2017[i,::]==all_corridor[j,::]).all():
                    index.append(j)
                    break
        print(index)
        all_corridor_calm = np.delete(all_corridor,index,axis = 0)
        self.pilot_train_v_r = np.vstack((pilot_corridor_2017,pilot_corridor_2018,self.pilot_train_v_r))#
        self.pilot_test_v_r = np.vstack((pilot_corridor_2019,pilot_corridor_2020,self.pilot_test_v_r))#
        self.unlabeled_data_v_r = np.vstack((all_corridor,self.unlabeled_data_v_r))
        self.unlabeled_data_v_r_calm = np.vstack((all_corridor_calm,self.unlabeled_data_v_r_calm))
        # mean_all_train = np.mean(np.abs(self.unlabeled_data_v_r_calm),axis = 1)
        # self.unlabeled_data_v_r_calm = self.unlabeled_data_v_r_calm[mean_all_train<5,::]
        
        self.pilot_train_time = np.hstack((pilot_time_2017,pilot_time_2018, self.pilot_train_time))
        self.pilot_test_time = np.hstack((pilot_time_2019, pilot_time_2020, self.pilot_test_time))
        wtws_test = np.hstack((pilot_wtws_2019,pilot_wtws_2020,wtws_test))
        wtws_acc_test = wtws_acc_test+np.sum(np.hstack((pilot_wtws_2019,pilot_wtws_2020)))
        aware_acc_test = aware_acc_test+np.sum(np.hstack((pilot_aware_2020,pilot_aware_2019)))
        wtws_acc_test = wtws_acc_test / self.pilot_test_v_r.shape[0]
        aware_acc_test = aware_acc_test / self.pilot_test_v_r.shape[0]


        self.pilot_ws_mags_test = np.vstack(
            ( pilot_ws_mags_2019.T, pilot_ws_mags_2020.T, pilot_ws_mags_test))  #
        self.pilot_turb_mags_test = np.vstack(
            (pilot_turb_mags_2019.T, pilot_turb_mags_2020.T, pilot_turb_mags_test))  #

        self.pilot_ws_mag_train = np.vstack((pilot_ws_mags_2017.T,pilot_ws_mags_2018.T,pilot_ws_mag_train))
        self.pilot_turb_mag_train = np.vstack((pilot_turb_mags_2017.T,pilot_turb_mags_2018.T, pilot_turb_mags_train))

        self.DATA_NUM = self.unlabeled_data_v_r.shape[0]
        self.DATA_NUM_calm = self.unlabeled_data_v_r_calm.shape[0]
        # calm_train_num_unlabel = 10
        # calm_train = 5 * (np.array([i for i in range(calm_train_num_unlabel)]) - calm_train_num_unlabel * 0.5) / (calm_train_num_unlabel * 0.5)
        # self.calm_train = np.ones((calm_train_num_unlabel, self.pilot_train_v_r.shape[1])) * calm_train.reshape(-1, 1)
        if standard:
            self.std = preprocessing.StandardScaler()
            self.unlabeled_data_v_r = self.std.fit_transform(self.unlabeled_data_v_r)
            self.pilot_train_v_r = self.std.transform(self.pilot_train_v_r)
            self.pilot_test_v_r = self.std.transform(self.pilot_test_v_r)
            self.calm_train = self.std.transform(self.calm_train)

        self.calm_test_init()
        if calm_train_num == None:
            self.calm_train_num = self.pilot_train_v_r.shape[0]
        else:
            self.calm_train_num = calm_train_num
        self.calm_train_svm = self.calm_train_generator(standard = False)#np.ones((self.calm_train_num, self.pilot_train_v_r.shape[1])) * calm_train.reshape(-1, 1)
        self.calm_train = self.calm_train_svm
    def calm_train_generator(self,standard = False):
        all_corridor_2017_3,_ = all_data_read_by_month(2017, month = 10)
        all_corridor_2017_4,_ = all_data_read_by_month(2017, month = 11)
        all_corridor_2018_3,_ = all_data_read_by_month(2018, month = 10)
        all_corridor_2018_4,_ = all_data_read_by_month(2018, month = 11)
        all_calm_train = np.vstack((all_corridor_2017_3,all_corridor_2017_4,all_corridor_2018_3,all_corridor_2018_4))
        # index = [20489, 28011, 84616, 84829, 133615, 3215, 3938, 7503, 7604, 7606, 7628, 7630, 7636, 7639, 7695, 155180]
        index = []
        for i in np.arange(self.pilot_train_v_r.shape[0]):
            for j in np.arange(all_calm_train.shape[0]):
                if (self.pilot_train_v_r[i,::]==all_calm_train[j,::]).all():
                    index.append(j)
                    break
        print(index)
        all_calm_train = np.delete(all_calm_train,index,axis = 0)
        # mean_all_train = np.mean(np.abs(self.unlabeled_data_v_r_calm),axis = 1)
        # calm_train_idxs = np.where(mean_all_train<5)[0]
        # calm_train = self.unlabeled_data_v_r[random.sample(list(calm_train_idxs),self.calm_train_num),::]    
        calm_train = all_calm_train[random.sample(list(np.arange(all_calm_train.shape[0])),self.calm_train_num),::]

        return calm_train
    def calm_test_init(self):
        # all_corridor_2018_3 = all_data_read_by_month(2018, month = 10)
        # all_corridor_2018_4 = all_data_read_by_month(2018, month = 11)
        all_corridor_2019_3,name_month_2019_3 = all_data_read_by_month(2019, month = 10)
        all_corridor_2019_4,name_month_2019_4 = all_data_read_by_month(2019, month = 11)
        all_corridor_2020_3,name_month_2020_3 = all_data_read_by_month(2020, month = 10)
        all_corridor_2020_4,name_month_2020_4 = all_data_read_by_month(2020, month = 11)
        self.all_corridor_test = np.vstack((#all_corridor_2018_3,all_corridor_2018_4,
                                       all_corridor_2019_3,all_corridor_2019_4,
                                       all_corridor_2020_3,all_corridor_2020_4))
        self.calm_name_test = np.hstack((#all_corridor_2018_3,all_corridor_2018_4,
                                       name_month_2019_3,name_month_2019_4,
                                       name_month_2020_3,name_month_2020_4))
        # index = [189272]
        index = []
        for i in np.arange(self.pilot_test_v_r.shape[0]):
            for j in np.arange(self.all_corridor_test.shape[0]):
                if (self.pilot_test_v_r[i,::]==self.all_corridor_test[j,::]).all():
                    index.append(j)
                    break
        print(index)
        self.all_corridor_test = np.delete(self.all_corridor_test,index,axis = 0)
        self.calm_name_test = np.delete(self.calm_name_test,index,axis = 0)
        all_indexes = [i for i in range(self.all_corridor_test.shape[0])]
        self.calm_ratios = [1,10,100,200,400,600,800,1000]
        self.still_index = []
        self.calm_test_num = []
        for i in range(len(self.calm_ratios)):
            calm_test_num = int(self.calm_ratios[i]*self.pilot_test_v_r.shape[0])
            self.still_index.append(random.sample(all_indexes,calm_test_num))
            self.calm_test_num.append(calm_test_num)
    def calm_test_generator(self,calm_ratio_index=0,standard = True):
        # all_corridor_test = self.std.fit_transform(all_corridor_test)
        # self.still_index = np.argsort(np.std(all_corridor_test, axis=1))
        calm_test = self.all_corridor_test[self.still_index[calm_ratio_index], ::].reshape(
            (self.calm_test_num[calm_ratio_index], self.pilot_train_v_r.shape[1]))
        calm_test_names = self.calm_name_test[self.still_index[calm_ratio_index]]
        if standard:
            calm_test = self.std.fit_transform(calm_test)
        return calm_test
    def calm_test_generator_turb_or_shear(self, calm_ratio_=1, pilot_test_num = 0):
        calm_test_num = int(calm_ratio_*pilot_test_num)
        all_indexes = [i for i in range(self.all_corridor_test.shape[0])]
        calm_test = self.all_corridor_test[random.sample(all_indexes,calm_test_num),::]
        return calm_test
    def pilot_semi_learning_generator(self,semi_ratio):
        semi_supervise_pilot_num = ceil(self.pilot_train_v_r.shape[0] * semi_ratio)  # 70,90,110,
        semi_indexs = [i for i in range(self.pilot_train_v_r.shape[0])]  # pilot_corridor.shape[0]
        semi_index = random.sample(semi_indexs, semi_supervise_pilot_num)
        return self.pilot_train_v_r[semi_index,::]
    def time_read(self,index):
        #find the corresponding time stamp of the request pilot report
        return self.pilot_test_time[index]
    def read_data_for_transfer_learning(self, transfer_unsupervise_size = 5000):
        all_corridor_1 = all_2020_data_read_Northern_RW(loc=1)
        pilot_corridor_2020_HKG2_1, pilot_time_2020_HKG2_1, pilot_wtws_2020_HKG2_1, pilot_aware_2020_HKG2_1,_,_ = pilot_corridor_read(2020, loc=1)
        pilot_corridor_2021_HKG2_1, pilot_time_2021_HKG2_1, pilot_wtws_2021_HKG2_1, pilot_aware_2021_HKG2_1,_,_ = pilot_corridor_read(2021, loc=1)
        all_corridor_2 = all_2020_data_read_Northern_RW(loc=2)
        pilot_corridor_2020_HKG2_2, pilot_time_2020_HKG2_2, pilot_wtws_2020_HKG2_2, pilot_aware_2020_HKG2_2,_,_ = pilot_corridor_read(2020, loc=2)
        pilot_corridor_2021_HKG2_2, pilot_time_2021_HKG2_2, pilot_wtws_2021_HKG2_2, pilot_aware_2021_HKG2_2,_,_ = pilot_corridor_read(2021, loc=2)

        all_corridor_v_r = np.vstack((all_corridor_1, all_corridor_2))
        pilot_corridor_2020_HKG2 = np.vstack((pilot_corridor_2020_HKG2_1, pilot_corridor_2020_HKG2_2))
        pilot_corridor_2021_HKG2 = np.vstack((pilot_corridor_2021_HKG2_1, pilot_corridor_2021_HKG2_2))
        self.still_index_transfer = np.argsort(np.max(all_corridor_v_r,axis = 1)-np.min(all_corridor_v_r,axis = 1))

        self.all_corridor_v_r_transfer = self.std.transform(all_corridor_v_r)
        pilot_corridor_2020_HKG2 = self.std.transform(pilot_corridor_2020_HKG2)
        pilot_corridor_2021_HKG2 = self.std.transform(pilot_corridor_2021_HKG2)


        index_list = [i for i in range(self.all_corridor_v_r_transfer.shape[0])]
        idxs = random.sample(index_list, transfer_unsupervise_size)
        self.unlabeled_data_transfer = self.all_corridor_v_r_transfer[idxs, ::]

        transfer_idxs = [22, 6, 40, 3, 0, 36, 39, 17, 14, 9]
        self.pilot_train_transfer = pilot_corridor_2020_HKG2[transfer_idxs]
        transfer_index_list = [i for i in range(pilot_corridor_2020_HKG2.shape[0])]
        idx_list = transfer_index_list
        for i_idx in range(len(transfer_idxs)):
            idx_list.remove(transfer_idxs[i_idx])
        self.pilot_test_transfer = np.vstack((pilot_corridor_2021_HKG2,pilot_corridor_2020_HKG2[idx_list]))

    def calm_test_transfer_generator(self, calm_ratio=1):
        calm_test_num = int(calm_ratio*self.pilot_test_transfer.shape[0])
        calm_test_transfer = self.all_corridor_v_r_transfer[self.still_index_transfer[:calm_test_num], ::].reshape(
            (calm_test_num, self.pilot_train_transfer.shape[1]))
        return calm_test_transfer
    def read_data_month(self,year):
        corridor_month = []
        for month_i in range(12):
            corridor_month_i = all_data_read_by_month(year, month=month_i + 1)
            if corridor_month_i.shape[0]>0:
                corridor_month_i = self.std.transform(corridor_month_i)
            corridor_month.append(corridor_month_i)
        return corridor_month
    def read_data_month_transfer(self,year):
        corridor_month = []
        for month_i in range(12):
            corridor_month_i = all_data_read_by_month_north(year, month=month_i + 1)
            if corridor_month_i.shape[0]>0:
                corridor_month_i = self.std.transform(corridor_month_i)
            corridor_month.append(corridor_month_i)
        return corridor_month
