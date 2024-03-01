'''AE训练好后就固定，只训练FC使得两类的q越大越好'''
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
    def __init__(self,seed = 1, calm_train_num = 10, calm_range_set = 5):
        
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.calm_train_num = calm_train_num
        self.calm_range_set = calm_range_set
        all_corridor = all_data_read(2017,loc=3)
        pilot_corridor_2017, pilot_time_2017, pilot_wtws_2017, pilot_aware_2017,pilot_ws_mags_2017,pilot_turb_mags_2017 = pilot_corridor_read(2017, loc=3)
        pilot_corridor_2018, pilot_time_2018, pilot_wtws_2018, pilot_aware_2018,pilot_ws_mags_2018,pilot_turb_mags_2018 = pilot_corridor_read(2018, loc=3)
        pilot_corridor_2019, pilot_time_2019, pilot_wtws_2019, pilot_aware_2019,pilot_ws_mags_2019,pilot_turb_mags_2019 = pilot_corridor_read(2019, loc=3)
        pilot_corridor_2020, pilot_time_2020, pilot_wtws_2020, pilot_aware_2020,pilot_ws_mags_2020,pilot_turb_mags_2020 = pilot_corridor_read(2020, loc=3)

        self.unlabeled_data_v_r = all_corridor
        self.pilot_train_v_r = np.vstack((pilot_corridor_2017))  #
        self.pilot_test_v_r = np.vstack((pilot_corridor_2018, pilot_corridor_2019, pilot_corridor_2020))  #
        self.pilot_train_time = pilot_time_2017
        self.pilot_test_time = np.hstack((pilot_time_2018, pilot_time_2019, pilot_time_2020))
        pilot_ws_mags_test = np.vstack((pilot_ws_mags_2018.T, pilot_ws_mags_2019.T, pilot_ws_mags_2020.T))  #
        pilot_turb_mags_test = np.vstack((pilot_turb_mags_2018.T, pilot_turb_mags_2019.T, pilot_turb_mags_2020.T))  #

        all_corridor = all_data_read(2017,loc=4)
        pilot_corridor_2017, pilot_time_2017, pilot_wtws_2017, pilot_aware_2017,pilot_ws_mags_2017,pilot_turb_mags_2017 = pilot_corridor_read(2017, loc=4)
        pilot_corridor_2018, pilot_time_2018, pilot_wtws_2018, pilot_aware_2018,pilot_ws_mags_2018,pilot_turb_mags_2018 = pilot_corridor_read(2018, loc=4)
        pilot_corridor_2019, pilot_time_2019, pilot_wtws_2019, pilot_aware_2019,pilot_ws_mags_2019,pilot_turb_mags_2019 = pilot_corridor_read(2019, loc=4)
        pilot_corridor_2020, pilot_time_2020, pilot_wtws_2020, pilot_aware_2020,pilot_ws_mags_2020,pilot_turb_mags_2020 = pilot_corridor_read(2020, loc=4)

        self.pilot_train_v_r = np.vstack((pilot_corridor_2017,self.pilot_train_v_r))#
        self.pilot_test_v_r = np.vstack((pilot_corridor_2018,pilot_corridor_2019,pilot_corridor_2020,self.pilot_test_v_r))#
        self.unlabeled_data_v_r = np.vstack((all_corridor,self.unlabeled_data_v_r))
        self.pilot_train_time = np.hstack((pilot_time_2017,self.pilot_train_time))
        self.pilot_test_time = np.hstack((pilot_time_2018, pilot_time_2019, pilot_time_2020, self.pilot_test_time))

        self.pilot_ws_mags_test = np.vstack(
            (pilot_ws_mags_2018.T, pilot_ws_mags_2019.T, pilot_ws_mags_2020.T, pilot_ws_mags_test))  #
        self.pilot_turb_mags_test = np.vstack(
            (pilot_turb_mags_2018.T, pilot_turb_mags_2019.T, pilot_turb_mags_2020.T, pilot_turb_mags_test))  #

        self.DATA_NUM = self.unlabeled_data_v_r.shape[0]

        self.std = preprocessing.StandardScaler()
        self.unlabeled_data_v_r = self.std.fit_transform(self.unlabeled_data_v_r)
        self.pilot_train_v_r = self.std.transform(self.pilot_train_v_r)
        self.pilot_test_v_r = self.std.transform(self.pilot_test_v_r)

        calm_train = self.calm_range_set * (np.array([i for i in range(self.calm_train_num)]) - self.calm_train_num * 0.5) / (self.calm_train_num * 0.5)
        calm_train = np.ones((self.calm_train_num, self.pilot_train_v_r.shape[1])) * calm_train.reshape(-1, 1)
        self.calm_train = self.std.transform(calm_train)
        self.calm_test_init()
    def calm_test_init(self):
        all_corridor_2018_3 = all_data_read_by_month(2018, month = 10)
        all_corridor_2018_4 = all_data_read_by_month(2018, month = 11)
        all_corridor_2019_3 = all_data_read_by_month(2019, month = 10)
        all_corridor_2019_4 = all_data_read_by_month(2019, month = 11)
        all_corridor_2020_3 = all_data_read_by_month(2020, month = 10)
        all_corridor_2020_4 = all_data_read_by_month(2020, month = 11)
        self.all_corridor_test = np.vstack((all_corridor_2018_3,all_corridor_2018_4,
                                       all_corridor_2019_3,all_corridor_2019_4,
                                       all_corridor_2020_3,all_corridor_2020_4))
        all_indexes = [i for i in range(self.all_corridor_test.shape[0])]
        self.calm_ratios = [1,10,100,200,400,600,800,1000]
        self.still_index = []
        self.calm_test_num = []
        for i in range(len(self.calm_ratios)):
            calm_test_num = int(self.calm_ratios[i]*self.pilot_test_v_r.shape[0])
            self.still_index.append(random.sample(all_indexes,calm_test_num))
            self.calm_test_num.append(calm_test_num)
    def calm_test_generator(self,calm_ratio_index=0,):
        # all_corridor_test = self.std.fit_transform(all_corridor_test)
        # self.still_index = np.argsort(np.std(all_corridor_test, axis=1))
        calm_test = self.all_corridor_test[self.still_index[calm_ratio_index], ::].reshape(
            (self.calm_test_num[calm_ratio_index], self.pilot_train_v_r.shape[1]))
        calm_test = self.std.fit_transform(calm_test)
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
class read_seasonal_occurrence_prob_21_years(object):
    def __init__(self):
        self.hkia_statistics_month_2017 = np.array([35, 31, 35, 35, 35, 35, 36, 36, 35, 36, 35, 37])
        self.hkia_statistics_month_2018 = np.array([36, 32, 36, 35, 36, 35, 37, 37, 34, 37, 35, 37])
        self.hkia_statistics_month_2019 = np.array([37, 32, 37, 36, 36, 35, 37, 35, 33, 34, 33, 34])
        self.hkia_statistics_month_2020 = np.array([33, 18, 12, 9, 11, 10, 10, 10, 11, 11, 13, 13])
        self.hkia_statistics_month_2021 = np.array([11, 8, 11, 10, 11, 11, 12, 13, 14, 14, 15, 15])

        self.hkia_statistics_month_21_years = np.array(
            [548.1, 485.2, 534.2, 522, 523.9, 512.1, 541.5, 548.7, 529.2, 553.5, 541.3, 559.3])
        years_21_years_loc12, mm_21_years_loc12, HH_21_years_loc12 = self.read_pilot_reports_statistics(
            '..\Data_prep\\pilot_reported_21_years_loc12.mat')
        years_21_years_loc34, mm_21_years_loc34, HH_21_years_loc34 = self.read_pilot_reports_statistics(
            '..\\Data_prep\\pilot_reported_21_years_loc34.mat')
        pilot_month_21_years_loc12 = self.pilot_month_count(mm_21_years_loc12)
        pilot_month_21_years_loc34 = self.pilot_month_count(mm_21_years_loc34)
        ratio_pilot_month_21_years_loc12 = self.pilot_month_count_norm(mm_21_years_loc12,
                                                                       self.hkia_statistics_month_21_years)
        ratio_pilot_month_21_years_loc34 = self.pilot_month_count_norm(mm_21_years_loc34,
                                                                       self.hkia_statistics_month_21_years)
        self.ratio_pilot_month_21_years_north = self.normalize(ratio_pilot_month_21_years_loc12)
        self.ratio_pilot_month_21_years_south = self.normalize(ratio_pilot_month_21_years_loc34)

    def read_pilot_reports_statistics(self, path_name):
        pilot_reported = scio.loadmat(path_name)
        locs_ = pilot_reported['pilot_reported'][0, 0][0]
        years_ = pilot_reported['pilot_reported'][0, 0][1]
        mm_ = pilot_reported['pilot_reported'][0, 0][2]
        HH_ = pilot_reported['pilot_reported'][0, 0][3]
        return years_, mm_, HH_

    def pilot_month_count(self,mm_array):
        ratio_pilot_month = np.zeros((1, 12))
        for month_i in range(12):
            ratio_pilot_month[0, month_i] = np.sum(mm_array == (month_i + 1))
        return ratio_pilot_month


    def pilot_month_count_norm(self,mm_array, hkia_statistics_month):
        ratio_pilot_month = np.zeros((1, 12))
        for month_i in range(12):
            ratio_pilot_month[0, month_i] = np.sum(mm_array == (month_i + 1)) / hkia_statistics_month[month_i]
        return ratio_pilot_month
    def normalize(self,a):
        return (a-np.min(a))/(np.max(a)-np.min(a))
# for seed in range(1,100):
#     Data = data_frame(seed = seed)
#     Data.calm_test_init()
#     Data.calm_test_generator()
