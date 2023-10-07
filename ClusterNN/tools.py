import scipy.io as scio
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
def all_2017_data_read(loc):
    all_data_1 = scio.loadmat(f'../Data_prep/all_2017_data_loc{loc}_1.mat')
    all_corridor_1 = all_data_1['all_2017_data']['corridor_v_r'][0][0]
    all_data_2 = scio.loadmat(f'../Data_prep/all_2017_data_loc{loc}_2.mat')
    all_corridor_2 = all_data_2['all_2017_data']['corridor_v_r'][0][0]
    all_data_3 = scio.loadmat(f'../Data_prep/all_2017_data_loc{loc}_3.mat')
    all_corridor_3 = all_data_3['all_2017_data']['corridor_v_r'][0][0]
    all_corridor = np.vstack((all_corridor_1,all_corridor_2,all_corridor_3))
    return all_corridor

def pilot_corridor_read(year,loc):
    pilot_reported = scio.loadmat(f'../Data_prep/pilot_reported_{year}loc{loc}.mat')
    pilot_corridor = pilot_reported['pilot_reported'][0][0][1]
    pilot_time = pilot_reported['pilot_reported'][0][0][2]
    pilot_wtws = pilot_reported['pilot_reported'][0][0][3]
    pilot_aware = pilot_reported['pilot_reported'][0][0][4]
    pilot_ws_mags = pilot_reported['pilot_reported'][0][0][5]
    pilot_turb_mags = pilot_reported['pilot_reported'][0][0][6]
    return pilot_corridor,pilot_time,pilot_wtws,pilot_aware,pilot_ws_mags,pilot_turb_mags

def all_data_read_by_month(year,month):
    loc = 3
    all_data_1 = scio.loadmat(f'../Data_prep/all_{year}_data_loc{loc}_1.mat')
    all_data_2 = scio.loadmat(f'../Data_prep/all_{year}_data_loc{loc}_2.mat')
    all_data_3 = scio.loadmat(f'../Data_prep/all_{year}_data_loc{loc}_3.mat')
    if year == 2017:
        all_corridor_1 = all_data_1['all_2017_data']['corridor_v_r'][0][0]
        txt_name_1 = all_data_1['all_2017_data']['txt_name'][0][0]
        all_corridor_2 = all_data_2['all_2017_data']['corridor_v_r'][0][0]
        txt_name_2 = all_data_2['all_2017_data']['txt_name'][0][0]
        all_corridor_3 = all_data_3['all_2017_data']['corridor_v_r'][0][0]
        txt_name_3 = all_data_3['all_2017_data']['txt_name'][0][0]
    else:
        all_corridor_1 = all_data_1['all_data']['corridor_v_r'][0][0]
        txt_name_1 = all_data_1['all_data']['txt_name'][0][0]
        all_corridor_2 = all_data_2['all_data']['corridor_v_r'][0][0]
        txt_name_2 = all_data_2['all_data']['txt_name'][0][0]
        all_corridor_3 = all_data_3['all_data']['corridor_v_r'][0][0]
        txt_name_3 = all_data_3['all_data']['txt_name'][0][0]

    month_indexs_1 = np.array([int(txt_name_1[i][4:6]) for i in range(txt_name_1.shape[0])])
    month_indexs_2 = np.array([int(txt_name_2[i][4:6]) for i in range(txt_name_2.shape[0])])
    month_indexs_3 = np.array([int(txt_name_3[i][4:6]) for i in range(txt_name_3.shape[0])])
    indexs_1 = np.where(month_indexs_1==month)
    indexs_2 = np.where(month_indexs_2==month)
    indexs_3 = np.where(month_indexs_3==month)
    corridor_month = np.vstack((all_corridor_1[indexs_1],all_corridor_2[indexs_2],all_corridor_3[indexs_3]))
    loc = 4
    all_data_1 = scio.loadmat(f'../Data_prep/all_{year}_data_loc{loc}_1.mat')
    all_data_2 = scio.loadmat(f'../Data_prep/all_{year}_data_loc{loc}_2.mat')
    all_data_3 = scio.loadmat(f'../Data_prep/all_{year}_data_loc{loc}_3.mat')
    if year == 2017:
        all_corridor_1 = all_data_1['all_2017_data']['corridor_v_r'][0][0]
        txt_name_1 = all_data_1['all_2017_data']['txt_name'][0][0]
        all_corridor_2 = all_data_2['all_2017_data']['corridor_v_r'][0][0]
        txt_name_2 = all_data_2['all_2017_data']['txt_name'][0][0]
        all_corridor_3 = all_data_3['all_2017_data']['corridor_v_r'][0][0]
        txt_name_3 = all_data_3['all_2017_data']['txt_name'][0][0]
    else:
        all_corridor_1 = all_data_1['all_data']['corridor_v_r'][0][0]
        txt_name_1 = all_data_1['all_data']['txt_name'][0][0]
        all_corridor_2 = all_data_2['all_data']['corridor_v_r'][0][0]
        txt_name_2 = all_data_2['all_data']['txt_name'][0][0]
        all_corridor_3 = all_data_3['all_data']['corridor_v_r'][0][0]
        txt_name_3 = all_data_3['all_data']['txt_name'][0][0]

    month_indexs_1 = np.array([int(txt_name_1[i][4:6]) for i in range(txt_name_1.shape[0])])
    month_indexs_2 = np.array([int(txt_name_2[i][4:6]) for i in range(txt_name_2.shape[0])])
    month_indexs_3 = np.array([int(txt_name_3[i][4:6]) for i in range(txt_name_3.shape[0])])
    indexs_1 = np.where(month_indexs_1==month)
    indexs_2 = np.where(month_indexs_2==month)
    indexs_3 = np.where(month_indexs_3==month)
    corridor_month = np.vstack((corridor_month,all_corridor_1[indexs_1],all_corridor_2[indexs_2],all_corridor_3[indexs_3]))

    return corridor_month

def all_2020_data_read_Northern_RW(loc):
    all_data_1 = scio.loadmat(f'../Data_prep/HKG2_Data_test/all_2020_data_loc{loc}_1.mat')
    all_corridor_1 = all_data_1['all_data']['corridor_v_r'][0][0]
    all_data_2 = scio.loadmat(f'../Data_prep/HKG2_Data_test/all_2020_data_loc{loc}_2.mat')
    all_corridor_2 = all_data_2['all_data']['corridor_v_r'][0][0]
    all_data_3 = scio.loadmat(f'../Data_prep/HKG2_Data_test/all_2020_data_loc{loc}_3.mat')
    all_corridor_3 = all_data_3['all_data']['corridor_v_r'][0][0]
    all_corridor = np.vstack((all_corridor_1,all_corridor_2,all_corridor_3))
    return all_corridor

def all_data_read_by_month_north(year,month):
    loc = 1
    all_data_1 = scio.loadmat(f'../Data_prep/HKG2_Data_test/all_{year}_data_loc{loc}_1.mat')
    all_data_2 = scio.loadmat(f'../Data_prep/HKG2_Data_test/all_{year}_data_loc{loc}_2.mat')
    all_data_3 = scio.loadmat(f'../Data_prep/HKG2_Data_test/all_{year}_data_loc{loc}_3.mat')
    all_corridor_1 = all_data_1['all_data']['corridor_v_r'][0][0]
    txt_name_1 = all_data_1['all_data']['txt_name'][0][0]
    all_corridor_2 = all_data_2['all_data']['corridor_v_r'][0][0]
    txt_name_2 = all_data_2['all_data']['txt_name'][0][0]
    all_corridor_3 = all_data_3['all_data']['corridor_v_r'][0][0]
    txt_name_3 = all_data_3['all_data']['txt_name'][0][0]

    month_indexs_1 = np.array([int(txt_name_1[i][4:6]) for i in range(txt_name_1.shape[0])])
    month_indexs_2 = np.array([int(txt_name_2[i][4:6]) for i in range(txt_name_2.shape[0])])
    month_indexs_3 = np.array([int(txt_name_3[i][4:6]) for i in range(txt_name_3.shape[0])])
    indexs_1 = np.where(month_indexs_1==month)
    indexs_2 = np.where(month_indexs_2==month)
    indexs_3 = np.where(month_indexs_3==month)
    corridor_month = np.vstack((all_corridor_1[indexs_1],all_corridor_2[indexs_2],all_corridor_3[indexs_3]))
    loc = 2
    all_data_1 = scio.loadmat(f'../Data_prep/HKG2_Data_test/all_{year}_data_loc{loc}_1.mat')
    all_data_2 = scio.loadmat(f'../Data_prep/HKG2_Data_test/all_{year}_data_loc{loc}_2.mat')
    all_data_3 = scio.loadmat(f'../Data_prep/HKG2_Data_test/all_{year}_data_loc{loc}_3.mat')

    all_corridor_1 = all_data_1['all_data']['corridor_v_r'][0][0]
    txt_name_1 = all_data_1['all_data']['txt_name'][0][0]
    all_corridor_2 = all_data_2['all_data']['corridor_v_r'][0][0]
    txt_name_2 = all_data_2['all_data']['txt_name'][0][0]
    all_corridor_3 = all_data_3['all_data']['corridor_v_r'][0][0]
    txt_name_3 = all_data_3['all_data']['txt_name'][0][0]

    month_indexs_1 = np.array([int(txt_name_1[i][4:6]) for i in range(txt_name_1.shape[0])])
    month_indexs_2 = np.array([int(txt_name_2[i][4:6]) for i in range(txt_name_2.shape[0])])
    month_indexs_3 = np.array([int(txt_name_3[i][4:6]) for i in range(txt_name_3.shape[0])])
    indexs_1 = np.where(month_indexs_1==month)
    indexs_2 = np.where(month_indexs_2==month)
    indexs_3 = np.where(month_indexs_3==month)
    corridor_month = np.vstack((corridor_month,all_corridor_1[indexs_1],all_corridor_2[indexs_2],all_corridor_3[indexs_3]))

    return corridor_month