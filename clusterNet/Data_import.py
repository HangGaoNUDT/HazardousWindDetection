import os
os.environ["OMP_NUM_THREADS"] = '4'
from tools import *
def data_import():
    all_corridor = all_2017_data_read(loc=3)
    pilot_corridor_2017,pilot_time_2017,pilot_wtws_2017,pilot_aware_2017 = pilot_corridor_read(2017,loc=3)
    pilot_corridor_2018,pilot_time_2018,pilot_wtws_2018,pilot_aware_2018 = pilot_corridor_read(2018,loc=3)
#     pilot_corridor_2019,pilot_time_2019,pilot_wtws_2019,pilot_aware_2019 = pilot_corridor_read(2019,loc=3)
#     pilot_corridor_2020,pilot_time_2020,pilot_wtws_2020,pilot_aware_2020 = pilot_corridor_read(2020,loc=3)

    unlabeled_data = all_corridor
    pilot_corridor = np.vstack((pilot_corridor_2017))#
    pilot_test = np.vstack((pilot_corridor_2018))#,pilot_corridor_2019,pilot_corridor_2020

    wtws_acc_2017 = np.sum(pilot_wtws_2017)
    aware_acc_2017 = np.sum(pilot_aware_2017)
    wtws_acc_test = np.sum(np.hstack((pilot_wtws_2018)))#,pilot_wtws_2019,pilot_wtws_2020
    aware_acc_test = np.sum(np.hstack((pilot_aware_2018)))#,pilot_aware_2019,pilot_aware_2020

    # wind records on the 4th glide path
    all_corridor = all_2017_data_read(loc=4)
    pilot_corridor_2017,pilot_time_2017,pilot_wtws_2017,pilot_aware_2017 = pilot_corridor_read(2017,loc=4)
    pilot_corridor_2018,pilot_time_2018,pilot_wtws_2018,pilot_aware_2018 = pilot_corridor_read(2018,loc=4)
#     pilot_corridor_2019,pilot_time_2019,pilot_wtws_2019,pilot_aware_2019 = pilot_corridor_read(2019,loc=4)
#     pilot_corridor_2020,pilot_time_2020,pilot_wtws_2020,pilot_aware_2020 = pilot_corridor_read(2020,loc=4)

    unlabeled_data = np.vstack((all_corridor,unlabeled_data))
    pilot_corridor = np.vstack((pilot_corridor_2017,pilot_corridor))
    pilot_test = np.vstack((pilot_corridor_2018,pilot_test))#,pilot_corridor_2019,pilot_corridor_2020
    # Accuracy of the operational systems
    wtws_acc_2017 = wtws_acc_2017+np.sum(pilot_wtws_2017)
    aware_acc_2017 = aware_acc_2017+np.sum(pilot_aware_2017)
    wtws_acc_test = wtws_acc_test+np.sum(np.hstack((pilot_wtws_2018)))#,pilot_wtws_2019,pilot_wtws_2020
    aware_acc_test = aware_acc_test+np.sum(np.hstack((pilot_aware_2018)))#,pilot_aware_2019,pilot_aware_2020
    wtws_acc_2017 = wtws_acc_2017 / pilot_corridor.shape[0]
    aware_acc_2017 = aware_acc_2017 / pilot_corridor.shape[0]
    wtws_acc_test = wtws_acc_test / pilot_test.shape[0]
    aware_acc_test = aware_acc_test / pilot_test.shape[0]
    return unlabeled_data,pilot_corridor,pilot_test,wtws_acc_2017,aware_acc_2017,wtws_acc_test,aware_acc_test