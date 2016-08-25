#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression as LR
import re

from add_woe_dynamic import add_woe_dynamic
from get_average_interval import get_average_interval

lr = LR()

def wrapper(X, y, clf, feature_num):
    clf.fit(X,y)
    selector = RFE(clf, n_features_to_select = feature_num)
    selector.fit(X,y)
    return selector


if __name__ == '__main__':
    #数据准备
    data_1 = pd.read_csv('~/data_research/feature_data_bairong/feature_data_1.csv',encoding = 'utf-8')
    data_2 = pd.read_csv('~/data_research/feature_data_bairong/feature_data_2.csv',encoding = 'utf-8')
    data_3 = pd.read_csv('~/data_research/feature_data_bairong/feature_data_3.csv',encoding = 'utf-8')
    data_4 = pd.read_csv('~/data_research/feature_data_bairong/feature_data_4.csv',encoding = 'utf-8')
    data_5 = pd.read_csv('~/data_research/feature_data_bairong/feature_data_5.csv',encoding = 'utf-8')
    data = pd.concat([data_1, data_2, data_3, data_4, data_5], ignore_index = True)
    data = data.fillna('fuck')
    data['y'] = 1
    data['pivot_table'] = 1
    data.loc[data['M_status'] == u'未处理', 'y'] = 0
    data.loc[data['status'] == 'reject', 'y'] = 0

    a = pd.read_csv('~/data_research/feature_result_1.csv',header = None)
    feature_name = list(a[0])

    start = data.shape[1]

    for i in feature_name:
        if re.compile('^acm_').match(i):
            temp = data.pivot_table('pivot_table', index = [i], columns = ['y'], aggfunc = 'count')
        if re.compile('^al_').match(i):
            temp_1 = data[i]
            temp_2 = temp_1[temp_1!='fuck']
            bins = get_average_interval(temp_2, 10)
            index = pd.cut(temp_1, bins, right = False).replace(np.nan,'fuck')
            data[i] = index
            temp = data.pivot_table('pivot_table', index = index, columns = ['y'], aggfunc = 'count')
        temp = temp.fillna(0)
        series_1 = temp[1]
        series_0 = temp[0]
        add_woe_dynamic(data, series_1, series_0, i)

    X = data.iloc[:,start:]
    y = data['y']
    feature_name_woe = data.columns[start:]
    lr = LR()
    selector = wrapper(X, y, lr, 5)
    print feature_name_woe[selector.support_]



