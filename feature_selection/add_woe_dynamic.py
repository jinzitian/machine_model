#-*- coding: utf-8 -*-

import pandas as pd
import numpy as np


#可以动态处理区间为0的情况IV计算函数
#dataframe是包含‘y’和‘pivot_table‘专门计数的列的
#如果series_1的索引是区间形式，则dataframe的特征的值列也是化为区间形势的特征值，这样才能计算
def add_woe_dynamic(dataframe, series_1, series_0, feature_name):
    temp = pd.DataFrame(series_1).copy()
    temp['woe'] = 0
    total_1 = series_1.sum()
    total_0 = series_0.sum()
    length = len(series_1)
    j = 0
    index_list = []
    while j <= length-1:
        if series_1.iloc[j] == 0 and series_0.iloc[j] != 0:
            sub_index = []
            while j<=length-1:
                sub_index.append(series_1.index[j])
                if series_1.iloc[j] != 0:
                    break
                j += 1
            index_list.append(sub_index)
            j += 1

        elif series_1.iloc[j] != 0 and series_0.iloc[j] == 0:
            sub_index = []
            while j<=length-1:
                sub_index.append(series_1.index[j])
                if series_0.iloc[j] != 0:
                    break
                j += 1
            index_list.append(sub_index)
            j += 1

        elif series_1.iloc[j] == 0 and series_0.iloc[j] == 0:
            sub_index = []
            while j<=length-1:
                sub_index.append(series_1.index[j])
                if series_1.iloc[j] != 0 and series_0.iloc[j] != 0:
                    break
                j += 1
            index_list.append(sub_index)
            j += 1

        else :
            index_list.append([series_1.index[j]])
            j += 1
        
    if len(index_list)>=3:
        if series_1[index_list[-1]].sum() == 0 or series_0[index_list[-1]].sum() == 0:
            temp = index_list.pop()
            index_list[-1].extend(temp)
    if len(index_list)==1:
        print 'woe_error'
        return 0
    if len(index_list)==2:
        if series_1[index_list[-1]].sum() == 0 or series_0[index_list[-1]].sum() == 0:
            print 'woe_error'
            return 0 
    for i in index_list:
        good = series_1[i].sum()*1.0/total_1
        bad = series_0[i].sum()*1.0/total_0
        temp.loc[i,'woe'] = np.log(good/bad)
    #woe保存到文件
    with open('/home/jinzitian/machine_model/data/%s_woe'%feature_name,'w') as a:
        a.write(str({temp.index[i]:temp['woe'].iloc[i] for i in range(len(temp['woe']))}))
    dataframe[feature_name + '_woe'] = dataframe[feature_name].map(lambda x:temp['woe'].loc[x])

