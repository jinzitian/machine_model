#-*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd

from logistic_model_estimate import get_feature_woe
from logistic_model_estimate import _one_to_woe
from logistic_model_estimate import to_woe

#模型选取的特征
feature_list = ['call_count_per_day', 'phone_loan_times_per_platform',\
                'idcard_loan_platform_num', 'idcard_loan_times_per_platform']

#思路
#woe_dict应该是已经计算好的存储在数据库或者文件中的区间woe值，模型接收数据前应该经过woe映射的转换，然后再使用
#woe_dict应该有一个单独的计算脚本函数计算生成，并保存到文件或者数据库中
#数据服务器在启动时应该讲woe_dict读取到内存中，方便随时处理接收到的数据，再传入模型
woe_dict = {}
for i in feature_list:
    woe_dict[i] = get_feature_woe(i)

#读取序列化实例的函数
def read_model(model_file):
    a = open(model_file, 'rb')
    lr = pickle.load(a)
    a.close()
    return lr

#获取评分卡函数
def get_scorecard(lr, b = 500 , o = 1, p = 20):
    def scorecard(X):
        p0, p1 = lr.predict_log_proba(X).reshape(-1)
        return p/np.log(2)*p1/p0-p*np.log(o)/np.log(2)+b
    return scorecard


if __name__ == '__main__':

    lr = read_model('lr.pkl')
    scorecard = get_scorecard(lr)
    print scorecard
    X = to_woe(user_data_dict, woe_dict, feature_list)
    score = scorecard(X)
    print score
