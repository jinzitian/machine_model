#-*-coding: utf-8-*-

from sklearn.ensemble import AdaBoostClassifier as Ada
from sklearn.linear_model import LogisticRegressionCV as LR
from sklearn.metrics import precision_recall_curve, auc, roc_curve, f1_score
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import tree
#from sklearn.naive_bayes import MultinomialNB,GaussianNB
#from sklearn import svm
import pandas as pd
import numpy as np
import pickle

from feature_selection.get_average_interval import get_average_interval
from feature_selection.add_woe_dynamic import add_woe_dynamic

#此函数仅仅适用table类型的表格
def get_merge_data(*args):
    datalist=[]
    for filename in args:
        datalist.append(pd.read_table(filename))
    return pd.concat(datalist)


#特征列表
all_feature = [
                        'call_count',
                        'call_time',
                        'sustained_days',
                        'call_count_per_day',
                        'baidu_home_distance',
                        'baidu_work_distance',
                        'getui_home_distance',
                        'getui_work_distance',
                        'phone_loan_platform_num',
                        'phone_loan_times',
                        'phone_loan_times_per_platform',
                        'idcard_loan_platform_num',
                        'idcard_loan_times',
                        'idcard_loan_times_per_platform'
               ]


#需要得到区间的特征
to_bins_feature_list = ['call_count_per_day', 'phone_loan_times_per_platform',\
                'idcard_loan_platform_num', 'idcard_loan_times_per_platform', 'call_count',\
                'call_time', 'sustained_days']

#模型选取的特征
select_feature_list = ['call_count_per_day', 'phone_loan_times_per_platform',\
                'idcard_loan_platform_num', 'idcard_loan_times_per_platform',\
                'call_count', 'sustained_days', 'gender']


#定义生成特征区间字典的函数
def get_feature_bins_dict(dataframe, feature_list, nan_value = 'fuck'):
    feature_bins_dict = {}
    for i in feature_list:
        temp_1 = dataframe[i].astype(object)
        temp_2 = temp_1[temp_1 != nan_value]
        bins = get_average_interval(temp_2, 10)
        feature_bins_dict[i] = bins
    return feature_bins_dict

#得到特征离散化后的各区间频数
def _fetch_feature_discrete_frequency(dataframe, feature_name, bins = None, nan_value = 'fuck'):
    if bins == None:
        result = dataframe.pivot_table('pivot_table', index = feature_name, columns = ['y'], aggfunc = 'count')
        result = result.fillna(0)
    else:
        data = dataframe[feature_name].astype(object)
        data[data != nan_value] = data[data != nan_value].map(float)
        #将特征以区间形式进行切分（其他形式的切分需要单独做）
        cut = pd.cut(data, bins, right = False)
        index = cut.replace(np.nan, nan_value)
        #将特征列替换为切分后的形式
        dataframe[feature_name] = index 
        result = dataframe.pivot_table('pivot_table', index = index, columns = ['y'], aggfunc = 'count')
        result = result.fillna(0)
    return result

#动态计算并保存特征woe字典的函数
def compute_and_save_feature_woe(dataframe, feature_name, bins=None):
    a = _fetch_feature_discrete_frequency(dataframe, feature_name, bins)
    series_1 = a[1]
    series_0 = a[0]
    add_woe_dynamic(dataframe, series_1, series_0, feature_name)    

#获取woe字典的函数
def get_feature_woe(feature_name):
    with open('data/%s_woe'%feature_name,'r') as a:
        woe = a.read()
    return eval(woe)

#将每一个特征值，映射到对应的woe值
def _one_to_woe(value, feature_name, woe_dict, nan_value = 'fuck'):
    #如果是离散型的，直接返回对应的woe
    index = woe_dict[feature_name].keys()
    if value in index:
        return woe_dict[feature_name][value]
    if value == '' or value == np.nan or value == nan_value:
        return woe_dict[feature_name].get(nan_value, 0)
    #如果为区间型的，需要判断区间位子，然后返回woe
    if isinstance(type(value)(), float) or isinstance(type(value)(), long) or isinstance(type(value)(), int):
        if str(index[0])[0] == '(' or str(index[0])[0] == '[':
            if nan_value in index:
                index.pop(index.index(nan_value))
            temp_1 = [list(map(float,i[1:-1].split(','))) for i in index]
            temp_2 = sorted(temp_1)
            for i in temp_2[:-1]:
                if value >= i[0] and value < i[1]:
                    return woe_dict[feature_name][index[temp_1.index(i)]]
            if value >= temp_2[-1][0]:
                return woe_dict[feature_name][index[temp_1.index(temp_2[-1])]]
    return 0

#将一组用户数据转换为woe值，用于模型接收
def to_woe(user_data_dict, woe_dict, feature_names, nan_value = 'fuck'):
    data_list = []
    for i in feature_names:
        data_list.append(_one_to_woe(user_data_dict[i], i, woe_dict, nan_value))
    return data_list


#创建学习模型
lr = LR()
rf = RF()
ada_lr = Ada(base_estimator = LR(),algorithm='SAMME')

#clf1 = MultinomialNB()
#clf2 = GaussianNB()
#clf3 = svm.SVC(C = 10000)


#模型训练并生成woe字典
def model_fit(clf, X, y, feature_list, feature_bins_dict):
    #为训练数据添加特征woe值列，并保存woe字典
    for i in feature_list:
        compute_and_save_feature_woe(X, i, feature_bins_dict.get(i))
    #训练模型
    feature_names_woe = [i+'_woe' for i in feature_list]
    clf.fit(X[feature_names_woe], y)


#返回模型准确率，P/R的AUC，ROC的AUC
def model_estimate(clf, X, y, select_feature_list, to_bins_feature_list):
    result = []
    for i in range(10):
        num = int(len(y)*0.7)
        random_index = np.random.permutation(len(y))
        build_index = random_index[:num]
        test_index = random_index[num:]
        X_build = X.iloc[build_index].copy()
        y_build = y.iloc[build_index]
        X_test = X.iloc[test_index].copy()
        y_test = y.iloc[test_index]
        #修改特征区间字典
        feature_bins_dict = get_feature_bins_dict(X_build, to_bins_feature_list)
        #模型训练并生成woe字典
        model_fit(clf, X_build, y_build, select_feature_list, feature_bins_dict)
        #获取之前保存的woe字典，供测试数据转换使用
        woe_dict = {}
        for i in select_feature_list:
            woe_dict[i] = get_feature_woe(i)
        #获取转换为woe后的测试数据
        X_test_list = []
        for i in range(len(X_test)):
            X_test_list.append(to_woe(X_test.iloc[i], woe_dict, select_feature_list))
        X_test = np.array(X_test_list)
        #各项指标计算
        y_score = clf.predict_proba(X_test)
        fpr, tpr, t = roc_curve(y_test, y_score[:,1])
        p, r, t = precision_recall_curve(y_test, y_score[:,1])
        result.append([clf.score(X_test, y_test), auc(r, p), auc(fpr, tpr)])
    result = np.array(result)
    #返回的顺序依次为：准确率，P/R的AUC，ROC的AUC
    end = []
    end.append(('score',result.mean(axis = 0)[0]))
    end.append(('P/R_auc',result.mean(axis = 0)[1])) 
    end.append(('ROC_auc',result.mean(axis = 0)[2])) 
    return end

#得到评分卡模型
def get_scorecard(clf, b = 500 , o = 1, p = 20):
    def scorecard(X):
        p0, p1 = clf.predict_log_proba(X).reshape(1,-1)
        return p/np.log(2)*p1/p0-p*np.log(o)/np.log(2)+b
    return scorecard

#计算评分卡的KS指标
def compute_KS(bin_num, score_list, y_value):
    score = score_list[:]
    score.sort()
    cut = int(len(score)/bin_num)
    bins = [score[i*cut] for i in range(bin_num)]+[score[-1]+1]
    print bins
    score_True = [score_list[i] for i in range(len(y_value)) if y_value[i]==1]
    score_False = [score_list[i] for i in range(len(y_value)) if y_value[i]==0]
    new_index = ['[%s, %s)'%(bins[i],bins[i+1]) for i in range(len(bins)-1)]
    True_des = pd.value_counts(pd.cut(score_True, bins, right = False)).reindex(new_index)
    False_des = pd.value_counts(pd.cut(score_False, bins, right = False)).reindex(new_index)
    True_des = True_des.cumsum()/True_des.sum()
    False_des = False_des.cumsum()/False_des.sum()
    KS = (True_des - False_des).abs().max()
    return KS

#计算评分卡的PSI指标
def compute_PSI():
    pass

#将模型的实例序列化存储
def store_model(clf):
    a = open('lr.pkl', 'wb')
    pickle.dump(clf, a)
    a.close()


if __name__ == '__main__':

    #从table表格中获取数据，最终要的是dataframe
    good_data = get_merge_data('data/f_good','data/f_done')
    bad_data = get_merge_data('data/f_delay')
    good_data['y'] = 1
    bad_data['y'] = 0
    #得到整个训练数据
    good_bad_data = pd.concat([good_data[0:2000], bad_data[0:2000]])
    good_bad_data['pivot_table'] = 1
    good_bad_data = good_bad_data.fillna('fuck')
    '''
    pass_data = get_merge_data('data/f_pass')
    reject_data = get_merge_data('data/f_reject')
    pass_data['y'] = 1
    reject_data['y'] = 0
    #得到整个训练数据
    pass_reject_data = pd.concat([pass_data, reject_data])
    pass_reject_data['pivot_table'] = 1
    '''

    #模型评估
    X = good_bad_data 
    y = good_bad_data['y']
    result = model_estimate(lr, X, y, select_feature_list, to_bins_feature_list)
    print result

    
    #得到全样本下特征区间字典
    feature_bins_dict = get_feature_bins_dict(X, to_bins_feature_list)
    #通过评估后用全部数据训练，并保存训练好的模型
    model_fit(lr, X, y, select_feature_list, feature_bins_dict)
    store_model(lr)
       
        

