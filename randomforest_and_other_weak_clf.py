#-*-coding: utf-8-*-

from sklearn.ensemble import AdaBoostClassifier as Ada
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegressionCV as LR
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn import svm
import pandas as pd
import numpy as np
import pickle


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



#模型选取的特征
select_feature_list = ['call_count_per_day', 'phone_loan_times_per_platform',\
                'idcard_loan_platform_num', 'idcard_loan_times_per_platform',\
                'call_count', 'sustained_days', 'gender']


#创建学习模型
rf = RF(n_estimators = 40)
ada_tree = Ada(n_estimators = 40)
lr = LR()
nb1 = MultinomialNB()
nb2 = GaussianNB()
s_v_m = svm.SVC(C = 1)

ada_lr = Ada(base_estimator = LR(),n_estimators = 40,algorithm='SAMME')
ada_nb2 = Ada(base_estimator = GaussianNB(),n_estimators = 40,algorithm='SAMME')
ada_svm = Ada(base_estimator = svm.SVC(),n_estimators = 40,algorithm='SAMME')

#返回模型准确率
def model_estimate(clf, X, y, select_feature_list):
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
        #模型训练
        clf.fit(X_build[select_feature_list], y_build)
        #各项指标计算
        result.append(clf.score(X_test[select_feature_list], y_test))
    result = np.array(result)
    #返回的顺序依次为：准确率，P/R的AUC，ROC的AUC
    end = zip(['score'], [result.mean()])
    return end



#返回组合投票分类器模型准确率
def model_estimate_many(X, y, select_feature_list, *clf):
    r = []
    for i in range(10):
        num = int(len(y)*0.7)
        random_index = np.random.permutation(len(y))
        build_index = random_index[:num]
        test_index = random_index[num:]
        X_build = X.iloc[build_index].copy()
        y_build = y.iloc[build_index]
        X_test = X.iloc[test_index].copy()
        y_test = y.iloc[test_index]
        #模型训练
        result = []
        for i in clf:
            i.fit(X_build[select_feature_list], y_build)
            result.append(i.predict(X_test[select_feature_list]))
        result = np.array(result)
        result = result.sum(axis = 0)
        result[result <= 1] = 0
        result[result >= 2] = 1
        #各项指标计算
        r.append(np.sum(result == y_test)/len(result))
    result = np.array(result)
    #返回的顺序依次为：准确率，P/R的AUC，ROC的AUC
    end = zip(['score'], [result.mean()])
    return end



#将模型的实例序列化存储
def store_model(clf):
    a = open('rf.pkl', 'wb')
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
    good_bad_data = good_bad_data.fillna(-99)
    '''
    pass_data = get_merge_data('data/f_pass')
    reject_data = get_merge_data('data/f_reject')
    pass_data['y'] = 1
    reject_data['y'] = 0
    #得到整个训练数据
    pass_reject_data = pd.concat([pass_data, reject_data])
    pass_reject_data['pivot_table'] = 1
    '''

    X = good_bad_data 
    X.ix[X['gender'] == 'Female','gender'] = 0
    X.ix[X['gender'] == 'Male','gender'] = 1
    X.ix[X['gender'] == 'gender_none','gender']=2
    y = good_bad_data['y']
    '''
    #简单的组合模型效果
    result = model_estimate_many(X, y, select_feature_list,rf,s_v_m,nb2,lr)
    print result
    '''

    #单一模型效果
    result = model_estimate(rf, X, y, select_feature_list)
    print 'rf: ',result
    result = model_estimate(ada_tree, X, y, select_feature_list)
    print 'ada_tree: ',result
    result = model_estimate(ada_lr, X, y, select_feature_list)
    print 'ada_lr: ',result
    result = model_estimate(ada_nb2, X, y, select_feature_list)
    print 'ada_nb2: ',result
    result = model_estimate(ada_svm, X, y, select_feature_list)
    print 'ada_svm: ',result

    

    '''
    result = model_estimate(lr, X, y, select_feature_list)
    result = model_estimate(nb2, X, y, select_feature_list)
    result = model_estimate(s_v_m, X, y, select_feature_list)
    '''

    '''
    #通过评估后用全部数据训练，并保存训练好的模型
    rf.fit(X[select_feature_list], y)
    #store_model(rf)
    '''
        

