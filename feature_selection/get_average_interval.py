# -*-coding: utf-8 -*-


#定义数据切割函数(以数量均匀切割)
def get_average_interval(series, cut_num):
    a = list(series)
    a.sort()
    num = len(a)//cut_num
    bins = [a[0]]
    index = 0
    for i in range(cut_num-1):
        index += num
        bins.append(a[index])
    bins.append(a[-1]+1)
    final_bins = list(set(bins))
    final_bins.sort()
    return final_bins

