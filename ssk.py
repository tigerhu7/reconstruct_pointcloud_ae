__author__ = 'sskqgfnnh'

import torch
import random
import numpy as np

# def test_loss_fun(x, y):
#     z = x-y
#     z = z * z
#     z = np.sum(z, axis=1)
#     z = np.sqrt(z)
#     return z.mean()


def ColorSample(num):
    if num > 256 ** 3:
        print('color size is more than 256^3')
        return
    color_sample = np.array(random.sample(range(256 ** 3), num))
    b = color_sample % 256
    color_sample = color_sample // 256
    g = color_sample % 256
    color_sample = color_sample // 256
    r = color_sample % 256
    return np.hstack((r.reshape(len(r), -1), g.reshape(len(g), -1), b.reshape(len(b), -1)))

def PlotData(source_data, neural_data, plot_type=0, savename = './test.txt'):
    print("PlotCube: source_data shape = {}".format(source_data.shape))
    print("PlotCube: neural_data shape = {}".format(neural_data.shape))
    numX = source_data.shape[0]
    
    # 分为 encoding 和 decoding， 各自的neuron 的个数分别有 num_col 个~~~~~
    num_col = neural_data.shape[1] // 2   
    d = {}
    for i in range(numX):  # 对所有样本点， 进行遍历
        if plot_type == 0:
            t = tuple(neural_data[i, 0: num_col])   # 把 encoder 的 神经元取出来
        if plot_type == 1:
            t = tuple(neural_data[i, num_col + 1:])  # 把 decoder 的神经元 取出来
        if plot_type == 2:
            t = tuple(neural_data[i, :])  # 把所有的神经元 取出来~~
        

        if t in d:  # 就是说， 便利到某一个样本点时， 所激活的神经元与之前的某一个是一模一样的话， 那么就给这个组合+1
            d[t] = d[t] + 1
        else: # 否则， 就置为0
            d.setdefault(t, 0)

    # d 有多少个 激活的神经元的组合 
    # 每一个 d 给出一个编号count 和一个颜色color_smaple
    dict_len = len(d)
    print("PlotCube: dict_len = {}, i.e. How many combinations of activated neurons ".format(dict_len))
    count = 0
    color_sample = ColorSample(dict_len)
    for k, v in d.items():
        d[k] = np.hstack((count, color_sample[count, :])) # 然后在 d 里面， 放 count编号 和 颜色～～～
        count = count + 1


    # cdata 就是每一个样本， 给它对应的d
    # （因为两个不同样本， 可能会有相同的激活射神经元的组合， 那么这个时候， 这两个样本就赋予了相同的d）
    cdata = np.zeros(shape=[numX, 4], dtype=int)
    for i in range(numX):
        if plot_type == 0:
            t = tuple(neural_data[i, 0: num_col])
        if plot_type == 1:
            t = tuple(neural_data[i, num_col + 1:])
        if plot_type == 2:
            t = tuple(neural_data[i, :])
        cdata[i, :] = d[t]
    print('color_data shape = {}'.format(cdata.shape))



    # 把
    #fwriter = open("colored_data.txt", "w")
    fwriter = open( savename, "w")
    for i in range(numX):
        my_str = ""
        for j in range(source_data.shape[1]):
            my_str = my_str + str(source_data[i, j]) + " "
        for k in range(cdata.shape[1]):
            my_str = my_str + str(cdata[i, k]) + " "
        my_str = my_str + "\n"
        fwriter.write(my_str)
    fwriter.close()
    return source_data


