import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
TreeAndDv = pd.read_csv(u'./positive_sample.txt', header=None, low_memory=False, sep='\t')
TreeAndDv_1 = pd.read_csv(u'./negative_sample.txt', header=None, low_memory=False, sep='\t')
positive = []
negative = []
t = 0
while t < len(TreeAndDv_1[0]):
    b = []
    b.append(TreeAndDv_1[0][t])
    b.append(TreeAndDv_1[1][t])
    negative.append(b)
    t += 1
k = 0
while k < len(TreeAndDv[0]):
    b = []
    b.append(TreeAndDv[0][k])
    b.append(TreeAndDv[1][k])
    positive.append(b)
    k += 1

def get_vect(filename_1, filename_2, dis_num):
    if dis_num == 64:
        TreeAndDv_2 = pd.read_csv(filename_1, header=None, low_memory=False, sep=',')
        j = 0
        a = np.zeros((len(TreeAndDv_2[0]),64))
        while j < len(TreeAndDv_2[0]):
            t_0 = 0
            while t_0 < 64:
                a[j][t_0] = TreeAndDv_2[t_0][j]
                t_0 += 1
            j += 1
    else:
        TreeAndDv_2 = pd.read_csv(filename_1, header=None, low_memory=False, sep=',')
        j = 0
        a = np.zeros((len(TreeAndDv_2[0]),128))
        while j < len(TreeAndDv_2[0]):
            t_0 = 0
            while t_0 < 128:
                a[j][t_0] = TreeAndDv_2[t_0][j]
                t_0 += 1
            j += 1
    negative_vect = []
    for i in negative:
        vect = abs(a[i[0]] - a[i[1]])
        negative_vect.append(vect)
    positive_vect = []
    for i in positive:
        vect = abs(a[i[0]] - a[i[1]])
        positive_vect.append(vect)
    positive_vect = pd.DataFrame(positive_vect)
    negative_vect = pd.DataFrame(negative_vect)
    all_vect = np.vstack((positive_vect, negative_vect))
    all_vect = pd.DataFrame(all_vect)
    all_vect.to_csv(filename_2, sep=',', index=False, header=False)


get_vect("./nodevector/32+32.csv", './Absminus/32+32.csv', 64)


