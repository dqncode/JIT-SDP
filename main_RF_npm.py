# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from statistics import harmonic_mean
import struct
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from scipy.io import arff

from EasyEn import EasyEnsemble
from tomekl_rm import tomek_link_rm
from cSmotet import combine_smote_tomek
from rus import random_undersampling
from nMiss import near_miss
from rom import random_oversampling
from smo import smote
from bSmote import borderline_smote
from cEnn import combine_enn
# from EasyEn import EasyEnsemble
# from bs import BalanceCascade
from cnn import condensed_nn
from ednn import edited_nn
from rank_e import rankmeasure_e
from rank_c import rankmeasure_c
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import tree
from sklearn.metrics import roc_auc_score

datasets=["npm"]
method = "RF"
for j in range(1):
    dataset=datasets[j]

#读取arff数据
    with open("./datasets/"+dataset+".arff", encoding="utf-8") as f:
        header = []
        for line in f:
            if line.startswith("@attribute"):
                header.append(line.split()[1])
            elif line.startswith("@data"):
                break
        data = pd.read_csv(f, header=None)
        data.columns = header
    data

    data = data.fillna(0)
    #布尔类型转01
    for u in data.columns:
        # nan转0,object强制转换bool
        if data[u].dtype == object:
            data[u] = data[u].astype(bool)

        if data[u].dtype == bool :
            data[u] = data[u].astype('int')

    #contains_bug列 object转int

    X = data[['fix', 'ns', 'nd', 'nf', 'entrophy', 'la', 'ld', 'lt', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp']]
    y = data[['contains_bug']]

    # 划分数据集
    def divideddata(X,y):
        # 按照7：3划分训练集和数据集
        trn_Xq, tst_Xq, trn_yq, tst_yq = train_test_split(X, y, test_size=0.5)
        #移除ND，REXP，LA,LD度量
        trn_X = trn_Xq[['fix', 'ns',  'nf', 'entrophy',  'lt', 'ndev', 'age', 'nuc', 'exp', 'sexp']]
        tst_X = tst_Xq[['fix', 'ns', 'nf', 'entrophy', 'lt', 'ndev', 'age', 'nuc', 'exp', 'sexp']]
        trn_y = trn_yq
        tst_y = tst_yq
        # effort = tst_X[:, 7]  la+ld
        effort= tst_Xq['la']+tst_Xq['ld']
        # 由dataframe转换成矩阵
        trn_X = np.array(trn_X)
        tst_X = np.array(tst_X)
        trn_y = np.array(trn_y)
        tst_y = np.array(tst_y)
        # # y应该是一维数组格式
        trn_y = trn_y.reshape(-1)
        tst_y = tst_y.reshape(-1)
        # zscore归一化处理
        trn_X = preprocessing.scale(trn_X)
        tst_X = preprocessing.scale(tst_X)
        # trn_X=np.log(trn_X+1)
        # tst_X=np.log(tst_X+1)
        return trn_X, tst_X, trn_y, tst_y, effort


    # 传统评估方法
    def evaluate(y_true, y_pred):
        # pre<0.5  =0；  pre>=0.5  =1；
        for i in range(len(y_pred)):
            if y_pred[i] < 0.5:
                y_pred[i] = 0
            else:
                y_pred[i] = 1
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        F1 = 2 * recall * precision / (recall + precision)
        Pf = fp / (fp + tn)
        G1 = 2 * recall * (1 - Pf) / (recall + (1 - Pf))
        AUC = roc_auc_score(y_true, y_pred)
        MCC = np.array([tp + fn, tp + fp, fn + tn, fp + tn]).prod()
        MCC = (tp * tn - fn * fp) / np.sqrt(MCC)
        return recall, precision, accuracy, F1, Pf, G1, AUC,MCC

    #随机森林
    def rf_predict(trn_X, trn_y, tst_X, tst_y, effort, class_weight=None):
        sc = StandardScaler()
        trn_X = sc.fit_transform(trn_X)
        tst_X = sc.transform(tst_X)
        # 训练随机森林解决回归问题
        clf = RandomForestClassifier(n_estimators=200, random_state=1)
        #n_estimators表示数的个数，数值越大越好，但是会占用内存多
        clf.fit(trn_X, trn_y)
        tst_pred = clf.predict(tst_X)
        Erecall, Eprecision, Efmeasure, ePMI, eIFA, ePopt = rankmeasure_e(tst_pred, effort, tst_y)
        cErecall, cEprecision, cEfmeasure, cPMI, cIFA, cPopt = rankmeasure_c(tst_pred, effort, tst_y)
        recall, precision, accuracy, F1, Pf, G1, AUC, MCC = evaluate(tst_y, tst_pred)
        return Erecall, Eprecision, Efmeasure, ePMI, eIFA, ePopt, cErecall, cEprecision, cEfmeasure, cPMI, cIFA, cPopt, recall, precision, accuracy, F1, Pf, G1, AUC, MCC




    cnn = np.zeros(shape=(50, 20))

    for i in range(50):
        trn_X, tst_X, trn_y, tst_y, effort = divideddata(X,y)
        n_X, n_y = condensed_nn(trn_X, trn_y)
        cnn[i, :] = rf_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        print("condensed_nn is okay~")

    #存储

    cnn = pd.DataFrame(cnn)
    cnn.columns = ['Erecall', 'Eprecision', 'Efmeasure', 'ePMI', 'eIFA', 'ePopt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA', 'cPopt', 'recall', 'precision', 'accuracy', 'F1', 'Pf', 'G1', 'AUC', 'MCC']
    cnnoutpath = './output/'+method+'-'+dataset+'-cnn' + '.csv'
    cnn.to_csv(cnnoutpath, index=True, header=True)


    print("running is okay~")

    # measure.columns = ['recall', 'precision','accuracy','F1','G1']
    # 看csv的文件
    # import pandas as pd
    # data = pd.read_csv(r'D:/software_defect-caiyang/LRmeasure2.csv',sep=',',header='infer')
    # # #看npy的文件
    # import numpy as np
    # cumXs = np.load(file="D:/software_defect-yaner/measure.npy",allow_pickle=True)



