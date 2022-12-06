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
from tomekl_rm import tomek_link_rm
from cSmotet import combine_smote_tomek
from rus import random_undersampling
from nMiss import near_miss
from rom import random_oversampling
from smo import smote
from bSmote import borderline_smote
from cEnn import combine_enn
from cnn import condensed_nn
from ednn import edited_nn
from rank_e import rankmeasure_e
from rank_c import rankmeasure_c
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import roc_auc_score

datasets=["fabric","jgroups","camel","tomcat","brackets","neutron","spring-integration","broadleaf","nova","npm"]
method = "NB"
for j in range(2):
    dataset=datasets[j]
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
    #bool to 01
    for u in data.columns:
        # nan to 0, object to bool
        if data[u].dtype == object:
            data[u] = data[u].astype(bool)

        if data[u].dtype == bool :
            data[u] = data[u].astype('int')

    # object to int

    X = data[['fix', 'ns', 'nd', 'nf', 'entrophy', 'la', 'ld', 'lt', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp']]
    y = data[['contains_bug']]
    def odd_divided(data):
        if len(data) % 2 != 0:
            data = data.iloc[0:len(data) - 1]
            X = data[['fix', 'ns', 'nd', 'nf', 'entrophy', 'la', 'ld', 'lt', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp']]
            y = data[['contains_bug']]
        trn_X, tst_X, trn_y, tst_y, effort = divideddata(X,y)
        return trn_X, tst_X, trn_y, tst_y, effort

    # divide data
    def divideddata(X,y):
        trn_Xq, tst_Xq, trn_yq, tst_yq = train_test_split(X, y, test_size=0.5)
        #remove ND，REXP，LA,LD度量
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


    def evaluate_all(tst_pred,effort,tst_y):
        Erecall, Eprecision, Efmeasure, ePMI, eIFA, ePopt = rankmeasure_e(tst_pred, effort, tst_y)
        cErecall, cEprecision, cEfmeasure, cPMI, cIFA, cPopt = rankmeasure_c(tst_pred, effort, tst_y)
        recall, precision, accuracy, F1, Pf, G1, AUC, MCC = evaluate(tst_y, tst_pred)
        return Erecall, Eprecision, Efmeasure, ePMI, eIFA, ePopt, cErecall, cEprecision, cEfmeasure, cPMI, cIFA, cPopt, recall, precision, accuracy, F1, Pf, G1, AUC, MCC


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

    #逻辑回归分类器
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    def lr_predict(trn_X, trn_y, tst_X, tst_y, effort, class_weight=None):
        modelLR = LogisticRegression(max_iter=2000)
        modelLR.fit(trn_X, trn_y)
        tst_pred = modelLR.predict_proba(tst_X)
        Erecall, Eprecision, Efmeasure, ePMI, eIFA,ePopt = rankmeasure_e( tst_pred[:, 1], effort,tst_y)
        cErecall, cEprecision, cEfmeasure,cPMI, cIFA,cPopt = rankmeasure_c(tst_pred[:, 1], effort, tst_y)
        recall, precision, accuracy, F1, Pf, G1, AUC,MCC = evaluate(tst_y, tst_pred[:, 1])
        return Erecall, Eprecision, Efmeasure,  ePMI, eIFA,ePopt,cErecall, cEprecision, cEfmeasure, cPMI, cIFA,cPopt, recall, precision, accuracy, F1, Pf, G1, AUC,MCC
    #决策树
    #trn_X 训练数据，trn_y 训练标签，tst_X 测试数据，tst_y测试标签
    def dt_predict(trn_X, trn_y, tst_X, tst_y, effort, class_weight=None):

        vec = DictVectorizer(sparse=False)
        trn_X =  pd.DataFrame(trn_X)
        trn_y =  pd.DataFrame(trn_y)
        tst_X =  pd.DataFrame(tst_X)
        X_train = vec.fit_transform(trn_X.to_dict(orient='record'))
        Y_train = vec.fit_transform(trn_y.to_dict(orient='record'))
        clf = tree.DecisionTreeClassifier(criterion='gini')
        model = clf.fit(X_train, Y_train.astype('float'))
        tst_pred = model.predict(tst_X)

        Erecall, Eprecision, Efmeasure, ePMI, eIFA, ePopt = rankmeasure_e( tst_pred, effort,tst_y)
        cErecall, cEprecision, cEfmeasure, cPMI, cIFA, cPopt= rankmeasure_c(tst_pred, effort, tst_y)
        recall, precision, accuracy, F1, Pf, G1, AUC,MCC = evaluate(tst_y, tst_pred)
        return Erecall, Eprecision, Efmeasure, ePMI, eIFA, ePopt, cErecall, cEprecision, cEfmeasure, cPMI, cIFA, cPopt, recall, precision, accuracy, F1, Pf, G1, AUC, MCC

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


    # 高斯贝叶斯
    def nb_predict(trn_X, trn_y, tst_X, tst_y, effort, class_weight=None):
        clf = GaussianNB()
        clf.fit(trn_X, trn_y)
        tst_pred = clf.predict(tst_X)
        Erecall, Eprecision, Efmeasure, ePMI, eIFA, ePopt = rankmeasure_e(tst_pred, effort, tst_y)
        cErecall, cEprecision, cEfmeasure, cPMI, cIFA, cPopt = rankmeasure_c(tst_pred, effort, tst_y)
        recall, precision, accuracy, F1, Pf, G1, AUC, MCC = evaluate(tst_y, tst_pred)
        return Erecall, Eprecision, Efmeasure, ePMI, eIFA, ePopt, cErecall, cEprecision, cEfmeasure, cPMI, cIFA, cPopt, recall, precision, accuracy, F1, Pf, G1, AUC, MCC


    none = np.zeros(shape=(50, 20))
    rum = np.zeros(shape=(50, 20))
    nm = np.zeros(shape=(50, 20))
    enn = np.zeros(shape=(50, 20))
    tlr = np.zeros(shape=(50, 20))
    rom = np.zeros(shape=(50, 20))
    cnn = np.zeros(shape=(50, 20))
    smo = np.zeros(shape=(50, 20))
    bsmote = np.zeros(shape=(50, 20))
    csmote = np.zeros(shape=(50, 20))
    cenn = np.zeros(shape=(50, 20))
    for i in range(1):
        trn_X, tst_X, trn_y, tst_y, effort = divideddata(X,y)
        none[i, :] = lr_predict(trn_X, trn_y, tst_X, tst_y, effort, class_weight=None)
        print("None is okay~")
        n_X, n_y = random_undersampling(trn_X, trn_y, balanced_ratio=1)
        rum[i, :] = lr_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        print("random_undersampling is okay~")
        n_X, n_y = near_miss(trn_X, trn_y, k=3, balanced_ratio=1)
        nm[i, :] = lr_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        print("near_miss is okay~")
        n_X, n_y = edited_nn(trn_X, trn_y, k=15)
        enn[i, :] = lr_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        print("edited_nn is okay~")
        n_X, n_y = tomek_link_rm(X=trn_X, y=trn_y)
        tlr[i, :] = lr_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        print("tomek_link_rm is okay~")
        n_X, n_y = random_oversampling(X=trn_X, y=trn_y, balanced_ratio=1)
        rom[i, :] = lr_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        print("random_oversampling is okay~")
        n_X, n_y = smote(trn_X, trn_y, balanced_ratio=1,k=5)
        smo[i, :] = lr_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        print("smote is okay~")
        n_X, n_y = borderline_smote(trn_X, trn_y, balanced_ratio=1)
        bsmote[i, :] = lr_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        print("borderline_smote is okay~")
        n_X, n_y = combine_smote_tomek(trn_X, trn_y, balanced_ratio=1)
        csmote[i, :] = lr_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        print("combine_smote_tomek is okay~")
        n_X, n_y = condensed_nn(trn_X, trn_y)
        cnn[i, :] = nb_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        print("condensed_nn is okay~")
        n_X, n_y = combine_enn(trn_X, trn_y, k=7, balanced_ratio=1)
        cenn[i, :] = lr_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        print("combine_enn is okay~")
        print("BalanceCascade is okay~")
    #存储
    none = pd.DataFrame(none)
    none.columns = ['Erecall', 'Eprecision', 'Efmeasure', 'ePMI', 'eIFA', 'ePopt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA', 'cPopt', 'recall', 'precision', 'accuracy', 'F1', 'Pf', 'G1', 'AUC', 'MCC']
    noneoutpath = './output/'+method+'-'+dataset+'-none.csv'
    none.to_csv(noneoutpath, index=True, header=True)
    none = pd.DataFrame(none)
    none.columns = ['Erecall', 'Eprecision', 'Efmeasure', 'ePMI', 'eIFA', 'ePopt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA', 'cPopt', 'recall', 'precision', 'accuracy', 'F1', 'Pf', 'G1', 'AUC', 'MCC']
    noneoutpath = './output/'+'-none.csv'
    none.to_csv(noneoutpath, index=True, header=True)

    rum = pd.DataFrame(rum)
    rum.columns = ['Erecall', 'Eprecision', 'Efmeasure', 'ePMI', 'eIFA', 'ePopt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA', 'cPopt', 'recall', 'precision', 'accuracy', 'F1', 'Pf', 'G1', 'AUC', 'MCC']
    rumoutpath = './output/'+method+'-'+dataset+'-rum' + '.csv'
    rum.to_csv(rumoutpath, index=True, header=True)

    nm = pd.DataFrame(nm)
    nm.columns = ['Erecall', 'Eprecision', 'Efmeasure', 'ePMI', 'eIFA', 'ePopt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA', 'cPopt', 'recall', 'precision', 'accuracy', 'F1', 'Pf', 'G1', 'AUC', 'MCC']
    nmoutpath ='./output/'+method+'-'+dataset+'-nm' + '.csv'
    nm.to_csv(nmoutpath, index=True, header=True)

    enn = pd.DataFrame(enn)
    enn.columns = ['Erecall', 'Eprecision', 'Efmeasure', 'ePMI', 'eIFA', 'ePopt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA', 'cPopt', 'recall', 'precision', 'accuracy', 'F1', 'Pf', 'G1', 'AUC', 'MCC']
    ennoutpath = './output/'+method+'-'+dataset+'-enn' + '.csv'
    enn.to_csv(ennoutpath, index=True, header=True)

    tlr = pd.DataFrame(tlr)
    tlr.columns = ['Erecall', 'Eprecision', 'Efmeasure', 'ePMI', 'eIFA', 'ePopt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA', 'cPopt', 'recall', 'precision', 'accuracy', 'F1', 'Pf', 'G1', 'AUC', 'MCC']
    tlroutpath = './output/'+method+'-'+dataset+'-tlr' + '.csv'
    tlr.to_csv(tlroutpath, index=True, header=True)

    rom = pd.DataFrame(rom)
    rom.columns = ['Erecall', 'Eprecision', 'Efmeasure', 'ePMI', 'eIFA', 'ePopt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA', 'cPopt', 'recall', 'precision', 'accuracy', 'F1', 'Pf', 'G1', 'AUC', 'MCC']
    romoutpath = './output/'+method+'-'+dataset+'-rom' + '.csv'
    rom.to_csv(romoutpath, index=True, header=True)

    smo = pd.DataFrame(smo)
    smo.columns = ['Erecall', 'Eprecision', 'Efmeasure', 'ePMI', 'eIFA', 'ePopt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA', 'cPopt', 'recall', 'precision', 'accuracy', 'F1', 'Pf', 'G1', 'AUC', 'MCC']
    smooutpath = './output/'+method+'-'+dataset+'-smo' + '.csv'
    smo.to_csv(smooutpath, index=True, header=True)

    bsmote = pd.DataFrame(bsmote)
    bsmote.columns = ['Erecall', 'Eprecision', 'Efmeasure', 'ePMI', 'eIFA', 'ePopt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA', 'cPopt', 'recall', 'precision', 'accuracy', 'F1', 'Pf', 'G1', 'AUC', 'MCC']
    bsmoteoutpath = './output/'+method+'-'+dataset+'-bsmote' + '.csv'
    bsmote.to_csv(bsmoteoutpath, index=True, header=True)

    csmote = pd.DataFrame(csmote)
    csmote.columns = ['Erecall', 'Eprecision', 'Efmeasure', 'ePMI', 'eIFA', 'ePopt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA', 'cPopt', 'recall', 'precision', 'accuracy', 'F1', 'Pf', 'G1', 'AUC', 'MCC']
    csmoteoutpath = './output/'+method+'-'+dataset+'-csmote' + '.csv'
    csmote.to_csv(csmoteoutpath, index=True, header=True)

    cnn = pd.DataFrame(cnn)
    cnn.columns = ['Erecall', 'Eprecision', 'Efmeasure', 'ePMI', 'eIFA', 'ePopt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA', 'cPopt', 'recall', 'precision', 'accuracy', 'F1', 'Pf', 'G1', 'AUC', 'MCC']
    cnnoutpath = './output/'+method+'-'+dataset+'-cnn' + '.csv'
    cnn.to_csv(cnnoutpath, index=True, header=True)

    cenn = pd.DataFrame(cenn)
    cenn.columns = ['Erecall', 'Eprecision', 'Efmeasure', 'ePMI', 'eIFA', 'ePopt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA', 'cPopt', 'recall', 'precision', 'accuracy', 'F1', 'Pf', 'G1', 'AUC', 'MCC']
    cennoutpath = './output/'+method+'-'+dataset+'-cenn' + '.csv'
    cenn.to_csv(cennoutpath, index=True, header=True)


    print("running is okay~")



