# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#from statistics import harmonic_mean
#import struct
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from smo import smote
from rank_e import rankmeasure_e
from rank_c import rankmeasure_c
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import time

start = time.perf_counter()
datasets=["fabric","jgroups","camel","tomcat","brackets","neutron","spring-integration","broadleaf","nova","npm"]
method = "RF"
for j in range(10):
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
    #nan to 0,bool to 01
    for u in data.columns:
        if data[u].dtype == object:
            data[u] = data[u].astype(bool)

        if data[u].dtype == bool :
            data[u] = data[u].astype('int')


    X = data[['fix', 'ns', 'nd', 'nf', 'entrophy', 'la', 'ld', 'lt', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp']]
    y = data[['contains_bug']]

    def odd_divided(data):
        if len(data) % 2 != 0:
            data = data.iloc[0:len(data) - 1]
            X = data[['fix', 'ns', 'nd', 'nf', 'entrophy', 'la', 'ld', 'lt', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp']]
            y = data[['contains_bug']]
        trn_X, tst_X, trn_y, tst_y, effort = divideddata(X,y)
        return trn_X, tst_X, trn_y, tst_y, effort


    def divideddata(X,y):
        trn_Xq, tst_Xq, trn_yq, tst_yq = train_test_split(X, y, test_size=0.5)
        #remove ND，REXP，LA,LD
        trn_X = trn_Xq[['fix', 'ns',  'nf', 'entrophy',  'lt', 'ndev', 'age', 'nuc', 'exp', 'sexp']]
        tst_X = tst_Xq[['fix', 'ns', 'nf', 'entrophy', 'lt', 'ndev', 'age', 'nuc', 'exp', 'sexp']]
        trn_y = trn_yq
        tst_y = tst_yq
        effort= tst_Xq['la']+tst_Xq['ld']
        trn_X = np.array(trn_X)
        tst_X = np.array(tst_X)
        trn_y = np.array(trn_y)
        tst_y = np.array(tst_y)
        trn_y = trn_y.reshape(-1)
        tst_y = tst_y.reshape(-1)
        trn_X = preprocessing.scale(trn_X)
        tst_X = preprocessing.scale(tst_X)
        return trn_X, tst_X, trn_y, tst_y, effort


    def evaluate_all(tst_pred,effort,tst_y):
        Erecall, Eprecision, Efmeasure, ePMI, eIFA, ePopt = rankmeasure_e(tst_pred, effort, tst_y)
        cErecall, cEprecision, cEfmeasure, cPMI, cIFA, cPopt = rankmeasure_c(tst_pred, effort, tst_y)
        recall, precision, accuracy, F1, Pf, G1, AUC, MCC = evaluate(tst_y, tst_pred)
        return Erecall, Eprecision, Efmeasure, ePMI, eIFA, ePopt, cErecall, cEprecision, cEfmeasure, cPMI, cIFA, cPopt, recall, precision, accuracy, F1, Pf, G1, AUC, MCC


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


    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)

    def lr_smo_opt(X, y, class_weight=None):
        k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        bst_k = k_list[0]
        bst_m = 0
        results= np.zeros(20)
        for k in k_list:
            n_X, n_y = smote(X, y, balanced_ratio=1, k=k)
            kf = KFold(n_splits=5,shuffle=True)
            result = np.zeros(5)
            modelLR = LogisticRegression(max_iter=2000)
            i = 0
            for train_index, test_index in kf.split(n_X, n_y):

                trn_x, trn_y = n_X[train_index], n_y[train_index]
                tst_x, tst_y = n_X[test_index], n_y[test_index]
                modelLR.fit(trn_x, trn_y)
                tst_pred = modelLR.predict(tst_x)
                recall, precision, accuracy, F1, Pf, G1, AUC, MCC = evaluate(tst_y, tst_pred)
                result[i] = AUC
                i = i + 1
            if np.mean(result) > bst_m:
                bst_k = k
                bst_m = np.mean(result)
            results[k-1]=np.mean(result)
        return bst_k

    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    def lr_predict(trn_X, trn_y, tst_X, tst_y, effort, class_weight=None):
        modelLR = LogisticRegression(max_iter=2000)
        modelLR.fit(trn_X, trn_y)
        tst_pred = modelLR.predict_proba(tst_X)
        Erecall, Eprecision, Efmeasure, ePMI, eIFA, ePopt = rankmeasure_e(tst_pred[:, 1], effort, tst_y)
        cErecall, cEprecision, cEfmeasure, cPMI, cIFA, cPopt = rankmeasure_c(tst_pred[:, 1], effort, tst_y)
        recall, precision, accuracy, F1, Pf, G1, AUC, MCC = evaluate(tst_y, tst_pred[:, 1])
        return Erecall, Eprecision, Efmeasure, ePMI, eIFA, ePopt, cErecall, cEprecision, cEfmeasure, cPMI, cIFA, cPopt, recall, precision, accuracy, F1, Pf, G1, AUC, MCC


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

    def dt_smo_opt(X, y,  class_weight=None):
        cvdata = np.c_[X, y]
        kf = KFold(n_splits=5)
        for train, valid in kf.split(cvdata):
            trn_data = cvdata[train]
            trn_X = trn_data[:, 0:9]
            trn_y = trn_data[:, 10]
            tst_data = cvdata[valid]
            tst_X = tst_data[:, 0:9]
            tst_y = tst_data[:, 10]
        k_list = [1,2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        bst_k = k_list[0]
        bst_m = 0
        for k in k_list:
          n_X, n_y = smote(trn_X, trn_y, balanced_ratio=1, k=k)
          vec = DictVectorizer(sparse=False)
          trn_X = pd.DataFrame(n_X)
          trn_y = pd.DataFrame(n_y)
          tst_X = pd.DataFrame(tst_X)
          X_train = vec.fit_transform(trn_X.to_dict(orient='record'))
          Y_train = vec.fit_transform(trn_y.to_dict(orient='record'))
          clf = tree.DecisionTreeClassifier(criterion='gini')
          model = clf.fit(X_train, Y_train.astype('float'))
          tst_pred = model.predict(tst_X)
          recall, precision, accuracy, F1, Pf, G1, AUC,MCC = evaluate(tst_y, tst_pred[:, 1])
          if np.mean(AUC) > bst_m:
              bst_k = k
              bst_m = np.mean(AUC)
        return  bst_k

    def rf_predict(trn_X, trn_y, tst_X, tst_y, effort, class_weight=None):
        sc = StandardScaler()
        trn_X = sc.fit_transform(trn_X)
        tst_X = sc.transform(tst_X)

        clf = RandomForestClassifier(n_estimators=200, random_state=1)

        clf.fit(trn_X, trn_y)
        tst_pred = clf.predict(tst_X)
        Erecall, Eprecision, Efmeasure, ePMI, eIFA, ePopt = rankmeasure_e(tst_pred, effort, tst_y)
        cErecall, cEprecision, cEfmeasure, cPMI, cIFA, cPopt = rankmeasure_c(tst_pred, effort, tst_y)
        recall, precision, accuracy, F1, Pf, G1, AUC, MCC = evaluate(tst_y, tst_pred)
        return Erecall, Eprecision, Efmeasure, ePMI, eIFA, ePopt, cErecall, cEprecision, cEfmeasure, cPMI, cIFA, cPopt, recall, precision, accuracy, F1, Pf, G1, AUC, MCC


    def rf_smo_opt(X, y, class_weight=None):
        k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        bst_k = k_list[0]
        bst_m = 0
        results = np.zeros(20)
        for k in k_list:
            n_X, n_y = smote(X, y, balanced_ratio=1, k=k)
            kf = KFold(n_splits=5, shuffle=True)
            clf = RandomForestClassifier(n_estimators=200, random_state=1)
            i = 0
            result = np.zeros(5)
            for train_index, test_index in kf.split(n_X, n_y):
                trn_X, trn_y = n_X[train_index], n_y[train_index]
                tst_X, tst_y = n_X[test_index], n_y[test_index]

                sc = StandardScaler()
                trn_X = sc.fit_transform(trn_X)
                tst_X = sc.transform(tst_X)
                clf.fit(trn_X, trn_y)
                tst_pred = clf.predict(tst_X)
                recall, precision, accuracy, F1, Pf, G1, AUC, MCC = evaluate(tst_y, tst_pred)
                result[i] = AUC
                i = i + 1
            if np.mean(result) > bst_m:
                bst_k = k
                bst_m = np.mean(result)
            results[k-1]=np.mean(result)
        return bst_k,results


    def nb_predict(trn_X, trn_y, tst_X, tst_y, effort, class_weight=None):
        clf = GaussianNB()
        clf.fit(trn_X, trn_y)
        tst_pred = clf.predict(tst_X)
        Erecall, Eprecision, Efmeasure, ePMI, eIFA, ePopt = rankmeasure_e(tst_pred, effort, tst_y)
        cErecall, cEprecision, cEfmeasure, cPMI, cIFA, cPopt = rankmeasure_c(tst_pred, effort, tst_y)
        recall, precision, accuracy, F1, Pf, G1, AUC, MCC = evaluate(tst_y, tst_pred)
        return Erecall, Eprecision, Efmeasure, ePMI, eIFA, ePopt, cErecall, cEprecision, cEfmeasure, cPMI, cIFA, cPopt, recall, precision, accuracy, F1, Pf, G1, AUC, MCC

    def nb_smo_opt(X, y, class_weight=None):
        k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        bst_k = k_list[0]
        bst_m = 0
        for k in k_list:
            n_X, n_y = smote(X, y, balanced_ratio=1, k=k)
            kf = KFold(n_splits=5, shuffle=True)
            clf = GaussianNB()
            i = 0
            result = np.zeros(5)
            for train_index, test_index in kf.split(n_X, n_y):
                trn_X, trn_y = n_X[train_index], n_y[train_index]
                tst_X, tst_y = n_X[test_index], n_y[test_index]

                clf.fit(trn_X, trn_y)
                tst_pred = clf.predict(tst_X)
                recall, precision, accuracy, F1, Pf, G1, AUC, MCC = evaluate(tst_y, tst_pred)
                result[i] = AUC
                i = i + 1
            if np.mean(result) > bst_m:
                bst_k = k
                bst_m = np.mean(result)
        return bst_k

    optsmo = np.zeros(shape=(50, 20))
    for i in range(50):
        trn_X, tst_X, trn_y, tst_y, effort = divideddata(X,y)
        best_k,results=rf_smo_opt(trn_X, trn_y, class_weight=None)
        n_X, n_y = smote(trn_X, trn_y, balanced_ratio=1,k=best_k)
        optsmo[i, :] = rf_predict(n_X, n_y, tst_X, tst_y, effort, class_weight=None)
        print("optsmote is okay~")

    optsmo = pd.DataFrame(optsmo)
    optsmo.columns = ['Erecall', 'Eprecision', 'Efmeasure', 'ePMI', 'eIFA', 'ePopt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA', 'cPopt', 'recall', 'precision', 'accuracy', 'F1', 'Pf', 'G1', 'AUC', 'MCC']
    optsmooutpath = './output/'+method+'-'+dataset+'-optsmo' + '.csv'
    optsmo.to_csv(optsmooutpath, index=True, header=True)

    end = time.perf_counter()
    print("running time is :")
    print (end - start)
    print("running is okay~")




