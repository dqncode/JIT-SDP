#SMOTE
import numpy as np
from imblearn.over_sampling import SMOTE

def smote(X, y, balanced_ratio,k):
    n_pos = int(np.sum(y==0)*balanced_ratio)
    model = SMOTE(sampling_strategy={1:n_pos},k_neighbors=k)
    print(n_pos)
    n_X, n_y = model.fit_resample(X, y)
    print('before: n_pos = %d, n_neg = %d'%(np.sum(y==1), np.sum(y==0)))
    print('after: n_pos = %d, n_neg = %d'%(np.sum(n_y==1), np.sum(n_y==0)))
    return n_X, n_y

# n_trn_X, n_trn_y = smote(trn_X, trn_y, balanced_ratio=0.25)