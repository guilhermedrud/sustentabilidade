import pandas as pd
import numpy as np
import sys
import os
import random

from statsmodels import robust
from scipy import stats

from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, precision_score, accuracy_score

from sklearn import ensemble
import lightgbm as lgb


# Parameters
LABEL_COLUMN_NAME = '7444'
UNWANTED_COLUMNS = ['empresa']

DEFAULT_LGB_PARAMS = {
    "max_bin": 512,
    "learning_rate": 0.05,
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 50,
    "verbose": -1,
   # "min_data": 5,
    "boost_from_average": True,
    "random_state": 1
}

N_FOLDS = 5
RANDOM_STATE = 1

def eval_bootstrap(df, features, md):
    X = df[features].values
    y = df[LABEL_COLUMN_NAME].values

    aa = []
    bb = []
#     cc = []
#     dd = []
    for i in range(1,5):
        a = []
     #   b = []
#         c = []
#         d = []
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=i)
        for (train, val) in cv.split(X, y):
            classifier = lgb.LGBMClassifier(**DEFAULT_LGB_PARAMS)

            classifier = classifier.fit(X[train], y[train])
            probas_ = classifier.predict_proba(X[val])
            
            auc = roc_auc_score(y[val], probas_[:, 1])
            #pred_test = classifier.predict(X[val]) #making predictions for test data
            #pred_train = classifier.predict(X[train]) #making predictions for train data
            #ppv = precision_score(y[val], pred_test,average='macro') #PPV is also the precision of the positive class

            a.insert(len(a), auc)
            #b.insert(len(b), ppv)
#             c.insert(len(c), r2)
#             d.insert(len(d), kappa)

        aa.append(np.mean(a))
        #bb.append(np.mean(b))
#         cc.append(np.mean(c))
#         dd.append(np.mean(d))
    return np.mean(aa)#,np.mean(bb)#,np.mean(dd)

def back_one(df, f, md):
    v = 0
    f1 = []
    f2 = []
    for i in f:
        f1.insert(len(f1), i)
        f2.insert(len(f2), i)
    A = eval_bootstrap(df, f1, md)
    z = A
    for i in f:
        f1.remove(i)
        A = eval_bootstrap(df, f1, md)
        print("%s,%f" % (f1,A))
        if A > z:
            v = 1
            z = A
            f2 = []
            for j in f1:
                f2.insert(len(f2), j)
        f1.insert(len(f1), i)
    return v,f2

# Reads dataset
df = pd.read_csv(sys.argv[1])
df.dropna(axis=0, subset=[LABEL_COLUMN_NAME], inplace=True)

RANDOM_STATE = 1
all_features = list(df.columns)

f = []
for x in UNWANTED_COLUMNS:
    if x in all_features: f.insert(len(f), x)
for x in f + [LABEL_COLUMN_NAME]:
    all_features.remove(x)

md = int(sys.argv[2])
f = []
i = 0

for f1 in all_features:
    if i == 50: break
    if f1 in f: continue
    k = -1
    x = f1
    i = i + 1
    j = 0
    for f2 in all_features:
        if f2 in f:
            continue
        j = j + 1
        f.insert(len(f), f2)
        A = eval_bootstrap(df, f, md)
        print("%s,%f" % (f,A))
        z = A
        f.remove(f2)
        sys.stdout.flush()
        if z > k:
            x = f2
            k = z
    f.insert(len(f), x)
    if i > 2:
        v,f = back_one(df, f, md)
        while v == 1:
            v,f = back_one(df, f, md)
        i = len(f)
