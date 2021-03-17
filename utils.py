import numpy as np
from sklearn import metrics
from imblearn.over_sampling import *


def apply_adasyn(x_train, y_train):
    sm = ADASYN(sampling_strategy='minority')
    x_train_res, y_train_res = sm.fit_resample(x_train, y_train)
    return x_train_res, y_train_res


def apply_smote(x_train, y_train):
    sm = SMOTE(sampling_strategy='minority')
    x_train_res, y_train_res = sm.fit_resample(x_train, y_train)
    return x_train_res, y_train_res


def apply_b_smote(x_train, y_train):
    sm = BorderlineSMOTE(sampling_strategy='minority')
    x_train_res, y_train_res = sm.fit_resample(x_train, y_train)
    return x_train_res, y_train_res


def apply_km_smote(x_train, y_train):
    sm = KMeansSMOTE(sampling_strategy='minority')
    x_train_res, y_train_res = sm.fit_resample(x_train, y_train)
    return x_train_res, y_train_res


def apply_svm_smote(x_train, y_train):
    sm = KMeansSMOTE(sampling_strategy='minority')
    x_train_res, y_train_res = sm.fit_resample(x_train, y_train)
    return x_train_res, y_train_res


def load_dataset(dir_name, node_embedding):
    V = dict()
    Y = dict()
    phases = ['train', 'valid', 'test']
    for phase in phases:
        v_fname = dir_name + '/x_' + phase + '_4_300_' + node_embedding + '_pca.npy'
        y_fname = dir_name + '/y_' + phase + '_4_300_' + node_embedding + '_pca.npy'
        V[phase] = np.load(v_fname)
        V[phase] = V[phase].reshape(V[phase].shape[0], -1)
        Y[phase] = np.load(y_fname).reshape(-1, 1)
    return V, Y


def concat_train_valid(V, Y):
    x_train = np.concatenate((V['train'], V['valid']), axis=0)
    y_train = np.concatenate((Y['train'], Y['valid']), axis=0)
    return x_train, y_train


def print_metrics(preds, y_test):
    acc = metrics.accuracy_score(y_test, preds)
    roc_auc = metrics.roc_auc_score(y_test, preds)
    prec = metrics.precision_score(y_test, preds)
    rec = metrics.recall_score(y_test, preds)
    f1 = metrics.f1_score(y_test, preds)
    print('Accuracy:', acc)
    print('ROC AUC:', roc_auc)
    print('Precision:', prec)
    print('Recall:', rec)
    print('F-1 Score:', f1)


if __name__ == '__main__':
    data_dir = 'data'
    final_node_embedding = 'final'
    V, Y = load_dataset(data_dir, final_node_embedding)
    print(V['train'].shape, V['valid'].shape, V['test'].shape)
    print(Y['train'].shape, Y['valid'].shape, Y['test'].shape)
    X_train, Y_train = concat_train_valid(V, Y)
    print(X_train.shape, Y_train.shape)
