from imblearn.over_sampling import *


def apply_adasyn(x_train, y_train):
    sm = ADASYN(sampling_strategy='minority')
    x_train_res, y_train_res = sm.fit_resample(x_train, y_train)
    return x_train_res, y_train_res


def apply_smote(x_train, y_train):
    sm = SMOTE(sampling_strategy='minority', k_neighbors=5)
    x_train_res, y_train_res = sm.fit_resample(x_train, y_train)
    return x_train_res, y_train_res


def apply_b_smote(x_train, y_train):
    sm = BorderlineSMOTE(sampling_strategy='minority')
    x_train_res, y_train_res = sm.fit_resample(x_train, y_train)
    return x_train_res, y_train_res


def apply_km_smote(x_train, y_train):
    sm = KMeansSMOTE(sampling_strategy='minority', cluster_balance_threshold=0.01)
    x_train_res, y_train_res = sm.fit_resample(x_train, y_train)
    return x_train_res, y_train_res


def apply_svm_smote(x_train, y_train):
    sm = SVMSMOTE(sampling_strategy='minority')
    x_train_res, y_train_res = sm.fit_resample(x_train, y_train)
    return x_train_res, y_train_res


def load_saved_model(file_path='checkpoints/rf_25.pkl'):
    with open(file_path, 'rb') as fid:
        model = pickle.load(fid) 
    return model

