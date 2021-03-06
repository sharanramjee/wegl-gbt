from utils import *
from balancing import *
from xgboost import XGBClassifier
import imblearn.ensemble as imb_ensemble
from sklearn.experimental import enable_hist_gradient_boosting
import sklearn.ensemble as sk_ensemble


def make_preds(model, x_test):
    preds = model.predict(x_test)
    return preds


def sklearn_rf(x_train, y_train, n_estimators, min_samples_split, min_samples_leaf):
    model = sk_ensemble.RandomForestClassifier(
            n_estimators=n_estimators,
            #max_depth=5,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=16,
            class_weight='balanced',
            )
    model.fit(x_train, y_train)
    return model


def balanced_rf(x_train, y_train, n_estimators, min_samples_split, min_samples_leaf):
    model = imb_ensemble.BalancedRandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=16,
            )
    model.fit(x_train, y_train)
    return model


def sklearn_gbt(x_train, y_train, n_estimators, min_samples_split, min_samples_leaf):
    model = sk_ensemble.GradientBoostingClassifier(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            )
    model.fit(x_train, y_train)
    return model


def xgboost_gbt(x_train, y_train, n_estimators, booster):
    model = XGBClassifier(
            n_estimators=n_estimators,
            booster=booster,
            n_jobs=16,
            )
    model.fit(x_train, y_train)
    return model


def sklearn_bagging(x_train, y_train, n_estimators):
    model = sk_ensemble.BaggingClassifier(
            n_estimators=n_estimators,
            n_jobs=16,
            )
    model.fit(x_train, y_train)
    return model


def balanced_bagging(x_train, y_train, n_estimators):
    model = imb_emsemble.BalancedBaggingClassifier(
            n_estimators=n_estimators,
            n_jobs=16,
            )
    model.fit(x_train, y_train)
    return model


def easy_ensemble(x_train, y_train, n_estimators, sampling_strategy):
    model = imb_ensemble.BalancedBaggingClassifier(
            n_estimators=n_estimators,
            sampling_strategy=sampling_strategy,
            n_jobs=4,
            )
    model.fit(x_train, y_train)
    return model


def ada_boost(x_train, y_train, n_estimators):
    model = sk_ensemble.AdaBoostClassifier(
            n_estimators=n_estimators,
            )
    model.fit(x_train, y_train)
    return model


def isolation_forest(x_train, y_train, n_estimators):
    model = sk_ensemble.IsolationForest(
            n_estimators=n_estimators,
            n_jobs=16,
            )
    model.fit(x_train, y_train)
    return model


def hgbt(x_train, y_train):
    model = sk_ensemble.HistGradientBoostingClassifier(
            l2_regularization=200,
            )
    model.fit(x_train, y_train)
    return model



if __name__ == '__main__':
    # Load dataset
    data_dir = 'data'
    final_node_embedding = 'final'
    V, Y = load_dataset(data_dir, final_node_embedding)
    X_train, Y_train = concat_train_valid(V, Y)
    #X_train = V['train']
    #Y_train = Y['train']
    X_test = V['test']
    Y_test = Y['test']
    X_train, Y_train = apply_smote(X_train, Y_train)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    #model = sklearn_rf(X_train, Y_train, 25, 2, 2)        # RF
    #model = balanced_rf(X_train, Y_train, 25, 2, 2)       # Balanced RF (doesn't work)
    #model = sklearn_gbt(X_train, Y_train, 30, 2, 2)       # GBT model (doesn't work)
    #model = xgboost_gbt(X_train, Y_train, 30, 'dart')     # XGBoost GBT
    #model = sklearn_bagging(X_train, Y_train, 25)         # Bagging Model
    #model = balanced_bagging(X_train, Y_train, 25)        # Balanced Bagging (doesn't work)
    #model = easy_ensemble(X_train, Y_train, 25, 'auto')   # Easy Ensemble (doesn't work)
    #model = ada_boost(X_train, Y_train, 25)               # AdaBoost (doesn't work)
    #model = hgbt(X_train, Y_train)                        # HGBT
    for i in range(10):
        print('Experiment:', i)
        model = hgbt(X_train, Y_train)
        #preds = make_preds(model, X_test)
        #print_metrics(preds, Y_test)
        #print_conf(preds, Y_test)
        print_results(model, V, Y)
