from utils import *
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedRandomForestClassifier


def balanced_rf(x_train, y_train, n_estimators, min_samples_split, min_samples_leaf):
    model = BalandedRandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=16
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


def make_preds(model, x_test):
    preds = model.predict(x_test)
    return preds


if __name__ == '__main__':
    # Load dataset
    data_dir = 'data'
    final_node_embedding = 'final'
    V, Y = load_dataset(data_dir, final_node_embedding)
    X_train, Y_train = concat_train_valid(V, Y)
    X_test = V['test']
    Y_test = Y['test']
    X_train, Y_train = apply_smote(X_train, Y_train)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    model = xgboost_gbt(X_train, Y_train, 25, 'dart')   # XGBoost GBT
    #model = balanced_rf(X_train, Y_train, 25, 2, 2)
    preds = make_preds(model, X_test)
    print_metrics(preds, Y_test)
