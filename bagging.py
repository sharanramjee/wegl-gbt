from utils import *
from sklearn import metrics
from sklearn.ensemble import BaggingClassifier


def bagging_model(x_train, y_train):
    model = BaggingClassifier(
            n_estimators=25,
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

    # XGBoost GBT
    model = bagging_model(X_train, Y_train)
    preds = make_preds(model, X_test)
    print_metrics(preds, Y_test)
    print(metrics.confusion_matrix(Y_test, preds).ravel())
