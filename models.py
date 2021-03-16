from xgboost import XGBClassifier
from utils import load_dataset, concat_train_valid, evaluate_model


def xgboost_gbt(x_train, y_train):
    model = XGBClassifier()
    model.fit(x_train, y_train)
    return model


if __name__ == '__main__':
    # Load dataset
    data_dir = 'data'
    final_node_embedding = 'final'
    V, Y = load_dataset(data_dir, final_node_embedding)
    X_train, Y_train = concat_train_valid(V, Y)
    X_test = V['test']
    Y_test = Y['test']
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    # XGBoost GBT
    model = xgboost_gbt(X_train, Y_train)
    evaluate_model(model, X_test, Y_test)
