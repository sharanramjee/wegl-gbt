from utils import *
from models import *
import sklearn.ensemble as sk_ensemble


def rf_embedding(V, n_estimators):
    model = sk_ensemble.RandomTreesEmbedding(
            n_estimators=n_estimators,
            n_jobs=16,
            ).fit(V['train'])
    V['train'] = model.transform(V['train']).toarray()
    V['valid'] = model.transform(V['valid']).toarray()
    V['test'] = model.transform(V['test']).toarray()
    return V


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
    V['train'] = X_train
    V['test'] = X_test
    Y['train'] = Y_train
    Y['test'] = Y_test

    V = rf_embedding(V, 100)
    print('After embedding:', V['train'].shape, Y['train'].shape, V['test'].shape, Y['test'].shape)
    model = xgboost_gbt(V['train'], Y_train, 100, 'dart')
    print_results(model, V, Y)

