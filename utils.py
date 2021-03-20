import pickle
import numpy as np
from sklearn import metrics
from prettytable import PrettyTable
from collections import defaultdict

import ogb
from ogb.graphproppred import Evaluator
from ogb.graphproppred import PygGraphPropPredDataset


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


def print_conf(preds, y_test):
    tp = list()
    fp = list()
    tn = list()
    fn = list()
    for i in range(len(preds)):
        if preds[i] == y_test[i] == 1:
            tp.append(i)
        if preds[i] == 1 and y_test[i] == 0:
            fp.append(i)
        if preds[i] == y_test[i] == 0:
            tn.append(i)
        if preds[i] == 0 and y_test[i] == 1:
            fn.append(i)
    print('Num TP:', len(tp), 'Examples:', tp[:10])
    print('Num FP:', len(fp), 'Examples:', fp[:10])
    print('Num TN:', len(tn), 'Examples:', tn[:10])
    print('Num FN:', len(fn), 'Examples:', fn[:10])     


def print_metrics(preds, y_test):
    acc = metrics.accuracy_score(y_test, preds)
    roc_auc = metrics.roc_auc_score(y_test, preds)
    prec = metrics.precision_score(y_test, preds)
    rec = metrics.recall_score(y_test, preds)
    f1 = metrics.f1_score(y_test, preds)
    conf_mat = metrics.confusion_matrix(y_test, preds)
    print('Accuracy:', acc)
    print('ROC-AUC:', roc_auc)
    print('Precision:', prec)
    print('Recall:', rec)
    print('F-1 Score:', f1)
    print('Confusion Matrix:', conf_mat)


def print_results(model, V, Y):
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv")
    results_table = PrettyTable()
    results_table.title = 'Final ROC-AUC(%) results for the {0} dataset with \'{1}\' node embedding and one-hot 13-dim edge embedding'.format(dataset.name, 'final')
    results_table.field_names = ['Classifier', '# Diffusion Layers', 'Node Embedding Size', 'Train.', 'Val.', 'Test']
    evaluator = Evaluator(name=dataset.name)
    results = defaultdict(list)
    phases = ['train', 'valid', 'test']
    for phase in phases:
        pred_probs = model.predict_proba(V[phase].reshape(V[phase].shape[0], -1))
        input_dict = {'y_true': np.array(Y[phase]).reshape(-1,1),
                'y_pred': pred_probs[:, 1].reshape(-1,1)}
        result_dict = evaluator.eval(input_dict)
        results[phase].append(result_dict['rocauc'])
    results_table.add_row(['rf', str(4), str(300)] + ['{0:.2f} $\pm$ {1:.2f}'.\
            format(100 * np.mean(results[phase]), \
            100 * np.std(results[phase])) for phase in phases])
    print('\n\n' + results_table.title)
    print(results_table)


if __name__ == '__main__':
    data_dir = 'data'
    final_node_embedding = 'final'
    V, Y = load_dataset(data_dir, final_node_embedding)
    print(V['train'].shape, V['valid'].shape, V['test'].shape)
    print(Y['train'].shape, Y['valid'].shape, Y['test'].shape)
    X_train, Y_train = concat_train_valid(V, Y)
    print(X_train.shape, Y_train.shape)
