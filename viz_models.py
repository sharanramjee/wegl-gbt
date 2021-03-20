import os
from utils import *
from sklearn import tree
from sklearn import tree
import matplotlib.pyplot as plt
from models import hgbt, sklearn_rf
from sklearn.tree import export_graphviz


def draw_model(model):
    fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (20,4), dpi=900)
    for index in range(0, 5):
        tree.plot_tree(
                model.estimators_[index],
                #feature_names = fn,class_names=cn,
                filled = True,
                ax = axes[index])
        axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
    fig.savefig('model_viz.png')


def count_params(model):
    num_trees = len(model.estimators_)
    param_count = 0
    for i in range(num_trees):
        tree = model.estimators_[i].tree_
        param_count += tree.node_count
    print('Num params:', param_count)


if __name__ == '__main__':
    data_dir = 'data'
    final_node_embedding = 'final'
    V, Y = load_dataset(data_dir, final_node_embedding)
    X_train, Y_train = concat_train_valid(V, Y)
    X_test = V['test']
    Y_test = Y['test']
    X_train, Y_train = apply_smote(X_train, Y_train)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    model = sklearn_rf(X_train, Y_train, 19, 2, 2)
    print_results(model, V, Y)
    #model = load_saved_model()
    #draw_model(model)
    count_params(model)
    
