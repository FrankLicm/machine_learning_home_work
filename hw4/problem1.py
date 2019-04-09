import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import time


def classify(Y_train, X_train, Y_test, X_test, clf):
    clf.fit(X_train, Y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    return compute_error(pred_train, Y_train), \
           compute_error(pred_test, Y_test)

def plot_error_rate(er_1, er_2, er_3):
    df_error = pd.DataFrame([er_1, er_2, er_3]).T
    df_error.columns = ['Max split = 1', 'Max split = 5', 'Max split = 10']
    plot1 = df_error.plot(linewidth=3, figsize=(8, 6),
                          color=['lightblue', 'darkblue', 'deepskyblue'], grid=True)
    plot1.set_xlabel('Number of weak learners', fontsize=12)
    plot1.set_xticklabels(range(0, 300, 50))
    plot1.set_ylabel('Error rate', fontsize=12)
    plot1.set_title('Error rate vs number of weak learners', fontsize=16)
    plt.axhline(y=er_2[0], linewidth=1, color='red', ls='dashed')
    plt.show()


def plot_error_rate2(er_1, er_2):
    df_error = pd.DataFrame([er_1, er_2]).T
    df_error.columns = ['Max split = 5', 'Max split = 5, shrink']
    plot1 = df_error.plot(linewidth=3, figsize=(8, 6),
                          color=['lightblue', 'darkblue'], grid=True)
    plot1.set_xlabel('Number of weak learners', fontsize=12)
    plot1.set_xticklabels(range(0, 300, 50))
    plot1.set_ylabel('Error rate', fontsize=12)
    plot1.set_title('Error rate vs number of weak learners', fontsize=16)
    plt.axhline(y=er_2[0], linewidth=1, color='red', ls='dashed')
    plt.show()

def adaboost_m1(Y_train, X_train, Y_test, X_test, M, clf, lr):
    n_train, n_test = len(X_train), len(X_test)
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
    for i in range(M):
        clf.fit(X_train, Y_train, sample_weight=w)
        pred_train_i = clf.predict(X_train)
        pred_test_i = clf.predict(X_test)
        error = [int(x) for x in (pred_train_i != Y_train)]
        error2 = [x if x == 1 else -1 for x in error]
        err_m = np.dot(w, error) / sum(w)
        alpha_m = lr * 0.5 * np.log((1 - err_m) / float(err_m))
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in error2]))
        pred_train = [sum(x) for x in zip(pred_train,
                                          [x * alpha_m for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test,
                                         [x * alpha_m for x in pred_test_i])]
    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    return compute_error(pred_train, Y_train), \
           compute_error(pred_test, Y_test)

def Y_encoding(data):
    Y = []
    for i in range(len(data)):
        if data['target'][i] == 'g':
            Y.append(1)
        else:
            Y.append(-1)
    return Y


def compute_error(pred, Y):
    return sum(pred != Y) / float(len(Y))


if __name__ == '__main__':
    data = pd.read_csv('ionosphere_processed.csv', index_col=0)
    data_mat = data.as_matrix()
    n_rows, n_cols = data_mat.shape
    X = data_mat[:, :n_cols - 1]
    y = Y_encoding(data)
    y = np.asarray(y)
    t = time.time()
    kf = KFold(n_splits=5)
    errors1 = []
    errors2 = []
    errors3 = []
    errors4 = []
    j = 0
    for train_index, test_index in kf.split(X):
        j += 1
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
        clf_tree = DecisionTreeClassifier(max_leaf_nodes=2)
        er_tree = classify(Y_train, X_train, Y_test, X_test, clf_tree)
        er_train1, er_test1 = [er_tree[0]], [er_tree[1]]
        x_range = range(1, 300, 10)
        for i in x_range:
            er_i = adaboost_m1(Y_train, X_train, Y_test, X_test, i, clf_tree, 1)
            er_train1.append(er_i[0])
            er_test1.append(er_i[1])
            print("kfold_num:%d, max split=1, iter:%d" % (j, i))
        errors1.append(er_test1)
        clf_tree2 = DecisionTreeClassifier(max_leaf_nodes=6)
        er_tree2 = classify(Y_train, X_train, Y_test, X_test, clf_tree2)
        er_train2, er_test2 = [er_tree2[0]], [er_tree2[1]]
        x_range = range(1, 300, 10)
        for i in x_range:
            er_i = adaboost_m1(Y_train, X_train, Y_test, X_test, i, clf_tree2, 1)
            er_train2.append(er_i[0])
            er_test2.append(er_i[1])
            print("kfold_num:%d, max split=5, iter:%d" % (j, i))
        errors2.append(er_test2)
        clf_tree3 = DecisionTreeClassifier(max_leaf_nodes=11)
        er_tree3 = classify(Y_train, X_train, Y_test, X_test, clf_tree3)
        er_train3, er_test3 = [er_tree3[0]], [er_tree3[1]]
        x_range = range(1, 300, 10)
        for i in x_range:
            er_i = adaboost_m1(Y_train, X_train, Y_test, X_test, i, clf_tree3, 1)
            er_train3.append(er_i[0])
            er_test3.append(er_i[1])
            print("kfold_num:%d, max split=10, iter:%d" % (j, i))
        errors3.append(er_test3)
        clf_tree4 = DecisionTreeClassifier(max_leaf_nodes=6)
        er_tree4 = classify(Y_train, X_train, Y_test, X_test, clf_tree4)
        er_train4, er_test4 = [er_tree4[0]], [er_tree4[1]]
        x_range = range(1, 300, 10)
        for i in x_range:
            er_i = adaboost_m1(Y_train, X_train, Y_test, X_test, i, clf_tree4, 0.9)
            er_train4.append(er_i[0])
            er_test4.append(er_i[1])
            print("kfold_num:%d, max split=5, shrink, iter:%d" % (j, i))
        errors4.append(er_test4)
    errors1 = np.sum(errors1, 0) / 5
    errors2 = np.sum(errors2, 0) / 5
    errors3 = np.sum(errors3, 0) / 5
    errors4 = np.sum(errors4, 0) / 5
    plot_error_rate(errors1, errors2, errors3)
    plot_error_rate2(errors2, errors4)
