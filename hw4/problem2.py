
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.model_selection import KFold


def shuffle_combine(a, b):
    random_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(random_state)
    np.random.shuffle(b)

def plot_error_rate(er_1, er_2, er_3):
    df_error = pd.DataFrame([er_1, er_2, er_3]).T
    df_error.columns = ['m=1', 'm=sqrt(p)', 'm=p']
    plot1 = df_error.plot(linewidth=3, figsize=(8, 6),
                          color=['lightblue', 'darkblue', 'deepskyblue'], grid=True)
    plot1.set_xlabel('Number of trees', fontsize=12)
    plot1.set_xticklabels(range(0, 500, 50))
    plot1.set_ylabel('Error rate', fontsize=12)
    plot1.set_title('Error rate vs number of trees', fontsize=16)
    plt.axhline(y=er_2[0], linewidth=1, color='red', ls='dashed')
    plt.show()


def plot_error_rate2(er_1, er_2):
    df_error = pd.DataFrame([er_1, er_2]).T
    df_error.columns = ['m=sqrt(p)', 'm=sqrt(p) - depth control']
    plot1 = df_error.plot(linewidth=3, figsize=(8, 6),
                          color=['lightblue', 'darkblue'], grid=True)
    plot1.set_xlabel('Number of trees', fontsize=12)
    plot1.set_xticklabels(range(0, 500, 50))
    plot1.set_ylabel('Error rate', fontsize=12)
    plot1.set_title('Error rate vs number of trees', fontsize=16)
    plt.axhline(y=er_2[0], linewidth=1, color='red', ls='dashed')
    plt.show()

def Y_encoding(data):
  Y = []
  for i in range(len(data)):
    if data['target'][i] == 'g':
      Y.append(1)
    else:
      Y.append(-1)
  return Y


class RandomForestClassifier(object):
    def __init__(self, n_estimators=32, max_features=None, max_depth=None,
                 min_samples_leaf=1, bootstrap=0.9):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.forest = []

    def fit(self, X, y):
        self.forest = []
        n_samples = len(y)
        n_sub_samples = round(n_samples * self.bootstrap)
        for i in range(self.n_estimators):
            shuffle_combine(X, y)
            X_subset = X[:int(n_sub_samples)]
            y_subset = y[:int(n_sub_samples)]
            tree = DecisionTreeClassifier(max_features=self.max_features, max_depth=self.max_depth,
                                          min_samples_leaf=self.min_samples_leaf)
            tree.fit(X_subset, y_subset)
            self.forest.append(tree)

    def predict(self, X):
        n_samples = X.shape[0]
        n_trees = len(self.forest)
        predictions = np.empty([n_trees, n_samples])
        for i in range(n_trees):
            predictions[i] = self.forest[i].predict(X)
        return mode(predictions)[0][0]

    def compute_accuracy(self, X, y):
        y_predict = self.predict(X)
        n_samples = len(y)
        correct = 0
        for i in range(n_samples):
            if y_predict[i] == y[i]:
                correct = correct + 1
        accuracy = correct / n_samples
        return accuracy

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
        x_range = range(1, 500, 10)
        err1 = []
        for i in x_range:
            rf = RandomForestClassifier(n_estimators=i, max_features=1)
            rf.fit(X_train, Y_train)
            err1.append(1 - rf.compute_accuracy(X_test, Y_test))
            print("kfold_num:%d, max_features=1, iter:%d" % (j, i))
        errors1.append(err1)
        x_range = range(1, 500, 10)
        err2 = []
        for i in x_range:
            rf = RandomForestClassifier(n_estimators=i, max_features=int(np.sqrt(34)))
            rf.fit(X_train, Y_train)
            err2.append(1 - rf.compute_accuracy(X_test, Y_test))
            print("kfold_num:%d, max_features=sqrt(p), iter:%d" % (j, i))
        errors2.append(err2)
        x_range = range(1, 500, 10)
        err3 = []
        for i in x_range:
            rf = RandomForestClassifier(n_estimators=i, max_features=34)
            rf.fit(X_train, Y_train)
            err3.append(1 - rf.compute_accuracy(X_test, Y_test))
            print("kfold_num:%d, max_features=p, iter:%d" % (j, i))
        errors3.append(err3)
        x_range = range(1, 500, 10)
        err4 = []
        for i in x_range:
            rf = RandomForestClassifier(n_estimators=i, max_features=int(np.sqrt(34)), min_samples_leaf=10)
            rf.fit(X_train, Y_train)
            err4.append(1 - rf.compute_accuracy(X_test, Y_test))
            print("kfold_num:%d, max_features=sqrt(p), min_samples_leaf=10, iter:%d" % (j, i))
        errors4.append(err4)
    errors1 = np.sum(errors1, 0) / 5
    errors2 = np.sum(errors2, 0) / 5
    errors3 = np.sum(errors3, 0) / 5
    errors4 = np.sum(errors4, 0) / 5
    plot_error_rate(errors1, errors2, errors3)
    plot_error_rate2(errors2, errors4)


