#-- Iván Felipe Rodríguez &  Remi Megret

import copy
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

# libact classes
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.models import SklearnAdapter
from libact.models import *
from libact.query_strategies import *
from libact.labelers import IdealLabeler


def run(trn_ds, tst_ds, lbr, model, qs, quota, fully_labeled_trn_ds):
    E_in, E_out, E_full = [], [], []

    for _ in range(quota):
        # Standard usage of libact objects
        ask_id = qs.make_query()
        X, _ = zip(*trn_ds.data)
        lb = lbr.label(X[ask_id])
        trn_ds.update(ask_id, lb)

        model.train(trn_ds)

        E_in = np.append(E_in, 1 - model.score(trn_ds))
        E_out = np.append(E_out, 1 - model.score(tst_ds))
        E_full = np.append(E_full, 1 - model.score(fully_labeled_trn_ds))

    return E_in, E_out, E_full


def split_train_test(X, y, test_size, n_labeled):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y)
    X_train0, _, y_train0, _ = train_test_split(
        X_train, y_train, train_size=n_labeled, stratify=y_train)
    #trn_ds = Dataset(X_train, np.concatenate([y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
    trn_ds = Dataset(X_train, np.concatenate(
        [y_train0, [None] * (len(y_train) - n_labeled)]))
    tst_ds = Dataset(X_test, y_test)
    fully_labeled_trn_ds = Dataset(X_train, y_train)

    return trn_ds, tst_ds, y_train, fully_labeled_trn_ds


def active_learning(data, labels, test_size, n_labeled):
    # Load dataset
    trn_ds, tst_ds, y_train, fully_labeled_trn_ds = split_train_test(
        data, labels, test_size, n_labeled)
    trn_ds2 = copy.deepcopy(trn_ds)
    lbr = IdealLabeler(fully_labeled_trn_ds)

    quota = len(y_train) - n_labeled    # number of samples to query

    # Comparing UncertaintySampling strategy with RandomSampling.
    # model is the base learner, e.g. LogisticRegression, SVM ... etc.
    clf = SklearnProbaAdapter(GradientBoostingClassifier(
        n_estimators=5, learning_rate=1.0, max_depth=2, random_state=0))
    qs = UncertaintySampling(trn_ds, method='lc', model=clf)
    model = clf
    E_in_1, E_out_1, E_full_1 = run(
        trn_ds, tst_ds, lbr, model, qs, quota, fully_labeled_trn_ds)

    qs2 = RandomSampling(trn_ds2)
    model = clf
    E_in_2, E_out_2, E_full_2 = run(
        trn_ds2, tst_ds, lbr, model, qs2, quota, fully_labeled_trn_ds)

    # Plot the learning curve of UncertaintySampling to RandomSampling
    # The x-axis is the number of queries, and the y-axis is the corresponding
    # error rate.
    rows = ["E_in_1", "E_in_2", "E_out_1", "E_out_2", "E_full_1", "E_full_2"]
    data = pd.DataFrame(data=[E_in_1, E_in_2, E_out_1,
                              E_out_2, E_full_1, E_full_2], index=rows)
    return data


def plotting(data, colors=['darkblue', 'orange', 'b', 'r', 'lightblue', 'pink']):
    query_num = np.arange(1, len(E_in_1) + 1)
    keys = data.keys()
    for k in range(0, len(keys), 2):
        plt.plot(query_num, data[keys[k]], color=colors[
                 k], label='qs ' + keys[k][:-2])
        plt.plot(query_num, data[
                 keys[k + 1]], color=colors[k + 1], label='random ' + keys[k + 1][:-2])

    plt.xlabel('Number of Queries')
    plt.ylabel('Error')
    plt.title('Experiment Result')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=5)
    plt.show()


def AL(data, labels, test_size, n_label, num_experiments):
    experiments = {}
    for i in range(num_experiments):
        result = active_learning(data, labels, test_size, n_labeled)
        experiments["Experiment" + str(i)] = result
    return pd.Panel(experiments)

def main():
    data=np.loadtxt("data.txt")
    labels=np.loadtxt("labels.txt")
    test_size = 0.25 # the percentage of samples in the dataset that will be                  # randomly selected and assigned to the test set
    n_labeled = 4 # number of samples that are initially labeled
    experiment = active_learning(data,labels,test_size,n_labeled)
    experiment.to_csv('experiment.csv')
main()

