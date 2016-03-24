import numpy as np
import sklearn
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import sklearn.linear_model
import pandas

def mean_on_fold(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options):
    return np.ones((test.sum(), 1))*y[train].mean(), None


def random_on_fold(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options):
    return np.random.randn(test.sum(), 1), None


def xu_et_al_on_fold(feature_sets, train, test, y, y_all, X, dim, dimsum, learn_options):
    coef = pandas.read_csv(learn_options['xu_matrix_file'], skiprows=1, delimiter='\t')
    coef = coef[['A', 'T', 'C', 'G']] # swap columns so that they are in correct order
    coef = coef.values.flatten()[:, None]
    X = X.copy()
    X = np.append(X, np.zeros((X.shape[0], 3*4)), axis=1)
    X = X[:, 3*4:]
    y_pred = 1./(1+np.exp(-np.dot(X[test], coef)))

    return y_pred, coef

def doench_on_fold(feature_sets, train, test, y, y_all, X, dim, dimsum, learn_options):
    auto_class_weight = None  # 'auto'/None
    verbose = False
    penalty = [0.005*pow(1.15, x) for x in range(0, 45)]  # ian's code:  tvals = [0.005*pow(1.15,x) for x in range(0,45)]
    y_bin = y_all[learn_options['binary target name']].values[:, None]

    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(y_all['Target gene'].values[train])
    gene_classes = label_encoder.transform(y_all['Target gene'].values[train])
    cv = sklearn.cross_validation.StratifiedKFold(gene_classes, n_folds=10, shuffle=True)

    best_penalty = None

    cv_results = np.zeros((10, len(penalty)))

    for j, split in enumerate(cv):
        train_inner, test_inner = split
        for i, c in enumerate(penalty):
            # fit an L1-penalized SVM classifier
            clf = LinearSVC(penalty='l1', C=c, dual=False, class_weight=auto_class_weight)
            clf.fit(X[train][train_inner], y_bin[train][train_inner].flatten())

            # pass features with non-zero coeff to Logistic with l2 penalty (original code?)
            non_zero_coeff = (clf.coef_ != 0.0)

            if np.all(non_zero_coeff is False):
                # if all are zero, turn one on so as to be able to run the code.
                non_zero_coeff[0] = True

            clf = LogisticRegression(penalty='l2', class_weight=auto_class_weight)
            clf.fit(X[train][train_inner][:, non_zero_coeff.flatten()], y[train][train_inner].flatten())
            y_test = clf.predict_proba(X[train][test_inner][:, non_zero_coeff.flatten()])[:, 1]

            fpr, tpr, _ = sklearn.metrics.roc_curve(y_bin[train][test_inner], y_test)
            assert np.nan not in fpr, "found nan fpr"
            assert np.nan not in tpr, "found nan tpr"
            roc_auc = sklearn.metrics.auc(fpr, tpr)
            if verbose:
                print j, i, roc_auc
            cv_results[j][i] = roc_auc

    best_penalty = penalty[np.argmax(np.mean(cv_results, axis=0))]
    print "best AUC for penalty: ", np.median(cv_results, axis=0)
    clf = LinearSVC(penalty='l1', C=best_penalty, dual=False, class_weight=auto_class_weight)
    clf.fit(X[train], y_bin[train].flatten())
    non_zero_coeff = (clf.coef_ != 0.0)

    clf = LogisticRegression(penalty='l2', class_weight=auto_class_weight)
    clf.fit(X[train][:, non_zero_coeff.flatten()], y[train].flatten())
    y_pred = clf.predict_proba(X[test][:, non_zero_coeff.flatten()])[:, 1:2]

    return y_pred, clf


def sgrna_from_doench_on_fold(feature_sets, train, test, y, y_all, X, dim, dimsum, learn_options):
    assert len(feature_sets.keys()) == 1, "should only use sgRNA Score here"
    assert feature_sets.keys()[0] == "sgRNA Score"
    y_pred = X[test][:, 0]
    return y_pred, None


def SVC_on_fold(feature_sets, train, test, y, y_all, X, dim, dimsum, learn_options):
    y_bin = y_all[learn_options['binary target name']].values[:, None]
    clf = LinearSVC(penalty='l2', dual=False)
    clf.fit(X[train], y_bin[train].flatten())
    #y_pred = clf.predict(X[test])[:, None] # this returns 0/1
    y_pred = clf.decision_function(X[test])[:, None]
    return y_pred, clf


    
