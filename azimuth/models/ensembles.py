import numpy as np
import sklearn.linear_model
import sklearn.ensemble as en
from sklearn.grid_search import GridSearchCV
import sklearn
from sklearn.linear_model import LinearRegression
import scipy as sp
from regression import linreg_on_fold
import sklearn
import sklearn.tree as tree
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score

def spearman_scoring(clf, X, y):
    y_pred = clf.predict(X).flatten()
    return sp.stats.spearmanr(y_pred, y.flatten())[0]



def adaboost_on_fold(feature_sets, train, test, y, y_all, X, dim, dimsum, learn_options, classification=False):
    '''
    AdaBoostRegressor/Classifier from scikitlearn.
    '''
        
    if learn_options['adaboost_version'] == 'python':
        if not learn_options['adaboost_CV']:
            if not classification:
                clf = en.GradientBoostingRegressor(loss=learn_options['adaboost_loss'], learning_rate=learn_options['adaboost_learning_rate'],
                                                   n_estimators=learn_options['adaboost_n_estimators'],
                                                   alpha=learn_options['adaboost_alpha'],
                                                   subsample=1.0, min_samples_split=2, min_samples_leaf=1,                                    max_depth=learn_options['adaboost_max_depth'],
                                                   init=None, random_state=None, max_features=None,
                                                   verbose=0, max_leaf_nodes=None, warm_start=False)
            else:
                clf = en.GradientBoostingClassifier(learning_rate=learn_options['adaboost_learning_rate'],
                                                   n_estimators=learn_options['adaboost_n_estimators'],
                                                   subsample=1.0, min_samples_split=2, min_samples_leaf=1,                                    max_depth=learn_options['adaboost_max_depth'],
                                                   init=None, random_state=None, max_features=None,
                                                   verbose=0, max_leaf_nodes=None, warm_start=False)

            clf.fit(X[train], y[train].flatten())
            y_pred = clf.predict(X[test])[:, None]
        else: # optimize the parameters if the adaboosted algorithm                       

            if learn_options["algorithm_hyperparam_search"]=="bo":
                print

                from hyperopt import hp, fmin, tpe, rand
                                           
                def adaboost_scoring_bo(params):
                    # label_encoder = sklearn.preprocessing.LabelEncoder()
                    # label_encoder.fit(y_all['Target gene'].values[train])
                    # gene_classes = label_encoder.transform(y_all['Target gene'].values[train])
                    # n_folds = len(np.unique(gene_classes))
                    cv = sklearn.cross_validation.KFold(y_all['Target gene'].values[train].shape[0], n_folds=20, shuffle=True) 
                    est = en.GradientBoostingRegressor(n_estimators=1000, learning_rate=params['learning_rate'], max_depth=params['max_depth'], 
                                                       min_samples_leaf=params['min_samples_leaf'], max_features=params['max_features'])
                    scorer = cross_val_score(est, X[train], y[train].flatten(), cv=cv, n_jobs=20)
                    return np.median(scorer)         
                space = {
                        'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
                         'max_depth': hp.quniform('max_depth', 1, 8, 1),
                         'min_samples_leaf': hp.quniform('min_samples_leaf', 3, 20, 1),
                         'max_features': hp.uniform('max_features', 0.05, 1.0)}

                best = fmin(adaboost_scoring_bo, space, algo=tpe.suggest, max_evals=50, verbose=1)
                print best
                clf = en.GradientBoostingRegressor(n_estimators=learn_options['adaboost_n_estimators'], 
                                                   learning_rate=best['learning_rate'], 
                                                   max_depth=best['max_depth'], 
                                                   min_samples_leaf=best['min_samples_leaf'], 
                                                   max_features=best['max_features'])

                clf.fit(X[train], y[train].flatten())
            elif learn_options["algorithm_hyperparam_search"]=="grid":
                 assert not classification, "need to tweak code below to do classificaton, as above"
                 n_jobs = 20

                 print "Adaboost with GridSearch"
                 from sklearn.grid_search import GridSearchCV
                 #param_grid = {'learning_rate': [0.1, 0.05, 0.01],
                 #              'max_depth': [4, 5, 6, 7],
                 #              'min_samples_leaf': [5, 7, 10, 12, 15],
                 #              'max_features': [1.0, 0.5, 0.3, 0.1]}
                 param_grid = {'learning_rate': [0.1, 0.01],
                               'max_depth': [4, 7],
                               'min_samples_leaf': [5, 15],
                               'max_features': [1.0, 0.1]}


                 label_encoder = sklearn.preprocessing.LabelEncoder()
                 label_encoder.fit(y_all['Target gene'].values[train])
                 gene_classes = label_encoder.transform(y_all['Target gene'].values[train])
                 n_folds = len(np.unique(gene_classes))
                 cv = sklearn.cross_validation.StratifiedKFold(gene_classes, n_folds=n_folds, shuffle=True)

                 est = en.GradientBoostingRegressor(loss=learn_options['adaboost_loss'], n_estimators=learn_options['adaboost_n_estimators'])
                 clf = GridSearchCV(est, param_grid, n_jobs=n_jobs, verbose=1, cv=cv, scoring=spearman_scoring, iid=False).fit(X[train], y[train].flatten())
                 print clf.best_params_
            else:
                raise Exception("if using adaboost_CV then need to specify grid (grid search) or bo (bayesian optimization)")


            y_pred = clf.predict(X[test])[:, None]
    else:
        raise NotImplementedError

    return y_pred, clf


def LASSOs_ensemble_on_fold(feature_sets, train, test, y, y_all, X, dim, dimsum, learn_options):
    train_indices = np.where(train)[0]
    sel = len(train_indices)*0.10
    permuted_ind = np.random.permutation(train_indices)
    valid_indices = permuted_ind[:sel]
    train_indices = permuted_ind[sel:]
    train_sub = np.zeros_like(train, dtype=bool)
    valid_sub = np.zeros_like(train, dtype=bool)
    train_sub[train_indices] = True
    valid_sub[valid_indices] = True

    validations = np.zeros((len(valid_indices), len(feature_sets.keys())))
    predictions = np.zeros((test.sum(), len(feature_sets.keys())))

    for i, feature_name in enumerate(feature_sets.keys()):
        X_feature = feature_sets[feature_name].values
        y_pred, m = linreg_on_fold(feature_sets, train_sub, valid_sub, y, y_all, X_feature, dim, dimsum, learn_options)
        predictions[:, i] = m.predict(X_feature[test]).flatten()
        validations[:, i] = y_pred.flatten()

    clf = LinearRegression()
    clf.fit(validations, y[valid_sub])
    y_pred = clf.predict(predictions)

    return y_pred, None


def randomforest_on_fold(feature_sets, train, test, y, y_all, X, dim, dimsum, learn_options):
    '''
    RandomForestRegressor from scikitlearn.
    '''
    clf = en.RandomForestRegressor(oob_score=True)
    clf.fit(X[train], y[train][:, 0])
    y_pred = clf.predict(X[test])[:, None]
    return y_pred, clf


def decisiontree_on_fold(feature_sets, train, test, y, y_all, X, dim, dimsum, learn_options):
    '''
    DecisionTreeRegressor from scikitlearn.
    '''
    clf = tree.DecisionTreeRegressor()
    clf.fit(X[train], y[train][:, 0])
    y_pred = clf.predict(X[test])[:, None]
    return y_pred, clf


def linear_stacking(y_train, X_train, X_test):
    clf = sklearn.linear_model.LinearRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred.flatten()


def pairwise_majority_voting(y):
    N = y.shape[0]
    y_pred = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            y_pred[i, j] = (y[i] > y[j]).sum() > y.shape[1]/2

    return y_pred.sum(1)/y_pred.sum(1).max()


def median(y):
    return np.median(y, axis=1)


def GBR_stacking(y_train, X_train, X_test):
    param_grid = {'learning_rate': [0.1, 0.05, 0.01],
                  'max_depth': [2, 3, 4, 5],  # [2, 3, 4, 6],
                  'min_samples_leaf': [1, 2, 3],  # ,5, 7],
                  'max_features': [1.0, 0.5, 0.3, 0.1]}

    est = en.GradientBoostingRegressor(loss='ls', n_estimators=100)
    clf = GridSearchCV(est, param_grid, n_jobs=3, verbose=1, cv=20, scoring=spearman_scoring).fit(X_train, y_train.flatten())
    # clf.fit(X_train, y_train.flatten())
    return clf.predict(X_test)


def GP_stacking(y_train, X_train, X_test):
    import GPy
    m = GPy.models.SparseGPRegression(X_train, y_train, num_inducing=20, kernel=GPy.kern.RBF(X_train.shape[1]))
    m.optimize('bfgs', messages=0)
    y_pred = m.predict(X_test)[0]
    return y_pred.flatten()


def SVM_stacking(y_train, X_train, X_test):
    parameters = {'kernel': ('linear', 'rbf'), 'C': np.linspace(1, 10, 10), 'gamma': np.linspace(1e-3, 1., 10)}
    svr = svm.SVR()
    clf = GridSearchCV(svr, parameters, n_jobs=3, verbose=1, cv=10, scoring=spearman_scoring)
    clf.fit(X_train, y_train.flatten())
    return clf.predict(X_test)
