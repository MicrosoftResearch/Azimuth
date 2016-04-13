import numpy as np
import sklearn
from sklearn.linear_model import ARDRegression, LinearRegression
from sklearn.metrics import roc_curve, auc
import sklearn.linear_model
import azimuth.util
import azimuth.metrics as ranking_metrics
import azimuth.predict
import numbers

def ARDRegression_on_fold(feature_sets, train, test, y, y_all, X, dim, dimsum, learn_options):
    '''
    '''
    clf = ARDRegression()
    clf.fit(X[train], y[train][:, 0])
    y_pred = clf.predict(X[test])[:, None]
    return y_pred, clf


def train_linreg_model(alpha, l1r, learn_options, fold, X, y, y_all):
    '''
    fold is something like train_inner (boolean array specifying what is in the fold)
    '''
    if learn_options["penalty"] == "L2":
        clf = sklearn.linear_model.Ridge(alpha=alpha, fit_intercept=learn_options["fit_intercept"], normalize=learn_options['normalize_features'], copy_X=True, max_iter=None, tol=0.001, solver='auto')
        weights = get_weights(learn_options, fold, y, y_all)
        clf.fit(X[fold], y[fold], sample_weight=weights)
    elif learn_options["penalty"] == 'EN' or learn_options["penalty"] == 'L1':
        if learn_options["loss"] == "squared":
            clf = sklearn.linear_model.ElasticNet(alpha=alpha, l1_ratio=l1r, fit_intercept=learn_options["fit_intercept"], normalize=learn_options['normalize_features'], max_iter=3000)
        elif learn_options["loss"] == "huber":
            clf = sklearn.linear_model.SGDRegressor('huber', epsilon=0.7, alpha=alpha,
                                                    l1_ratio=l1r, fit_intercept=learn_options["fit_intercept"], n_iter=10,
                                                    penalty='elasticnet', shuffle=True, normalize=learn_options['normalize_features'])
        clf.fit(X[fold], y[fold])
    return clf


def logreg_on_fold(feature_sets, train, test, y, y_all, X, dim, dimsum, learn_options):
    '''
    (L1/L2 penalized) logistic reggresion using scikitlearn
    '''

    assert len(np.unique(y)) <= 2, "if using logreg need binary targets"
    assert learn_options["weighted"] is None, "cannot do weighted Log reg"
    assert learn_options['feature_select'] is False, "cannot do feature selection yet in logistic regression--see linreg_on_fold to implement"
    
    cv, n_folds = set_up_inner_folds(learn_options, y_all.iloc[train])

    assert learn_options['penalty'] == "L1" or learn_options['penalty'] == "L2", "can only use L1 or L2 with logistic regression"
    
    tol = 0.00001#0.0001
    
    performance = np.zeros((len(learn_options["alpha"]), 1))
    # degenerate_pred = np.zeros((len(learn_options["alpha"])))
    for train_inner, test_inner in cv:
        for i, alpha in enumerate(learn_options["alpha"]):
            clf = sklearn.linear_model.LogisticRegression(penalty=learn_options['penalty'].lower(), dual=False, fit_intercept=learn_options["fit_intercept"], class_weight=learn_options["class_weight"], tol=tol, C=1.0/alpha)
            
            clf.fit(X[train][train_inner], y[train][train_inner].flatten())
            #tmp_pred = clf.predict(X[train][test_inner])
            tmp_pred = clf.predict_proba(X[train][test_inner])[:,1]
            
            if learn_options["training_metric"] == "AUC":
                fpr, tpr, _ = roc_curve(y_all[learn_options["ground_truth_label"]][train][test_inner], tmp_pred)
                assert ~np.any(np.isnan(fpr)), "found nan fpr"
                assert ~np.any(np.isnan(tpr)), "found nan tpr"
                tmp_auc = auc(fpr, tpr)
                performance[i] += tmp_auc
            else:
                raise Exception("can only use AUC metric for cv with classification")

    performance /= n_folds

    max_score_ind = np.where(performance == np.nanmax(performance))
    assert max_score_ind != len(performance), "enlarge alpha range as hitting max boundary"

    # in the unlikely event of tied scores, take the first one.
    if len(max_score_ind[0]) > 1:
        max_score_ind = [max_score_ind[0][0], max_score_ind[1][0]]

    best_alpha = learn_options["alpha"][max_score_ind[0]]

    best_alpha = best_alpha[0]
    if not isinstance(best_alpha, numbers.Number):
        raise Exception("best_alpha must be a number but is %s" % type(best_alpha))

    print "\t\tbest alpha is %f from range=%s" % (best_alpha, learn_options["alpha"][[0, -1]])
    max_perf = np.nanmax(performance)

    if max_perf < 0.0:
        raise Exception("performance is negative")

    print "\t\tbest performance is %f" % np.nanmax(performance)

    clf = sklearn.linear_model.LogisticRegression(penalty=learn_options['penalty'],
                                                  dual=False, fit_intercept=learn_options["fit_intercept"],             class_weight=learn_options["class_weight"], tol=tol, C=1.0/best_alpha)
    clf.fit(X[train], y[train].flatten())

    # debugging check that get samed paramter estimation when have no regularization and use 
    # either data with only that feature on, or all data), AND WITH NO INTERCEPT
    if False:        
        # grab only feature "GA3"        
        keep_ind = np.where(feature_sets['mutletpos'].columns=="GA3")[0]
        print "%s, %s" % (str(clf.intercept_ ), str(clf.coef_[0, keep_ind]))
        clf.fit(X[train][:,keep_ind], y[train].flatten())
        print "%s, %s" % (str(clf.intercept_ ), str(clf.coef_))
        import ipdb; ipdb.set_trace()               

    
    #y_pred = clf.predict(X[test])
    y_pred = clf.predict_proba(X[test])[:,1]
    y_pred = y_pred[:, None]    
    #fpr, tpr, _ = roc_curve(y, y_pred); tmp_auc = auc(fpr, tpr)
    #import ipdb; ipdb.set_trace()
    return y_pred, clf


def linreg_on_fold(feature_sets, train, test, y, y_all, X, dim, dimsum, learn_options):
    '''
    linreg using scikitlearn, using more standard regression models with penalization requiring
    nested-cross-validation
    '''

    if learn_options["weighted"] is not None and (learn_options["penalty"] != "L2" or learn_options["method"] != "linreg"):
        raise NotImplementedError("weighted prediction not implemented for any methods by L2 at the moment")
        
    if not learn_options.has_key("fit_intercept"):
        learn_options["fit_intercept"] = True
    if not learn_options.has_key('normalize_features'):
        learn_options['normalize_features'] = True

    cv, n_folds = set_up_inner_folds(learn_options, y_all.iloc[train])

    if learn_options['penalty'] == "L1":
        l1_ratio = [1.0]
    elif learn_options['penalty'] == "L2":
        l1_ratio = [0.0]
    elif learn_options['penalty'] == "EN":  # elastic net
        l1_ratio = np.linspace(0.0, 1.0, 20)

    performance = np.zeros((len(learn_options["alpha"]), len(l1_ratio)))
    degenerate_pred = np.zeros((len(learn_options["alpha"])))
    for train_inner, test_inner in cv:
        for i, alpha in enumerate(learn_options["alpha"]):
            for j, l1r in enumerate(l1_ratio):
                clf = train_linreg_model(alpha, l1r, learn_options, train_inner, X[train], y[train], y_all.iloc[train])
                if learn_options["feature_select"]:
                    clf, tmp_pred = feature_select(clf, learn_options, test_inner, train_inner, X[train], y[train])
                else:
                    tmp_pred = clf.predict(X[train][test_inner])

                if learn_options["training_metric"] == "AUC":
                    fpr, tpr, _ = roc_curve(y_all[learn_options["ground_truth_label"]][train][test_inner], tmp_pred)
                    assert ~np.any(np.isnan(fpr)), "found nan fpr"
                    assert ~np.any(np.isnan(tpr)), "found nan tpr"
                    tmp_auc = auc(fpr, tpr)
                    performance[i, j] += tmp_auc

                elif learn_options['training_metric'] == 'spearmanr':
                    spearman = azimuth.util.spearmanr_nonan(y_all[learn_options['ground_truth_label']][train][test_inner], tmp_pred.flatten())[0]
                    performance[i, j] += spearman

                elif learn_options['training_metric'] == 'score':
                    performance[i, j] += clf.score(X[test_inner], y_all[learn_options['ground_truth_label']][train][test_inner])

                elif learn_options["training_metric"] == "NDCG":
                    assert "thresh" not in learn_options["ground_truth_label"], "for NDCG must not use thresholded ranks, but pure ranks"

                    # sorted = tmp_pred[np.argsort(y_all[ground_truth_label].values[test_inner])[::-1]].flatten()
                    # sortedgt = np.sort(y_all[ground_truth_label].values[test_inner])[::-1].flatten()
                    # tmp_perf = ranking_metrics.ndcg_at_k_ties(sorted, learn_options["NDGC_k"], sortedgt)
                    tmp_truth = y_all[learn_options["ground_truth_label"]].values[train][test_inner].flatten()
                    tmp_perf = ranking_metrics.ndcg_at_k_ties(tmp_truth, tmp_pred.flatten(), learn_options["NDGC_k"])
                    performance[i, j] += tmp_perf

                    degenerate_pred_tmp = len(np.unique(tmp_pred)) < len(tmp_pred)/2.0
                    degenerate_pred[i] += degenerate_pred_tmp

                    # sanity checking metric wrt ties, etc.
                    # rmse = np.sqrt(np.mean((tmp_pred - tmp_truth)**2))
                    tmp_pred_r, tmp_truth_r = ranking_metrics.rank_data(tmp_pred, tmp_truth)
                    # rmse_r = np.sqrt(np.mean((tmp_pred_r-tmp_truth_r)**2))

    performance /= n_folds

    max_score_ind = np.where(performance == np.nanmax(performance))
    assert max_score_ind != len(performance), "enlarge alpha range as hitting max boundary"
    # assert degenerate_pred[max_score_ind[0][0]]==0, "found degenerate predictions at max score"

    # in the unlikely event of tied scores, take the first one.
    if len(max_score_ind[0]) > 1:
        max_score_ind = [max_score_ind[0][0], max_score_ind[1][0]]

    best_alpha, best_l1r = learn_options["alpha"][max_score_ind[0]], l1_ratio[max_score_ind[1]]

    print "\t\tbest alpha is %f from range=%s" % (best_alpha, learn_options["alpha"][[0, -1]])
    
    if learn_options['penalty'] == "EN":
        print "\t\tbest l1_ratio is %f from range=%s" % (best_l1r, l1_ratio[[0, -1]])
    max_perf = np.nanmax(performance)

    if max_perf < 0.0:
        raise Exception("performance is negative")

    print "\t\tbest performance is %f" % max_perf

    clf = train_linreg_model(best_alpha, l1r, learn_options, train, X, y, y_all)
    if learn_options["feature_select"]:
        raise Exception("untested in a long time, should double check")
        clf, y_pred = feature_select(clf, learn_options, test, train, X, y)
    else:
        y_pred = clf.predict(X[test])

    if learn_options["penalty"] != "L2":
        y_pred = y_pred[:, None]
            
    return y_pred, clf


def feature_select(clf, learn_options, test_inner, train_inner, X, y):
    assert not learn_options["weighted"] is not None, "cannot currently do feature selection with weighted regression"
    assert learn_options["loss"] is not "huber", "won't use huber loss function with feature selection"
    non_zero_coeff = (clf.coef_ != 0.0)
    if non_zero_coeff.sum() > 0:
        clf = LinearRegression()
        clf.fit(X[train_inner][:, non_zero_coeff.flatten()], y[train_inner])
        tmp_pred = clf.predict(X[test_inner][:, non_zero_coeff.flatten()])
    else:
        tmp_pred = np.ones_like(test_inner)
    return clf, tmp_pred


def get_weights(learn_options, fold, y, y_all):
    '''
    fold is an object like train_inner which is boolean for which indexes are in the fold
    '''
    weights = None
    if learn_options["weighted"] == "variance":
        weights = 1.0/y_all["variance"].values[fold]
    elif learn_options["weighted"] == "ndcg":
        # DCG: r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        N = len(fold)
        r = np.ones(N)
        discount = np.concatenate((np.array([r[0]]), r[1:] / np.log2(np.arange(2, r.size + 1))))[::1]
        ind = np.argsort(y[fold], axis=0).flatten()
        weights = np.ones(len(ind))
        weights[ind] = discount
    elif learn_options["weighted"] == "rank":
        N = len(y[fold])
        inverse_ranks = (np.arange(N) + 1.0)[::-1]
        ind = np.argsort(y[fold], axis=0).flatten()
        weights = np.ones(len(ind))
        weights[ind] = inverse_ranks
    elif learn_options["weighted"] == "score":
        N = len(y[fold])
        score = y[fold] + np.abs(np.min(y[fold]))
        ind = np.argsort(y[fold], axis=0).flatten()
        weights = np.ones(len(ind))
        weights[ind] = score
    elif learn_options["weighted"] == "random":
        N = len(y[fold])
        weights = np.random.rand(N)
    elif learn_options["weighted"] is not None:
        raise Exception("invalid weighted type, %s" % learn_options["weighted"])
    # plt.plot(weights, y[train_inner],'.')
    return weights


def set_up_inner_folds(learn_options, y):            
    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(y['Target gene'].values)    
    gene_classes = label_encoder.transform(y['Target gene'].values)
    n_genes = len(np.unique(gene_classes))    
    if learn_options['ignore_gene_level_for_inner_loop'] or learn_options["cv"] == "stratified" or n_genes==1:
        if 'n_folds' not in learn_options.keys():
            n_folds = len(np.unique(gene_classes))
        else:
            n_folds = learn_options['n_folds']        
        cv = sklearn.cross_validation.StratifiedKFold(gene_classes, n_folds=n_folds, shuffle=True)
    elif learn_options["cv"] == "gene":
        gene_list = np.unique(y['Target gene'].values)
        cv = []
        for gene in gene_list:
            cv.append(azimuth.predict.get_train_test(gene, y))
        n_folds = len(cv)
    return cv, n_folds
