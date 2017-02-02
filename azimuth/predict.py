import numpy as np
import sklearn
from sklearn.metrics import roc_curve, auc
import sklearn.metrics
import sklearn.cross_validation
import copy
import util
import time
import metrics as ranking_metrics
import azimuth.models.regression
import azimuth.models.ensembles
import azimuth.models.DNN
import azimuth.models.baselines
import multiprocessing


def fill_in_truth_and_predictions(truth, predictions, fold, y_all, y_pred, learn_options, test):
    truth[fold]['ranks'] = np.hstack((truth[fold]['ranks'],
                                      y_all[learn_options['rank-transformed target name']].values[test].flatten()))

    truth[fold]['thrs'] = np.hstack((truth[fold]['thrs'],
                                     y_all[learn_options['binary target name']].values[test].flatten()))

    if 'raw_target_name' in learn_options.keys():
        truth[fold]['raw'] = np.hstack((truth[fold]['raw'],
                                        y_all[learn_options['raw target name']].values[test].flatten()))

    predictions[fold] = np.hstack((predictions[fold], y_pred.flatten()))

    return truth, predictions


def construct_filename(learn_options, TEST):
    if learn_options.has_key("V"):
        filename = "V%s" % learn_options["V"]
    else:
        filename = "offV1"

    if TEST:
        filename = "TEST."

    filename += learn_options["method"]
    filename += '.order%d' % learn_options["order"]
    # try:
    #     learn_options["target_name"] = ".%s" % learn_options["target_name"].split(" ")[1]
    # except:
    #     pass
    filename += learn_options["target_name"]
    if learn_options["method"] == "GPy":
        pass
        # filename += ".R%d" % opt_options['num_restarts']
        # filename += ".K%s" % learn_options['kerntype']
        # if learn_options.has_key('degree'):
        #     filename += "d%d" % learn_options['degree']
        # if learn_options['warped']:
        #     filename += ".Warp"
    elif learn_options["method"] == "linreg":
        filename += "." + learn_options["penalty"]
    filename += "." + learn_options["cv"]

    if learn_options["training_metric"] == "NDCG":
        filename += ".NDGC_%d" % learn_options["NDGC_k"]
    elif learn_options["training_metric"] == "AUC":
        filename += ".AUC"
    elif learn_options["training_metric"] == 'spearmanr':
        filename += ".spearman"

    print "filename = %s" % filename
    return filename

def print_summary(global_metric, results, learn_options, feature_sets, flags):
    print "\nSummary:"
    print learn_options
    print "\t\tglobal %s=%.2f" % (learn_options['metric'], global_metric)
    print "\t\tmedian %s across folds=%.2f" % (learn_options['metric'], np.median(results[0]))
    print "\t\torder=%d" % learn_options["order"]
    if learn_options.has_key('kerntype'): "\t\tkern type = %s" % learn_options['kerntype']
    if learn_options.has_key('degree'): print "\t\tdegree=%d" % learn_options['degree']
    print "\t\ttarget_name=%s" % learn_options["target_name"]

    for k in flags.keys():
        print '\t\t' + k + '=' + str(learn_options[k])

    print "\t\tfeature set:"
    for set in feature_sets.keys():
        print "\t\t\t%s" % set
    print "\t\ttotal # features=%d" % results[4]

def extract_fpr_tpr_for_fold(aucs, fold, i, predictions, truth, y_binary, test, y_pred):
    assert len(np.unique(y_binary))<=2, "if using AUC need binary targets"
    fpr, tpr, _ = roc_curve(y_binary[test], y_pred)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

def extract_NDCG_for_fold(metrics, fold, i, predictions, truth, y_ground_truth, test, y_pred, learn_options):
    NDCG_fold = ranking_metrics.ndcg_at_k_ties(y_ground_truth[test].flatten(), y_pred.flatten(), learn_options["NDGC_k"])
    metrics.append(NDCG_fold)

def extract_spearman_for_fold(metrics, fold, i, predictions, truth, y_ground_truth, test, y_pred, learn_options):
    spearman = util.spearmanr_nonan(y_ground_truth[test].flatten(), y_pred.flatten())[0]
    assert not np.isnan(spearman), "found nan spearman"
    metrics.append(spearman)

def get_train_test(test_gene, y_all, train_genes=None):
    # this is a bit convoluted because the train_genes+test_genes may not add up to all genes
    # for e.g. when we load up V3, but then use only V2, etc.

    not_test = (y_all.index.get_level_values('Target gene').values != test_gene)

    if train_genes is not None:
        in_train_genes = np.zeros(not_test.shape, dtype=bool)
        for t_gene in train_genes:
            in_train_genes = np.logical_or(in_train_genes, (y_all.index.get_level_values('Target gene').values == t_gene))
        train = np.logical_and(not_test, in_train_genes)
    else:
        train = not_test
    #y_all['test'] as to do with extra pairs in V2
    if test_gene == 'dummy':
        test = train
    else:
         test = (y_all.index.get_level_values('Target gene').values== test_gene)

    # convert to indices
    test = np.where(test == True)[0]
    train = np.where(train == True)[0]
    return train, test


def cross_validate(y_all, feature_sets, learn_options=None, TEST=False, train_genes=None, CV=True):
    '''
    feature_sets is a dictionary of "set name" to pandas.DataFrame
    one set might be single-nucleotide, position-independent features of order X, for e.g.
    Method: "GPy" or "linreg"
    Metric: NDCG (learning to rank metric, Normalized Discounted Cumulative Gain); AUC
    Output: cv_score_median, gene_rocs
    When CV=False, it trains on everything (and tests on everything, just to fit the code)
    '''

    print "range of y_all is [%f, %f]" % (np.min(y_all[learn_options['target_name']].values), np.max(y_all[learn_options['target_name']].values))

    allowed_methods = ["GPy", "linreg", "AdaBoostRegressor", "AdaBoostClassifier",
                       "DecisionTreeRegressor", "RandomForestRegressor",
                       "ARDRegression", "GPy_fs", "mean", "random", "DNN",
                       "lasso_ensemble", "doench", "logregL1", "sgrna_from_doench", 'SVC', 'xu_et_al']

    assert learn_options["method"] in allowed_methods,"invalid method: %s" % learn_options["method"]
    assert learn_options["method"] == "linreg" and learn_options['penalty'] == 'L2' or learn_options["weighted"] is None, "weighted only works with linreg L2 right now"

    # construct filename from options
    filename = construct_filename(learn_options, TEST)

    print "Cross-validating genes..."
    t2 = time.time()

    y = np.array(y_all[learn_options["target_name"]].values[:,None],dtype=np.float64)

    # concatenate feature sets in to one nparray, and get dimension of each
    inputs, dim, dimsum, feature_names = util.concatenate_feature_sets(feature_sets)
    #import pickle; pickle.dump([y, inputs, feature_names, learn_options], open("saved_models/inputs.p", "wb" )); import ipdb; ipdb.set_trace()

    if not CV:
        assert learn_options['cv'] == 'gene', 'Must use gene-CV when CV is False (I need to use all of the genes and stratified complicates that)'

    # set-up for cross-validation
    ## for outer loop, the one Doench et al use genes for
    if learn_options["cv"] == "stratified":
        assert not learn_options.has_key("extra_pairs") or learn_options['extra pairs'], "can't use extra pairs with stratified CV, need to figure out how to properly account for genes affected by two drugs"
        label_encoder = sklearn.preprocessing.LabelEncoder()
        label_encoder.fit(y_all['Target gene'].values)
        gene_classes = label_encoder.transform(y_all['Target gene'].values)
        if 'n_folds' in learn_options.keys():
            n_folds = learn_options['n_folds']
        elif learn_options['train_genes'] is not None and learn_options["test_genes"] is not None:
            n_folds = len(learn_options["test_genes"])
        else:
            n_folds = len(learn_options['all_genes'])

        cv = sklearn.cross_validation.StratifiedKFold(gene_classes, n_folds=n_folds, shuffle=True)
        fold_labels = ["fold%d" % i for i in range(1,n_folds+1)]
        if learn_options['num_genes_remove_train'] is not None:
            raise NotImplementedException()
    elif learn_options["cv"]=="gene":
        cv = []

        if not CV:
            train_test_tmp = get_train_test('dummy', y_all) # get train, test split using a dummy gene
            #train_tmp, test_tmp = train_test_tmp
            # not a typo, using training set to test on as well, just for this case. Test set is not used
            # for internal cross-val, etc. anyway.
            #train_test_tmp = (train_tmp, train_tmp)
            cv.append(train_test_tmp)
            fold_labels = ["dummy_for_no_cv"]#learn_options['all_genes']

        elif learn_options['train_genes'] is not None and learn_options["test_genes"] is not None:
            assert learn_options['train_genes'] is not None and learn_options['test_genes'] is not None, "use both or neither"
            for i, gene in enumerate(learn_options['test_genes']):
                cv.append(get_train_test(gene, y_all, learn_options['train_genes']))
            fold_labels = learn_options["test_genes"]
            # if train and test genes are seperate, there should be only one fold
            train_test_disjoint = set.isdisjoint(set(learn_options["train_genes"].tolist()), set(learn_options["test_genes"].tolist()))

        else:
            for i, gene in enumerate(learn_options['all_genes']):
                train_test_tmp = get_train_test(gene, y_all)
                cv.append(train_test_tmp)
            fold_labels = learn_options['all_genes']

        if learn_options['num_genes_remove_train'] is not None:
            for i, (train,test) in enumerate(cv):
                unique_genes = np.random.permutation(np.unique(np.unique(y_all['Target gene'][train])))
                genes_to_keep = unique_genes[0:len(unique_genes) - learn_options['num_genes_remove_train']]
                guides_to_keep = []
                filtered_train = []
                for j, gene in enumerate(y_all['Target gene']):
                    if j in train and gene in genes_to_keep:
                        filtered_train.append(j)
                cv_i_orig = copy.deepcopy(cv[i])
                cv[i] = (filtered_train, test)
                if learn_options['num_genes_remove_train']==0:
                    assert np.all(cv_i_orig[0]==cv[i][0])
                    assert np.all(cv_i_orig[1]==cv[i][1])
                print "# train/train after/before is %s, %s" % (len(cv[i][0]), len(cv_i_orig[0]))
                print "# test/test after/before is %s, %s" % (len(cv[i][1]), len(cv_i_orig[1]))
    else:
        raise Exception("invalid cv options given: %s" % learn_options["cv"])

    cv = [c for c in cv] #make list from generator, so can subset for TEST case
    if TEST:
        ind_to_use = [0]#[0,1]
        cv = [cv[i] for i in ind_to_use]
        fold_labels = [fold_labels[i] for i in ind_to_use]

    truth =  dict([(t, dict([(m, np.array([])) for m in ['raw', 'ranks', 'thrs']])) for t in fold_labels])
    predictions =  dict([(t, np.array([])) for t in fold_labels])

    m = {}
    metrics = []

    #do the cross-validation
    num_proc = learn_options["num_proc"]
    if num_proc > 1:
        num_proc = np.min([num_proc,len(cv)])
        print "using multiprocessing with %d procs--one for each fold" % num_proc
        jobs = []
        pool = multiprocessing.Pool(processes=num_proc)
        for i,fold in enumerate(cv):
            train,test = fold
            print "working on fold %d of %d, with %d train and %d test" % (i, len(cv), len(train), len(test))
            if learn_options["method"]=="GPy":
                job = pool.apply_async(azimuth.models.GP.gp_on_fold, args=(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options))
            elif learn_options["method"]=="linreg":
                job = pool.apply_async(azimuth.models.regression.linreg_on_fold, args=(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options))
            elif learn_options["method"]=="logregL1":
                job = pool.apply_async(azimuth.models.regression.logreg_on_fold, args=(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options))
            elif learn_options["method"]=="AdaBoostRegressor":
                 job = pool.apply_async(azimuth.models.ensembles.adaboost_on_fold, args=(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options, False))
            elif learn_options["method"]=="AdaBoostClassifier":
                 job = pool.apply_async(azimuth.models.ensembles.adaboost_on_fold, args=(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options, True))
            elif learn_options["method"]=="DecisionTreeRegressor":
                job = pool.apply_async(azimuth.models.ensembles.decisiontree_on_fold, args=(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options))
            elif learn_options["method"]=="RandomForestRegressor":
                job = pool.apply_async(azimuth.models.ensembles.randomforest_on_fold, args=(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options))
            elif learn_options["method"]=="ARDRegression":
                job = pool.apply_async(azimuth.models.regression.ARDRegression_on_fold, args=(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options))
            elif learn_options["method"] == "random":
                job = pool.apply_async(azimuth.models.baselines.random_on_fold, args=(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options))
            elif learn_options["method"] == "mean":
                job = pool.apply_async(azimuth.models.baselines.mean_on_fold, args=(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options))
            elif learn_options["method"] == "SVC":
                job = pool.apply_async(azimuth.models.baselines.SVC_on_fold, args=(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options))
            elif learn_options["method"] == "DNN":
                job = pool.apply_async(azimuth.models.DNN.DNN_on_fold, args=(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options))
            elif learn_options["method"] == "lasso_ensemble":
                job = pool.apply_async(azimuth.models.ensembles.LASSOs_ensemble_on_fold, args=(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options))
            elif learn_options["method"] == "doench":
                    job = pool.apply_async(azimuth.models.baselines.doench_on_fold, args=(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options))
            elif learn_options["method"] == "sgrna_from_doench":
                    job = pool.apply_async(azimuth.models.baselines.sgrna_from_doench_on_fold, args=(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options))
            elif learn_options["method"] == "xu_et_al":
                    job = pool.apply_async(azimuth.models.baselines.xu_et_al_on_fold, args=(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options))
            else:
                raise Exception("did not find method=%s" % learn_options["method"])
            jobs.append(job)
        pool.close()
        pool.join()
        for i,fold in enumerate(cv):#i in range(0,len(jobs)):
            y_pred, m[i] = jobs[i].get()
            train,test = fold

            if learn_options["training_metric"]=="AUC":
                extract_fpr_tpr_for_fold(metrics, fold_labels[i], i, predictions, truth, y_all[learn_options["ground_truth_label"]].values, test, y_pred)
            elif learn_options["training_metric"]=="NDCG":
                extract_NDCG_for_fold(metrics, fold_labels[i], i, predictions, truth, y_all[learn_options["ground_truth_label"]].values, test, y_pred, learn_options)
            elif learn_options["training_metric"] == 'spearmanr':
                 extract_spearman_for_fold(metrics, fold_labels[i], i, predictions, truth, y_all[learn_options["ground_truth_label"]].values, test, y_pred, learn_options)
            else:
                raise Exception("invalid 'training_metric' in learn_options: %s" % learn_options["training_metric"])

            truth, predictions = fill_in_truth_and_predictions(truth, predictions, fold_labels[i], y_all, y_pred, learn_options, test)

        pool.terminate()

    else:
        # non parallel version
        for i,fold in enumerate(cv):
            train,test = fold
            if learn_options["method"]=="GPy":
                y_pred, m[i] = gp_on_fold(azimuth.models.GP.feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options)
            elif learn_options["method"]=="linreg":
                y_pred, m[i] = azimuth.models.regression.linreg_on_fold(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options)
            elif learn_options["method"]=="logregL1":
                y_pred, m[i] = azimuth.models.regression.logreg_on_fold(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options)
            elif learn_options["method"]=="AdaBoostRegressor":
                 y_pred, m[i] = azimuth.models.ensembles.adaboost_on_fold(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options, classification=False)
            elif learn_options["method"]=="AdaBoostClassifier":
                 y_pred, m[i] = azimuth.models.ensembles.adaboost_on_fold(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options, classification=True)
            elif learn_options["method"]=="DecisionTreeRegressor":
                 y_pred, m[i] = azimuth.models.ensembles.decisiontree_on_fold(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options)
            elif learn_options["method"]=="RandomForestRegressor":
                 y_pred, m[i] = azimuth.models.ensembles.randomforest_on_fold(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options)
            elif learn_options["method"]=="ARDRegression":
                 y_pred, m[i] = azimuth.models.regression.ARDRegression_on_fold(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options)
            elif learn_options["method"]=="GPy_fs":
                 y_pred, m[i] = azimuth.models.GP.gp_with_fs_on_fold(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options)
            elif learn_options["method"] == "random":
                 y_pred, m[i] = azimuth.models.baselines.random_on_fold(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options)
            elif learn_options["method"] == "mean":
                 y_pred, m[i] = azimuth.models.baselines.mean_on_fold(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options)
            elif learn_options["method"] == "SVC":
                 y_pred, m[i] = azimuth.models.baselines.SVC_on_fold(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options)
            elif learn_options["method"] == "DNN":
                 y_pred, m[i] = azimuth.models.DNN.DNN_on_fold(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options)
            elif learn_options["method"] == "lasso_ensemble":
                 y_pred, m[i] = azimuth.models.ensembles.LASSOs_ensemble_on_fold(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options)
            elif learn_options["method"] == "doench":
                 y_pred, m[i] = azimuth.models.baselines.doench_on_fold(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options)
            elif learn_options["method"] == "sgrna_from_doench":
                 y_pred, m[i] = azimuth.models.baselines.sgrna_from_doench_on_fold(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options)
            elif learn_options["method"] == "xu_et_al":
                 y_pred, m[i] = azimuth.models.baselines.xu_et_al_on_fold(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options)
            else:
                raise Exception("invalid method found: %s" % learn_options["method"])

            if learn_options["training_metric"]=="AUC":
                # fills in truth and predictions
                extract_fpr_tpr_for_fold(metrics, fold_labels[i], i, predictions, truth, y_all[learn_options['ground_truth_label']].values, test, y_pred)
            elif learn_options["training_metric"]=="NDCG":
                extract_NDCG_for_fold(metrics, fold_labels[i], i, predictions, truth, y_all[learn_options["ground_truth_label"]].values, test, y_pred, learn_options)
            elif learn_options["training_metric"] == 'spearmanr':
                extract_spearman_for_fold(metrics, fold_labels[i], i, predictions, truth, y_all[learn_options["ground_truth_label"]].values, test, y_pred, learn_options)

            truth, predictions = fill_in_truth_and_predictions(truth, predictions, fold_labels[i], y_all, y_pred, learn_options, test)

            print "\t\tRMSE: ", np.sqrt(((y_pred - y[test])**2).mean())
            print "\t\tSpearman correlation: ", util.spearmanr_nonan(y[test], y_pred)[0]
            print "\t\tfinished fold/gene %i of %i" % (i+1, len(fold_labels))


    cv_median_metric =[np.median(metrics)]
    gene_pred = [(truth, predictions)]
    print "\t\tmedian %s across gene folds: %.3f" % (learn_options["training_metric"], cv_median_metric[-1])

    t3 = time.time()
    print "\t\tElapsed time for cv is %.2f seconds" % (t3-t2)
    return metrics, gene_pred, fold_labels, m, dimsum, filename, feature_names
