import azimuth.predict as pd
import copy
import os
import numpy as np
import azimuth.util
import shutil
import pickle
import pylab as plt
import pandas
import azimuth.local_multiprocessing
import azimuth.load_data
import azimuth.features.featurization as feat

def check_feature_set_dims(feature_sets):
    F2 = None
    for set in feature_sets.keys():
        F = feature_sets[set].shape[0]
        if F2 is None: F = F2
        assert F == F2, "not same # individuals for feature %s" % set

    assert feature_sets !={}, "features are empty, check learn_options"


def set_target(learn_options, classification):
    assert 'target_name' not in learn_options.keys() or learn_options['target_name'] is not None, "changed it to be automatically set here"
    if not classification:
        learn_options["target_name"] = learn_options['rank-transformed target name']
        learn_options["training_metric"] = 'spearmanr'
        learn_options['ground_truth_label'] = learn_options['target_name']
    else:
        learn_options["target_name"] = learn_options['binary target name']
        learn_options["training_metric"] = 'AUC'
        learn_options['ground_truth_label'] = learn_options['binary target name']

    if learn_options["V"]==3:
        assert learn_options['target_name']=='score_drug_gene_rank' or learn_options['target_name']=='score_drug_gene_threshold', "cannot use raw scores when mergind data"
        assert learn_options["ground_truth_label"]=='score_drug_gene_rank' or learn_options["ground_truth_label"]=='score_drug_gene_threshold', "cannot use raw scores when mergind data"

    return learn_options

def GP_setup(learn_options, likelihood='gaussian', degree=3, set_target_fn=set_target):
    learn_options["method"] = "GPy"
    learn_options['kernel degree'] = degree

    if likelihood == 'warped':
        learn_options['warpedGP'] = True
    else:
        learn_options['warpedGP'] = False
    learn_options = set_target_fn(learn_options, classification=False)

    return learn_options

def SVC_setup(learn_options, likelihood='gaussian', degree=3,  set_target_fn=set_target):
    learn_options["method"] = "SVC"
    learn_options = set_target_fn(learn_options, classification=True)

    return learn_options

def L1_setup(learn_options, set_target_fn=set_target):
    learn_options = set_target_fn(learn_options, classification=False)
    learn_options["method"] = "linreg"
    learn_options["penalty"] = "L1"
    learn_options["feature_select"] = False
    if "alpha" not in learn_options.keys():
        learn_options["alpha"] = np.array([1e-6*pow(1.3,x) for x in range(0,100)])
    learn_options["loss"] = "squared"

    return learn_options

def L2_setup(learn_options, set_target_fn=set_target):
    learn_options = set_target_fn(learn_options, classification=False)
    learn_options["method"] = "linreg"
    learn_options["penalty"] = "L2"
    learn_options["feature_select"] = False
    if "alpha" not in learn_options.keys():
        learn_options["alpha"] = np.array([1e-6*pow(1.3,x) for x in range(0,100)])
    learn_options["loss"] = "squared"

    return learn_options

def mean_setup(learn_options, set_target_fn=set_target):
    learn_options = set_target_fn(learn_options, classification=False)
    learn_options['method'] = 'mean'
    return learn_options

def random_setup(learn_options, set_target_fn=set_target):
    learn_options = set_target_fn(learn_options, classification=False)
    learn_options['method'] = 'random'
    return learn_options

def elasticnet_setup(learn_options, set_target_fn=set_target):
    learn_options = set_target_fn(learn_options, classification=False)
    learn_options["method"] = "linreg"
    learn_options["penalty"] = "EN"
    learn_options["feature_select"] = False
    learn_options["loss"] = "squared"
    if "alpha" not in learn_options.keys():
        learn_options["alpha"] = np.array([1e-5*pow(2,x) for x in range(0,30)])
    return learn_options

def DNN_setup(learn_options, set_target_fn=set_target):
    learn_options = set_target_fn(learn_options, classification=False)
    learn_options['method'] = 'DNN'
    learn_options['DNN target variable'] = 'score'#'score_drug_gene_quantized'
    # learn_options['DNN architecture'] = (119, 10, 10, 10, 2)
    return learn_options

def RF_setup(learn_options, set_target_fn=set_target):
    learn_options = set_target_fn(learn_options, classification=False)
    learn_options['method'] = 'RandomForestRegressor'
    return learn_options

def doench_setup(learn_options, set_target_fn=set_target):
    learn_options = set_target_fn(learn_options, classification=True)
    learn_options['method'] = 'doench'
    return learn_options

def sgrna_from_doench_setup(learn_options, set_target_fn=set_target):
    learn_options = set_target_fn(learn_options, classification=False)
    learn_options['method'] = 'sgrna_from_doench'
    return learn_options

def linreg_setup(learn_options, set_target_fn=set_target):
    learn_options["method"] = "linreg"
    learn_options["penalty"] = "L1"
    learn_options["feature_select"] = False
    if "alpha" not in learn_options.keys():
        learn_options["alpha"] = np.array([0.0])
    learn_options["loss"] = "squared"
    learn_options = set_target_fn(learn_options, classification=False)

    return learn_options

def logregL1_setup(learn_options, set_target_fn=set_target):
    learn_options = set_target_fn(learn_options, classification=True)
    learn_options["method"] = "logregL1"
    learn_options["penalty"] = "L1"
    learn_options["feature_select"] = False
    if "alpha" not in learn_options.keys():
        learn_options["alpha"] = np.array([1e-6*pow(1.3,x) for x in range(0,100)])
    if not learn_options.has_key("fit_intercept"):
        learn_options["fit_intercept"] = True
    return learn_options

def LASSOs_ensemble_setup(learn_options, set_target_fn=set_target):
    learn_options = set_target_fn(learn_options, classification=False)
    learn_options["method"] = "lasso_ensemble"
    learn_options["penalty"] = "L1"
    learn_options["feature_select"] = False
    if "alpha" not in learn_options.keys():
        learn_options["alpha"] = np.array([1e-6*pow(1.3,x) for x in range(0,100)])
    learn_options["loss"] = "squared"

    return learn_options

def xu_et_al_setup(learn_options, set_target_fn=set_target):
    learn_options = set_target_fn(learn_options, classification=True)
    learn_options["method"] = "xu_et_al"

    return learn_options

def adaboost_setup(learn_options, num_estimators=100, max_depth=3, learning_rate=0.1, set_target_fn=set_target, model="AdaBoost"):
    """
    """
    learn_options = set_target_fn(learn_options, classification=False)
    if model=="AdaBoost":
        learn_options['method'] = "AdaBoostRegressor"
    elif model=="AdaBoostClassifier":
        learn_options['method'] = "AdaBoostClassifier"
    else:
        raise Exception("model must be either AdaBoost or AdaBoost Classifier")
    learn_options['adaboost_version'] = 'python' # "R" or "python"

    if 'adaboost_loss' not in learn_options.keys() and model=="AdaBoostRegressor":
        learn_options['adaboost_loss'] = 'ls' # alternatives: "lad", "huber", "quantile", see scikit docs for details
    if 'adaboost_alpha' not in learn_options.keys():
        learn_options['adaboost_alpha'] = 0.5 # this parameter is only used by the huber and quantile loss functions.

    if not learn_options['adaboost_CV']:
        learn_options['adaboost_learning_rate'] = learning_rate
        learn_options['adaboost_n_estimators'] = num_estimators
        learn_options['adaboost_max_depth'] = max_depth
    else:
        learn_options['adaboost_n_estimators'] = num_estimators

    return learn_options


def shared_setup(learn_options, order, test):
    if 'num_proc' not in learn_options.keys():
        learn_options['num_proc'] = None
    if 'num_thread_per_proc' not in learn_options.keys():
        learn_options['num_thread_per_proc'] = None

    num_proc = azimuth.local_multiprocessing.configure(TEST=test, num_proc=learn_options["num_proc"],
                                                num_thread_per_proc=learn_options["num_thread_per_proc"])
    learn_options["num_proc"] = num_proc

    learn_options["order"] = order  # gets used many places in code, not just here

    if "cv" not in learn_options.keys():
        # if no CV preference is specified, use leave-one-gene-out
        learn_options["cv"] = "gene"

    if "normalize_features" not in learn_options.keys():
        # if no CV preference is specified, use leave-one-gene-out
        learn_options["normalize_features"] = True

    if "weighted" not in learn_options.keys():
        learn_options['weighted'] = None

    if "all pairs" not in learn_options.keys():
        learn_options["all pairs"] = False

    if "include_known_pairs" not in learn_options.keys():
        learn_options["include_known_pairs"] = False

    if "include_gene_guide_feature" not in learn_options.keys():
        learn_options["include_gene_guide_feature"] = 0 #used as window size, so 0 is none

    #these should default to true to match experiments before they were options:
    if "gc_features" not in learn_options.keys():
        learn_options["gc_features"] = True
    if "nuc_features" not in learn_options.keys():
        learn_options["nuc_features"] = True

    if 'train_genes' not in learn_options.keys():
        learn_options["train_genes"] = None
    if 'test_genes' not in learn_options.keys():
        learn_options["test_genes"] = None

    if "num_proc" not in learn_options:
        learn_options["num_proc"] = None
    if "num_thread_per_proc" not in learn_options:
        learn_options["num_thread_per_proc"] = None

    if 'seed' not in learn_options:
        learn_options['seed'] = 1

    if "flipV1target" not in learn_options:
        learn_options["flipV1target"] = False

    if 'num_genes_remove_train' not in learn_options:
        learn_options['num_genes_remove_train'] = None

    if "include_microhomology" not in learn_options:
        learn_options["include_microhomology"] = False

    if "algorithm_hyperparam_search" not in learn_options:
        learn_options["algorithm_hyperparam_search"] = "grid" # other options is bo for bayesian optimization

    return num_proc

def setup(test=False, order=1, learn_options=None, data_file=None, pam_audit=True, length_audit=True):

    num_proc = shared_setup(learn_options, order, test)

    assert "testing_non_binary_target_name" in learn_options.keys(), "need this in order to get metrics, though used to be not needed, so you may newly see this error"
    if learn_options["testing_non_binary_target_name"] not in ['ranks', 'raw', 'thrs']:
        raise Exception('learn_otions["testing_non_binary_target_name"] must be in ["ranks", "raw", "thrs"]')

    Xdf, Y, gene_position, target_genes = azimuth.load_data.from_file(data_file, learn_options)
    learn_options['all_genes'] = target_genes

    if test:
        learn_options["order"] = 1

    if 'convert_30mer_to_31mer' in learn_options and learn_options['convert_30mer_to_31mer'] is True:
        print "WARNING!!! converting 30 mer to 31 mer (and then cutting off first nucleotide to go back to 30mer with a right shift)"
        for i in range(Xdf.shape[0]):
            Xdf['30mer'].iloc[i] = azimuth.util.convert_to_thirty_one(Xdf.iloc[i]["30mer"], Xdf.index.values[i][1], Xdf.iloc[i]["Strand"])
        # to_keep = Xdf['30mer'].isnull() == False
        # Xdf = Xdf[to_keep]
        # gene_position = gene_position[to_keep]
        # Y = Y[to_keep]
        Xdf["30mer"] = Xdf["30mer"].apply(lambda x: x[1:]) # chop the first nucleotide

    if learn_options.has_key('left_right_guide_ind') and learn_options['left_right_guide_ind'] is not None:
        seq_start, seq_end, expected_length = learn_options['left_right_guide_ind']
        assert len(Xdf["30mer"].values[0]) == expected_length
        Xdf['30mer'] = Xdf['30mer'].apply(lambda seq: seq[seq_start:seq_end])

    feature_sets = feat.featurize_data(Xdf, learn_options, Y, gene_position, pam_audit=pam_audit, length_audit=length_audit)
    np.random.seed(learn_options['seed'])

    return Y, feature_sets, target_genes, learn_options, num_proc


def run_models(models, orders, GP_likelihoods=['gaussian', 'warped'], WD_kernel_degrees=[3],
               adaboost_learning_rates=[0.1], adaboost_num_estimators=[100], adaboost_max_depths=[3],
               learn_options_set=None, test=False, CV=True, setup_function=setup, set_target_fn=set_target, pam_audit=True, length_audit=True, return_data=False):

    '''
    CV is set to false if want to train a final model and not cross-validate, but it goes in to what
    looks like cv code
    '''


    results = {}
    assert learn_options_set is not None, "need to specify learn_options_set"
    all_learn_options = {}

    #shorten so easier to display on graphs
    feat_models_short = {'L1':"L1", 'L2':"L2", 'elasticnet':"EN", 'linreg':"LR",
                         'RandomForest': "RF",
                         'AdaBoost':"AB", 'AdaBoostClassifier':"ABClass", 'doench': 'doench',
                         "logregL1": "logregL1", "sgrna_from_doench":"sgrna_from_doench", 'SVC': 'SVC', 'xu_et_al': 'xu_et_al'}

    if not CV:
        print "Received option CV=False, so I'm training using all of the data"
        assert len(learn_options_set.keys()) == 1, "when CV is False, only 1 set of learn options is allowed"
        assert len(models) == 1, "when CV is False, only 1 model is allowed"


    for learn_options_str in learn_options_set.keys():
        # these options get augmented in setup
        partial_learn_opt = learn_options_set[learn_options_str]
        # if the model requires encoded features
        for model in models:
            # models requiring explicit featurization
            if model in feat_models_short.keys():
                for order in orders:
                    print "running %s, order %d for %s" % (model, order, learn_options_str)

                    Y, feature_sets, target_genes, learn_options, num_proc = setup_function(test=test, order=order, learn_options=partial_learn_opt, pam_audit=pam_audit, length_audit=length_audit) # TODO precompute features for all orders, as this is repated for each model
                    
                    if model == 'L1':
                        learn_options_model = L1_setup(copy.deepcopy(learn_options), set_target_fn=set_target_fn)
                    elif model == 'L2':
                        learn_options_model = L2_setup(copy.deepcopy(learn_options), set_target_fn=set_target_fn)
                    elif model == 'elasticnet':
                        learn_options_model = elasticnet_setup(copy.deepcopy(learn_options), set_target_fn=set_target_fn)
                    elif model == 'linreg':
                        learn_options_model = linreg_setup(copy.deepcopy(learn_options), set_target_fn=set_target_fn)
                    elif model == "logregL1":
                        learn_options_model = logregL1_setup(copy.deepcopy(learn_options), set_target_fn=set_target_fn)
                    elif model == 'RandomForest':
                        learn_options_model = RF_setup(copy.deepcopy(learn_options), set_target_fn=set_target_fn)
                    elif model == 'SVC':
                        learn_options_model = SVC_setup(copy.deepcopy(learn_options), set_target_fn=set_target_fn)
                    elif model == 'doench':
                        learn_options_model = doench_setup(copy.deepcopy(learn_options), set_target_fn=set_target_fn)
                    elif model == 'sgrna_from_doench':
                        learn_options_model = sgrna_from_doench_setup(copy.deepcopy(learn_options), set_target_fn=set_target_fn)
                    elif model == 'xu_et_al':
                        learn_options_model = xu_et_al_setup(copy.deepcopy(learn_options), set_target_fn=set_target_fn)
                    elif model == 'AdaBoost' or 'AdaBoostClassifier':
                        for learning_rate in adaboost_learning_rates:
                            for num_estimators in adaboost_num_estimators:
                                for max_depth in adaboost_max_depths:
                                    learn_options_model = adaboost_setup(copy.deepcopy(learn_options), learning_rate=learning_rate, num_estimators=num_estimators, max_depth=max_depth, set_target_fn=set_target_fn, model=model)
                        model_string = feat_models_short[model] + '_or%d_md%d_lr%.2f_n%d_%s' % (learn_options_set[learn_options_str]["order"], max_depth, learning_rate, num_estimators, learn_options_str)
                    if model != 'AdaBoost':
                        model_string = feat_models_short[model] + '_ord%d_%s' % (learn_options_set[learn_options_str]["order"], learn_options_str)

                    results[model_string] = pd.cross_validate(Y, feature_sets, learn_options=learn_options_model, TEST=test, CV=CV)

                    all_learn_options[model_string] = learn_options_model
            # if the model doesn't require explicit featurization
            else:
                assert setup_fn==setup, "not yet modified to handle this"
                print "running %s for %s" % (model, learn_options_str)
                Y, feature_sets, target_genes, learn_options, num_proc = setup(test=test, order=1, learn_options=partial_learn_opt, pam_audit=pam_audit, length_audit=length_audit)
                if model == 'mean':
                    learn_options_model = mean_setup(copy.deepcopy(learn_options))
                elif model == 'random':
                    learn_options_model = random_setup(copy.deepcopy(learn_options))
                elif model == 'DNN':
                    learn_options_model = DNN_setup(copy.deepcopy(learn_options))
                elif model == 'GP':
                    for likelihood in GP_likelihoods:
                        for degree in WD_kernel_degrees:
                            learn_options_model = GP_setup(copy.deepcopy(learn_options), likelihood=likelihood, degree=degree)
                            model_string = '%s_%s_degree%d_%s' % (model, likelihood, degree, learn_options_str)
                            results[model_string] = pd.cross_validate(Y, feature_sets, learn_options=learn_options_model,TEST=test, CV=CV)

                else:
                    raise NotImplementedError("model %s not supported" % model)

                # "GP" already calls pd.cross_validate() and has its own model_string, so skip this.
                if model != "GP":
                    model_string = model + '_%s' % learn_options_str
                    results[model_string] = pd.cross_validate(Y, feature_sets, learn_options=learn_options_model, TEST=test, CV=CV)

            all_learn_options[model_string] = learn_options_model

    return results, all_learn_options


def pickle_runner_results(exp_name, results, all_learn_options, relpath="/../" + "results"):
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath) + relpath
    if not os.path.exists(dname):
        os.makedirs(dname)
        print "Created directory: %s" % str(dname)
    if exp_name is None:
        exp_name = results.keys()[0]
    myfile = dname+'/'+ exp_name + '.pickle'
    with open(myfile, 'wb') as f:
        print "writing results to %s" % myfile
        pickle.dump((results, all_learn_options), f, -1)

def runner(models, learn_options, GP_likelihoods=None, orders=None, WD_kernel_degrees=None, where='local', cluster_user='fusi', cluster='RR1-N13-09-H44', test=False, exp_name = None, **kwargs):

    if where == 'local':
        results, all_learn_options = run_models(models, orders=orders, GP_likelihoods=GP_likelihoods, learn_options_set=learn_options, WD_kernel_degrees=WD_kernel_degrees, test=test, **kwargs)
        all_metrics, gene_names = azimuth.util.get_all_metrics(results, learn_options)
        azimuth.util.plot_all_metrics(all_metrics, gene_names, all_learn_options, save=True)

        # for non-local (i.e. cluster), the comparable code is in cli_run_model.py
        pickle_runner_results(exp_name, results, all_learn_options)

        return results, all_learn_options, all_metrics, gene_names

    elif where == 'cluster':
        import cluster_job

        # create random cluster directory, dump learn options, and create cluster file
        tempdir, user, clust_filename = cluster_job.create(cluster_user, models, orders, WD_kernel_degrees, GP_likelihoods, exp_name=exp_name, learn_options=learn_options, **kwargs)

        # raw_input("Submit job to HPC and press any key when it's finished: ")
        # util.plot_cluster_results(directory=tempdir)

        #stdout = tempdir + r"/stdout"
        #stderr = tempdir + r"/stderr"
        #if not os.path.exists(stdout): os.makedirs(stdout)
        #if not os.path.exists(stderr): os.makedirs(stderr)

        return tempdir, clust_filename, user#, stdout, stderr

def save_final_model_V3(filename=None, include_position=True, learn_options=None, short_name='final', pam_audit=True, length_audit=True):
    '''
    run_models(produce_final_model=True) is what saves the model
    '''
    test = False
    assert filename is not None, "need to provide filename to save final model"

    if learn_options is None:
        if include_position:
            learn_options = {"V": 3,
                        'train_genes': azimuth.load_data.get_V3_genes(),
                        'test_genes': azimuth.load_data.get_V3_genes(),
                        "testing_non_binary_target_name": 'ranks',
                        'include_pi_nuc_feat': True,
                        "gc_features": True,
                        "nuc_features": True,
                        "include_gene_position": True,
                        "include_NGGX_interaction": True,
                        "include_Tm": True,
                        "include_strand": False,
                        "include_gene_feature": False,
                        "include_gene_guide_feature": 0,
                        "extra pairs": False,
                        "weighted": None,
                        "training_metric": 'spearmanr',
                        "NDGC_k": 10,
                        "cv": "gene",
                        "include_gene_effect": False,
                        "include_drug": False,
                        "include_sgRNAscore": False,
                        'adaboost_loss' : 'ls', # main "ls", alternatives: "lad", "huber", "quantile", see scikit docs for details
                        'adaboost_alpha': 0.5, # this parameter is only used by the huber and quantile loss functions.
                        'normalize_features': False,
                        'adaboost_CV' : False
                        }
        else:
            learn_options = {"V": 3,
                'train_genes': azimuth.load_data.get_V3_genes(),
                'test_genes': azimuth.load_data.get_V3_genes(),
                "testing_non_binary_target_name": 'ranks',
                'include_pi_nuc_feat': True,
                "gc_features": True,
                "nuc_features": True,
                "include_gene_position": False,
                "include_NGGX_interaction": True,
                "include_Tm": True,
                "include_strand": False,
                "include_gene_feature": False,
                "include_gene_guide_feature": 0,
                "extra pairs": False,
                "weighted": None,
                "training_metric": 'spearmanr',
                "NDGC_k": 10,
                "cv": "gene",
                "include_gene_effect": False,
                "include_drug": False,
                "include_sgRNAscore": False,
                'adaboost_loss' : 'ls', # main "ls", alternatives: "lad", "huber", "quantile", see scikit docs for details
                'adaboost_alpha': 0.5, # this parameter is only used by the huber and quantile loss functions.
                'normalize_features': False,
                 'adaboost_CV' : False
                }

    learn_options_set = {short_name: learn_options}
    results, all_learn_options = run_models(["AdaBoost"], orders=[2], adaboost_learning_rates=[0.1],
                                            adaboost_max_depths=[3], adaboost_num_estimators=[100],
                                            learn_options_set=learn_options_set,
                                            test=test, CV=False, pam_audit=length_audit, length_audit=length_audit)
    model = results.values()[0][3][0]

    with open(filename, 'wb') as f:
        pickle.dump((model, learn_options), f, -1)

    return model


def predict(seq, aa_cut=None, percent_peptide=None, model=None, model_file=None, pam_audit=True, length_audit=False, learn_options_override=None):
    """
    Args:
        seq: numpy array of 30 nt sequences.
        aa_cut: numpy array of amino acid cut positions (optional).
        percent_peptide: numpy array of percent peptide (optional).
        model: model instance to use for prediction (optional).
        model_file: file name of pickled model to use for prediction (optional).
        pam_audit: check PAM of each sequence.
        length_audit: check length of each sequence.
        learn_options_override: a dictionary indicating which learn_options to override (optional).

    Returns: a numpy array of predictions.
    """
    # assert not (model is None and model_file is None), "you have to specify either a model or a model_file"
    assert isinstance(seq, (np.ndarray)), "Please ensure seq is a numpy array"
    assert len(seq[0]) > 0, "Make sure that seq is not empty"
    assert isinstance(seq[0], basestring), "Please ensure input sequences are in string format, i.e. 'AGAG' rather than ['A' 'G' 'A' 'G'] or alternate representations"

    if aa_cut is not None:
        assert len(aa_cut) > 0, "Make sure that aa_cut is not empty"
        assert isinstance(aa_cut, (np.ndarray)), "Please ensure aa_cut is a numpy array"
        assert np.all(np.isreal(aa_cut)), "amino-acid cut position needs to be a real number"

    if percent_peptide is not None:
        assert len(percent_peptide) > 0, "Make sure that percent_peptide is not empty"
        assert isinstance(percent_peptide, (np.ndarray)), "Please ensure percent_peptide is a numpy array"
        assert np.all(np.isreal(percent_peptide)), "percent_peptide needs to be a real number"


    if model_file is None:
        azimuth_saved_model_dir = os.path.join(os.path.dirname(azimuth.__file__), 'saved_models')
        if np.any(percent_peptide == -1) or (percent_peptide is None and aa_cut is None):
            print("No model file specified, using V3_model_nopos")
            model_name = 'V3_model_nopos.pickle'
        else:
            print("No model file specified, using V3_model_full")
            model_name = 'V3_model_full.pickle'

        model_file = os.path.join(azimuth_saved_model_dir, model_name)

    if model is None:
        with open(model_file, 'rb') as f:
            model, learn_options = pickle.load(f)
    else:
        model, learn_options = model
        
    learn_options["V"] = 2

    learn_options = override_learn_options(learn_options_override, learn_options)

    # Y, feature_sets, target_genes, learn_options, num_proc = setup(test=False, order=2, learn_options=learn_options, data_file=test_filename)
    # inputs, dim, dimsum, feature_names = pd.concatenate_feature_sets(feature_sets)

    Xdf = pandas.DataFrame(columns=[u'30mer', u'Strand'], data=zip(seq, ['NA' for x in range(len(seq))]))

    if np.all(percent_peptide != -1) and (percent_peptide is not None and aa_cut is not None):
        gene_position = pandas.DataFrame(columns=[u'Percent Peptide', u'Amino Acid Cut position'], data=zip(percent_peptide, aa_cut))
    else:
        gene_position = pandas.DataFrame(columns=[u'Percent Peptide', u'Amino Acid Cut position'], data=zip(np.ones(seq.shape[0])*-1, np.ones(seq.shape[0])*-1))

    feature_sets = feat.featurize_data(Xdf, learn_options, pandas.DataFrame(), gene_position, pam_audit=pam_audit, length_audit=length_audit)
    inputs, dim, dimsum, feature_names = azimuth.util.concatenate_feature_sets(feature_sets)
    
    #print "CRISPR"
    #pandas.DataFrame(inputs).to_csv("CRISPR.inputs.test.csv")
    #import ipdb; ipdb.set_trace()

    # call to scikit-learn, returns a vector of predicted values    
    preds = model.predict(inputs)

    # also check that predictions are not 0/1 from a classifier.predict() (instead of predict_proba() or decision_function())
    unique_preds = np.unique(preds)
    ok = False
    for pr in preds:
        if pr not in [0,1]:
            ok = True
    assert ok, "model returned only 0s and 1s"
    return preds

def override_learn_options(learn_options_override, learn_options):
    """
    override all keys seen in learn_options_override to alter learn_options
    """
    if learn_options_override is not None:
        for k in learn_options_override.keys():
            learn_options[k] = learn_options_override[k]
    return learn_options

def fill_learn_options(learn_options_used_to_fill, learn_options_with_possible_missing):
    """
    only fill in keys that are missing from learn_options from learn_options_fill
    """
    if learn_options_used_to_fill is not None:
        for k in learn_options_used_to_fill.keys():
            if not learn_options_with_possible_missing.has_key(k):
                learn_options_with_possible_missing[k] = learn_options_used_to_fill[k]
    return learn_options_with_possible_missing


def write_results(predictions, file_to_predict):
    newfile = file_to_predict.replace(".csv", ".pred.csv")
    data = pandas.read_csv(file_to_predict)
    data['predictions'] = predictions
    data.to_csv(newfile)
    print "wrote results to %s" % newfile
    return data, newfile

if __name__ == '__main__':
    #save_final_model_V3(filename='azimuth/azure_models/V3_model_full.pickle', include_position=True)

    save_final_model_V3(filename='saved_models/V3_model_nopos.pickle', include_position=False)
    save_final_model_V3(filename='saved_models/V3_model_full.pickle', include_position=True)

    # predict('GGGCCGCTGTTGCAGGTGGCGGGTAGGATC', 'sense', 1200, 30.3, model_file='../saved_models/final_model_nicolo.pickle')


    learn_options = {"V": 3,
                "train_genes": azimuth.load_data.get_V3_genes(),
                "test_genes": azimuth.load_data.get_V3_genes(),
                "target_name": 'score_drug_gene_rank',
                "testing_non_binary_target_name": 'ranks',
                'include_pi_nuc_feat': True,
                "gc_features": True,
                "nuc_features": True,
                "include_gene_position": True,
                "include_NGGX_interaction": True,
                "include_Tm": True,
                "include_strand": False,
                "include_gene_feature": False,
                "include_gene_guide_feature": 0,
                "extra pairs": False,
                "weighted": None,
                "training_metric": 'spearmanr',
                "NDGC_k": 10,
                "cv": "gene",
                "adaboost_loss" : 'ls',
                "include_gene_effect": False,
                "include_drug": False,
                "include_sgRNAscore": False,
                'adaboost_loss' : 'ls', # main "ls", alternatives: "lad", "huber", "quantile", see scikit docs for details
                'adaboost_alpha': 0.5, # this parameter is only used by the huber and quantile loss functions.
                'adaboost_CV' : False
                }

    learn_options_set = {"post bug fix":learn_options}

    #runner(['AdaBoost'], learn_options_set, orders=[2], where='local', adaboost_learning_rates=[0.1],  adaboost_max_depths=[3], adaboost_num_estimators=[100], exp_name='post-index-fix')
# #util.feature_importances(results)
