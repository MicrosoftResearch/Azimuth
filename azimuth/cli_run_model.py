import os
from model_comparison import *
import argparse

# command-line version of model_comparison.py (see that file for more options?)

if __name__ == '__main__':
    usage = 'usage: cli_run_model.py model'
    parser = argparse.ArgumentParser(usage=usage)
    # model e.g. L1, or GP
    parser.add_argument('model')
    parser.add_argument('--test', dest='test', action='store_true', default=False)
    parser.add_argument('--order', dest='order', action='store', type=int, default=1)
    # applies only to GP:
    parser.add_argument('--likelihood', dest='likelihood', action='store', type=str, default='gaussian')
    # applies only to GP:
    parser.add_argument('--weighted-degree', dest='WD', action='store', type=int, default=3)

    parser.add_argument('--adaboost-learning-rate', dest='adaboost_lr', action='store', type=float, default=0.1)
    parser.add_argument('--adaboost-max-depth', dest='adaboost_max_depth', action='store', type=int, default=3)
    parser.add_argument('--adaboost-num-estimators', dest='adaboost_num_estimators', action='store', type=int, default=100)
    parser.add_argument('--adaboost-CV', dest='adaboost_CV', action='store_true', default=False)

    parser.add_argument('--output-dir', dest='output_dir', action='store', type=str, default='./')

    parser.add_argument('--exp-name', dest='exp_name', action='store', type=str, default=None)

    options = parser.parse_args()

    # store current directory
    cur_dir = os.getcwd()
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    # change directory to script directory so that relative paths work
    os.chdir(dname)

    with open(options.output_dir + '/learn_options.pickle', 'rb') as f:
        learn_options = pickle.load(f)

    results, all_learn_options = run_models([options.model], learn_options_set=learn_options, orders=[options.order], test=options.test,
                                            GP_likelihoods=[options.likelihood], WD_kernel_degrees=[options.WD],
                                            adaboost_num_estimators=[options.adaboost_num_estimators],
                                            adaboost_max_depths=[options.adaboost_max_depth], adaboost_learning_rates=[options.adaboost_lr],
                                            adaboost_CV=options.adaboost_CV)

    if options.exp_name is None:
        exp_name = results.keys()[0]
    else:
        exp_name = results.keys()[0]

    os.chdir(cur_dir)

    with open(options.output_dir+'/' + exp_name + '.pickle', 'wb') as f:
        pickle.dump((results, all_learn_options), f)
