"""
from https://gist.github.com/bwhite/3726239

Information Retrieval metrics

Useful Resources:
http://www.cs.utexas.edu/~mooney/ir-course/slides/Evaluation.ppt
http://www.nii.ac.jp/TechReports/05-014E.pdf
http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
http://hal.archives-ouvertes.fr/docs/00/72/67/60/PDF/07-busa-fekete.pdf
Learning to Rank for Information Retrieval (Tie-Yan Liu)
"""
import numpy as np
import scipy as sp
import scipy.stats
import util as ut
import time

def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item

    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).

    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75

    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def r_precision(r):
    """Score is precision after all relevant documents have been retrieved

    Relevance is binary (nonzero is relevant).

    >>> r = [0, 0, 1]
    >>> r_precision(r)
    0.33333333333333331
    >>> r = [0, 1, 0]
    >>> r_precision(r)
    0.5
    >>> r = [1, 0, 0]
    >>> r_precision(r)
    1.0

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        R Precision
    """
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.
    return np.mean(r[:z[-1] + 1])


def precision_at_k(r, k):
    """Score is precision @ k

    Relevance is binary (nonzero is relevant).

    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k


    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        Precision @ k

    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)

    Relevance is binary (nonzero is relevant).

    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision

    Relevance is binary (nonzero is relevant).

    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666

    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)

    Relevance is positive real values.  Can use binary
    as the previous methods.

    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]

    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)

    Relevance is positive real values.  Can use binary
    as the previous methods.

    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]

    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

## ------------------------------------------------------------------------------------
## custom stuff from us to avoid problem with ties

def ndcg_at_k_ties(labels, predictions, k, method=0, normalize_from_below_too=False, theta=None):
    '''
    See 2008 McSherry et al on how to efficiently compute NDCG with ties
    labels are ground truth

    if k=None then k gets set to len(labels)

    labels and predictions get flattened here

    set normalize_from_below_too=False for conventional ndcg_at_k_ties, but note this will only
    ensure the max is 1, not that the min is zero. to get that added guarantee, set this argument to True
    '''

    if isinstance(labels, list):
        labels = np.array(labels)
    if isinstance(predictions, list):
        predictions = np.array(predictions)

            
    assert len(labels.shape)==1 or np.min(labels.shape)==1, "should be 1D array or equivalent"
    assert len(predictions.shape)==1 or np.min(predictions.shape)==1, "should be 1D array or equivalent"
        
    labels = labels.flatten()
    predictions = predictions.flatten()

    assert np.all(labels.shape == predictions.shape), "labels and predictions should have the same shape"
        
    if k is None:
        k = len(labels)

    labels = labels.copy()

    dcg = dcg_at_k_ties(labels, predictions, k, method=method, theta=theta)
        
    dcg_max = dcg_at_k_ties(labels, labels, k, method, theta=theta)
    # NOTE: I have checked that dcg_at_k_ties and dcg_at_k match when there are no ties, or ties in the labels

    
    if normalize_from_below_too:
        dcg_min = dcg_at_k_ties(np.sort(labels)[::-1], np.sort(predictions), k, method, theta=theta)
    else:
        dcg_min = 0
    numerator = (dcg - dcg_min)
    assert numerator > -1e-5
    numerator = np.max((0, numerator))
    ndcg = numerator / (dcg_max - dcg_min)            
    assert ndcg <= 1.0 and ndcg >= 0.0, "ndcg=%f should be in [0,1]" % ndcg
    if not dcg_max: 
        ndcg = 0.
    return ndcg

def dcg_helper(discount_factors, gain, k, labels, method, predictions):
    #step through, in current order (of decreasing predictions), accumulating tied gains (which may be singletons)
    ii = 0
    dcg = 0.0
    while (ii < k):
        current_pred = predictions[ii]
        current_gain = gain(labels[ii], method)
        # intializing the tied cumulative variables
        cum_tied_gain = current_gain
        cum_tied_disc = discount_factors[ii]
        num_ties = 1
        ii += 1
        # count number of ties in predictions
        while (ii<len(predictions) and predictions[ii]==current_pred):  #while tied
            num_ties += 1.0
            cum_tied_gain += gain(labels[ii], method)
            if ii < k: cum_tied_disc += discount_factors[ii]
            ii += 1
        #if len(np.unique(predictions))==1:  import ipdb; ipdb.set_trace()
        avg_gain = cum_tied_gain/num_ties
        dcg += avg_gain*cum_tied_disc
        assert not np.isnan(dcg), "found nan dcg"
    return dcg

def dcg_at_k_ties(labels, predictions, k, method=0, theta=None):
    '''
    See 2008 McSherry et al on how to efficiently compute NDCG (method=0 here) with ties (in the predictions)
    'labels' are what the "ground truth" judges assign
    'predictions' are the algorithm predictions corresponding to each label
    Also, http://en.wikipedia.org/wiki/Discounted_cumulative_gain for basic defns
    '''
    assert isinstance(predictions,np.ndarray)
    assert len(labels) == len(predictions), "labels and predictions should be of same length"
    assert k <= len(labels), "k should be <= len(labels)"

    # order both labels and preds so that they are in order of decreasing predictive score
    sorted_ind = np.argsort(predictions)[::-1]
    predictions = predictions[sorted_ind]
    labels = labels[sorted_ind]

    def gain(label, method):
        if method==0:
            return label
        elif method==1:
            return 2**label-1.0
        elif method==2 or method==3 or method==4:
            return label
        else:
            raise NotImplementedError()

    if method==0:
        discount_factors = get_discount_factors(len(labels), discount='log2')
    elif method==1:
        raise Exception("need to implement: log_2(i+1)")
    elif method==2:
        discount_factors = get_discount_factors(len(labels), discount='linear')
    elif method==3:
        discount_factors = get_discount_factors(len(labels), discount='combination')
    elif method==4:
        assert theta is not None, "need to specify theta or theta"
        discount_factors = get_discount_factors(len(labels), discount='1/rtheta', theta=theta)

    else:
        raise NotImplementedError()

    assert len(discount_factors) == len(labels), "discount factors has wrong length"

    dcg = dcg_helper(discount_factors, gain, k, labels, method, predictions)
    assert not np.isnan(dcg), "found nan dcg"
    
    return dcg

def get_discount_factors(num_labels, discount='log2', theta=None):
    ii_range = np.arange(num_labels) + 1

    if discount == 'log2':
        discount_factors = np.concatenate((np.array([1.0]), 1.0/np.log2(ii_range[1:])))
    elif discount == 'linear':
        discount_factors = -ii_range/float(num_labels) + 1.0
    elif discount == 'combination':
        l2 = np.concatenate((np.array([1.0]), 1.0/np.log2(ii_range[1:])))
        linear = -ii_range/float(num_labels) + 1.0
        discount_factors = np.max((l2,linear), axis=0)
    elif discount == '1/rtheta':
        discount_factors = 1./(ii_range**theta)
    else:
        raise NotImplementedError

    return discount_factors

def rank_data(r, rground):
    # we checked this heavily, and is correct, e.g. rground will go from largest rank to smallest
    r = sp.stats.mstats.rankdata(r)
    rground = sp.stats.mstats.rankdata(rground)
    assert np.sum(r)==np.sum(rground), "ranks should add up to the same"
    return r, rground

#todo: look at rank_data for bug, look at highest alpha predictions for ties
def dcg_alt(relevances, rank=20):
    relevances = np.asarray(relevances)[:rank]
    n_relevances = len(relevances)
    if n_relevances == 0:
        return 0.
    discounts = np.log2(np.arange(n_relevances) + 2)
    return np.sum(relevances / discounts)

def ndcg_alt(relevances, rank=20):
    best_dcg = dcg_alt(sorted(relevances, reverse=True), rank)
    if best_dcg == 0:
        return 0.
    return dcg_alt(relevances, rank) / best_dcg

def ndcg_bootstrap_test(preds1, preds2, true_labels, num_bootstrap, method, k, normalize_from_below_too, seed = "78923"):
    """
    Basic idea: use bootstrap to get the standard deviation of the difference in NDCG, and then create a z-statistic,
    z = (ndcg1-ndcg2)/std(ndcg1-ndcg2), and then compute a p-value under the assumption that this is normally distributed.
    Robin et al . BMC Bioinformatics 2011, 12:77
    http://www.biomedcentral.com/1471-2105/12/77
    """

    return pv

def ndcg_at_k_swap_perm_test(preds1, preds2, true_labels, nperm, method, k, normalize_from_below_too, theta=None, balance_zeros=True):
            
        # pVal is the probability that we would observe as big an AUC diff as we
        # did if the ROC curves were drawn from the null hypothesis (which is that 
        # one model does not perform better than the other)
        #
        # null hypothesis is that the prediction ranking are the same, so we exchange a random 
        # number of them with each other. 
        #
        # see ndcg_at_k_ties for all but the first four parameters
        #
        # balance_zeros = True means that when we swap a zero for a non-zero value, we will also do a reverse swap
        #
        # this is a two-sided test, but since it is a symmetric null distribution, one should
        # be able to divide the p-value by 2 to get the one-sided version (but think this through before using)
        
        if isinstance(preds1, list):
            preds1 = np.array(preds1)
        else:
            preds1 = preds1.flatten()

        if isinstance(preds2, list):
            preds2 = np.array(preds2)
        else:
            preds2 = preds2.flatten()

        if isinstance(true_labels, list):
            true_labels = np.array(true_labels)
        else:
            true_labels = true_labels.flatten()

        assert len(preds1) == len(preds2), "need same number of preditions from each model"
        assert len(preds1) == len(true_labels), "need same number of preditions in truth and predictions"
        N = len(preds1)

        # re-sort all by truth ordering so that when swap they are aligned
        sorted_ind = np.argsort(true_labels)[::-1]
        true_labels = true_labels[sorted_ind]
        preds1 = preds1[sorted_ind]
        preds2 = preds2[sorted_ind]

        ranks1 = sp.stats.mstats.rankdata(preds1)
        ranks2 = sp.stats.mstats.rankdata(preds2)
        
        ndcg1 = ndcg_at_k_ties(true_labels, ranks1, k=k, method=method, normalize_from_below_too=normalize_from_below_too, theta=theta)
        ndcg2 = ndcg_at_k_ties(true_labels, ranks2, k=k, method=method, normalize_from_below_too=normalize_from_below_too, theta=theta)

        real_ndcg_diff = {}
        perm_ndcg_diff = {}

        real_ndcg_diff = np.abs(ndcg1 - ndcg2)                
        perm_ndcg_diff = np.nan*np.zeros(nperm)
            
        if False:#np.all(preds1 == preds2):
            pval[theta] = 1.0            
        else:                    
            zero_ind = true_labels == 0
            assert np.sum(zero_ind) < len(zero_ind), "balancing assumes there are more zeros than ones"

            for t in range(nperm):
                pair_ind_to_swap = np.random.rand(N) < 0.5

                ranks1_perm = ranks1.copy();
                ranks1_perm[pair_ind_to_swap] = ranks2[pair_ind_to_swap]

                ranks2_perm = ranks2.copy();
                ranks2_perm[pair_ind_to_swap] = ranks1[pair_ind_to_swap]

                ndcg1_perm = ndcg_at_k_ties(true_labels, ranks1_perm, k=k, method=method, normalize_from_below_too=normalize_from_below_too, theta=theta)
                ndcg2_perm = ndcg_at_k_ties(true_labels, ranks2_perm, k=k, method=method, normalize_from_below_too=normalize_from_below_too, theta=theta)

                for theta in theta:
                    tmp_diff = np.abs(ndcg1_perm[theta] - ndcg2_perm[theta])
                    perm_ndcg_diff[theta][t] = tmp_diff

            pval = {}            
            
            num_stat_greater = np.max((((perm_ndcg_diff > real_ndcg_diff).sum() + 1), 1.0))
            pval = num_stat_greater / nperm

        if False:
            plt.figure();
            plt.plot(np.sort(perm_ndcg_diff), '.')
            plt.plot(real_ndcg_diff*np.ones(perm_ndcg_diff.shape), 'k-')
            plt.show()
                        
        return pval, real_ndcg_diff, perm_ndcg_diff, ndcg1, ndcg2

if __name__ == "__main__":
    import cPickle as pickle
    import matplotlib.pyplot as plt
    import elevation.metrics
    import corrstats

    simulated_data = True
    permute_real_data = True
        
    T = 1000
    allp = np.nan*np.ones(T)

    nperm = 100
    #method = 4; normalize_from_below_too = True; 
    
    #theta_range = np.logspace(np.log10(0.01), np.log10(1.0), 3)  # Nicolo uses 10, so I grab the extremes and middle
    #theta_range = np.array([0.01])
    #weights = np.logspace(np.log10(0.0001), np.log10(10), 3); 
    #weights = np.array([100.0])
    weights = np.array([0.001])
    theta_range = weights# just to make life easier

    
    # only for simulated data
    N = 100
    frac_zeros = 0
    
    k = None

    allp = np.nan*np.zeros((len(theta_range) + 1, T))

    if not simulated_data:
        print "loading up saved data..." # two-fold CV data from CRISPR off-target GUIDE-SEQ
        with open(r'\\nerds5\kevin\from_nicolo\gs.pickle','rb') as f:  predictions, truth_all = pickle.load(f)
        print "done."
        N = len(truth_all[0])
            
    for t in range(T):

        # totally simulated
        if simulated_data:
            truth = np.random.rand(N)
            zero_ind = np.random.rand(N) < frac_zeros
            truth[zero_ind] = 0
            pred1 = np.random.rand(N)
            pred2 = np.random.rand(N)
        else:                                
            fold = 0
            truth = truth_all[fold]
            pred1 = predictions["CFD"][fold]
            pred2 = predictions["product"][fold]
                        
            if permute_real_data:
                truth = np.random.permutation(truth)

        t0 = time.time()        
        #pval, real_ndcg_diff,  perm_ndcg_diff, ndcg1, ndcg2 = ndcg_at_k_swap_perm_test(pred1, pred2, truth, nperm, method, k, normalize_from_below_too, theta_range=theta_range)        
        for i, w in enumerate(weights):
            weights_array = truth.copy()
            weights_array += w

            #corr0 = elevation.metrics.spearman_weighted(truth, pred1, w=weights_array)
            #corr1 = elevation.metrics.spearman_weighted(truth, pred2, w=weights_array)
            #corr01 = elevation.metrics.spearman_weighted(pred1, pred2, w=weights_array)
            #n0 = len(truth)        
            #t2, pvaltmp = corrstats.dependent_corr(corr0, corr1, corr01, n0, twotailed=True, method="steiger")

            pvaltmp, real_corr_diff, perm_corr_diff, corr1, corr2 = elevation.spearman_weighted_swap_perm_test(pred1, pred2, truth, nperm, weights_array)
                                                        
            allp[i, t] = pvaltmp
            t1 = time.time()

        #for i, theta in enumerate(theta_range.tolist() + ["all"]):
        #    print "%d, theta=%s) ndcg1=%f, ndcg2=%f, ndcg_diff=%f, p=%f, elapsed time=%f minutes, smallest_p=%f" % (t, str(theta), ndcg1[theta], ndcg2[theta], real_ndcg_diff[theta], pval[theta], (t1-t0)/60, 1.0/nperm)        
        #    allp[i, t] = pval[theta]
        #print "---------------"
        
    #for i, theta in enumerate(theta_range.tolist() + ["all"]):
    for i, theta in enumerate(theta_range.tolist()):
        #mytitle = "Norm. hist p-values nDCG\n %d null samples, w %d perm and N=%d, theta=%s" % (T, nperm, N, str(theta))
        mytitle = "Norm. hist p-values Steiger w weighted Spearman\n %d null samples, N=%d, weight=%s" % (T, N, str(theta))
        ut.qqplotp(allp[i,:], dohist=True, numbins=10, figsize=[6,6], title=mytitle, markersize=5)
        plt.show()
    
    #save_tmp_results = r'D:\Source\CRISPR\elevation\pickles\tmp.ndcg.stat.calibration.p'
    #pickle.dump([theta_range, allp, pval, real_ndcg_diff, perm_ndcg_diff, ndcg1, ndcg2], open(save_tmp_results, "wb" ))
    #[theta_range, allp, pval, real_ndcg_diff, perm_ndcg_diff, ndcg1, ndcg2] = pickle.load(open(save_tmp_results, "rb" ))


    import ipdb; ipdb.set_trace()


    # # e.g. where all predictions are the same
    # labels = np.arange(30)
    # predictions = np.ones(30)

    # discount_factors = get_discount_factors(labels)
    # avg_label = np.mean(labels)
    # avg_label_vec = avg_label*np.ones((len(labels),1))

    # for k in range(10):
    #     # one way
    #     dcg1 = np.dot(discount_factors[0:k,None].T, avg_label_vec[0:k])[0][0]
    #     # another way
    #     dcg2 = np.sum(discount_factors[0:k])*avg_label
    #     # using our function
    #     dcg3 = dcg_at_k_ties(labels,predictions,k)

    #     print "%f, %f, %f" % (dcg1, dcg2, dcg3)
    #     assert(np.abs(dcg1 - dcg2) < 1e-8)
    #     assert(np.abs(dcg2 - dcg3) < 1e-8)
    # print "check out ok for case with all ties in predictions"

    truth = np.array([3, 4, 2, 1, 0, 0, 0])
    pred1 = np.array([3, 4, 2, 1, 0, 0, 0])
    pred2 = np.array([2, 1, 3, 4, 5, 6, 7])

    truth3 = np.array([3, 4, 2, 1, 0, 0, 0])
    truth4 = np.zeros(7); truth4[0] = 1
    pred3 = np.array([2, 1, 3, 4, 5, 6, 7])*10
    pred4 = np.array([4, 3, 2, 1, 0, 0, 0])
    pred5 = np.array([4, 3, 1, 2, 0, 0, 0])

    nperm = 1000
    method = 4; theta = 0.5; normalize_from_below_too = True
    k = len(pred3)

    pval, real_ndcg_diff,  perm_ndcg_diff, ndcg1, ndcg2 = ndcg_at_k_swap_perm_test(pred1, pred2, truth, nperm, method, k, normalize_from_below_too, theta=theta)
    print "ndcg1=%f, ndcg2=%f, ndcg_diff=%f, p=%f" % (ndcg1, ndcg2, real_ndcg_diff, pval)
    
    pval, real_ndcg_diff,  perm_ndcg_diff, ndcg1, ndcg2 = ndcg_at_k_swap_perm_test(pred1, pred1, truth, nperm, method, k, normalize_from_below_too, theta=theta)    
    print "ndcg1=%f, ndcg2=%f, ndcg_diff=%f, p=%f" % (ndcg1, ndcg2, real_ndcg_diff, pval)

    pval, real_ndcg_diff,  perm_ndcg_diff, ndcg1, ndcg2 = ndcg_at_k_swap_perm_test(pred1, pred4, truth, nperm, method, k, normalize_from_below_too, theta=theta)    
    print "ndcg1=%f, ndcg2=%f, ndcg_diff=%f, p=%f" % (ndcg1, ndcg2, real_ndcg_diff, pval)

    pval, real_ndcg_diff,  perm_ndcg_diff, ndcg1, ndcg2 = ndcg_at_k_swap_perm_test(pred1, pred5, truth, nperm, method, k, normalize_from_below_too, theta=theta)    
    print "ndcg1=%f, ndcg2=%f, ndcg_diff=%f, p=%f" % (ndcg1, ndcg2, real_ndcg_diff, pval)

    import ipdb; ipdb.set_trace()


    #print ndcg_at_k_ties(truth, truth, k, method=0, normalize_from_below_too=True)
    #print ndcg_at_k_ties(truth, pred2, k, method=0, normalize_from_below_too=True)
    #print ndcg_at_k_ties(truth, pred3, k, method=0, normalize_from_below_too=True)
    #print ndcg_at_k_ties(truth3, pred3, k, method=3, normalize_from_below_too=True)
    print ndcg_at_k_ties(truth4, pred2, k, method=3, normalize_from_below_too=True)
        
    print ndcg_alt(truth[np.argsort(pred2)[::-1]], 5)
    print ndcg_at_k(truth[np.argsort(pred2)[::-1]], 5, method=1)
    print ndcg_at_k(truth[np.argsort(pred2)[::-1]], 5, method=0)

    print ndcg_at_k_ties(truth, pred2, 5, method=1)
    print ndcg_at_k_ties(truth, pred2, 5, method=0)
