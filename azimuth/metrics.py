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

def ndcg_at_k_ties(labels, predictions, k, method=0):
    '''
    See 2008 McSherry et al on how to efficiently compute NDCG with ties
    labels are ground truth
    '''

    labels = labels.copy()

    # NDCG requires the ground truth labels to be > 0 (the predictions can be negative)
    # if np.any(labels < 0):
    #     labels += np.abs(labels.min() + 1e-5)

    # this is the one with ties:
    dcg = dcg_at_k_ties(labels, predictions, k, method=method)

    # this one is the vanilla computation that ignores ties (and should match dcg_at_k_ties when no ties are present):
    # highest-to-lowest of the true labels (i.e. best first)
    dcg_max = dcg_at_k(sorted(labels, reverse=True), k, method)
    # NOTE: I have checked that dcg_at_k_ties and dcg_at_k match when there are no ties, or ties in the labels

    ndcg = dcg / dcg_max

    if not dcg_max:
        return 0.
    #assert ndcg >= (0.0 - 1e-8) and ndcg <= (1.0 + 1e-8), "ndcg should be between 0 and 1"

    return ndcg

def dcg_at_k_ties(labels, predictions, k, method=0):
    '''
    See 2008 McSherry et al on how to efficiently compute NDCG (method=0 here) with ties
    labels are what the "ground truth" judges assign
    predictions are the algorithm predictions corresponding to each label
    Also, http://en.wikipedia.org/wiki/Discounted_cumulative_gain for basic defns
    '''

    assert len(labels) == len(predictions), "labels and predictions should be of same length"
    assert k <= len(labels), "k should be <= len(labels)"
    # order both labels and preds so that they are in order of decreasing predictive score
    sorted_ind = np.argsort(predictions)[::-1]
    predictions = predictions[sorted_ind]
    labels = labels[sorted_ind]

    def gain(label, method=0):
        if method==0:
            return label
        elif method==1:
            return 2**label-1.0

    if method==0:
        discount_factors = get_discount_factors(labels)
    elif method==1:
        raise Exception("need to implement: log_2(i+1)")
    assert len(discount_factors) == len(labels), "discount factors has wrong length"

    #step through, in current order (of decreasing predictions), accumulating tied gains (which may be singletons)
    ii = 0
    dcg = 0.0
    while (ii < k):
        current_pred = predictions[ii]
        current_gain = gain(labels[ii])
        # intializing the tied cumulative variables
        cum_tied_gain = current_gain
        cum_tied_disc = discount_factors[ii]
        num_ties = 1
        ii += 1
        # count number of ties
        while (ii<len(predictions) and predictions[ii]==current_pred):  #while tied
            num_ties += 1.0
            cum_tied_gain += gain(labels[ii])
            if ii < k: cum_tied_disc += discount_factors[ii]
            ii += 1
        #if len(np.unique(predictions))==1:  import ipdb; ipdb.set_trace()
        avg_gain = cum_tied_gain/num_ties
        dcg += avg_gain*cum_tied_disc
        assert not np.isnan(dcg), "found nan dcg"
    assert not np.isnan(dcg), "found nan dcg"
    return dcg

def get_discount_factors(labels):
    ii_range = np.arange(len(labels)) + 1
    discount_factors = np.concatenate((np.array([1.0]), 1.0/np.log2(ii_range[1:])))
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

if __name__ == "__main__":
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

    print ndcg_alt(truth[np.argsort(pred2)[::-1]], 5)
    print ndcg_at_k(truth[np.argsort(pred2)[::-1]], 5, method=1)
    print ndcg_at_k(truth[np.argsort(pred2)[::-1]], 5, method=0)

    print ndcg_at_k_ties(truth, pred2, 5, method=1)
    print ndcg_at_k_ties(truth, pred2, 5, method=0)
