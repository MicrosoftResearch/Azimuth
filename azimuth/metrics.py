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

def ndcg_at_k_ties(labels, predictions, k, method, normalize_from_below_too=False):
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

    # NDCG requires the ground truth labels to be > 0 (the predictions can be negative)
    # if np.any(labels < 0):
    #     labels += np.abs(labels.min() + 1e-5)

    # this is the one with ties:
    dcg = dcg_at_k_ties(labels, predictions, k, method=method)

    # this one is the vanilla computation that ignores ties (and should match dcg_at_k_ties when no ties are present):
    # highest-to-lowest of the true labels (i.e. best first)
    # dcg_max = dcg_at_k(sorted(labels, reverse=True), k, method)
    dcg_max = dcg_at_k_ties(labels, labels, k, method)
    # NOTE: I have checked that dcg_at_k_ties and dcg_at_k match when there are no ties, or ties in the labels

    if not normalize_from_below_too:
        ndcg = dcg / dcg_max
    else:        
        dcg_min = dcg_at_k_ties(np.sort(labels)[::-1], np.sort(predictions), k, method)        
        ndcg = (dcg - dcg_min) / (dcg_max - dcg_min)
        assert ndcg <= 1.0 and ndcg >= 0.0, "ndcg=%f should be in [0,1]" % ndcg
                
    if not dcg_max:
        return 0.
    #assert ndcg >= (0.0 - 1e-8) and ndcg <= (1.0 + 1e-8), "ndcg should be between 0 and 1"

    return ndcg

def dcg_at_k_ties(labels, predictions, k, method=0):
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
        elif method==2 or method==3:
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

        
    assert len(discount_factors) == len(labels), "discount factors has wrong length"

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
    assert not np.isnan(dcg), "found nan dcg"
    return dcg

def get_discount_factors(num_labels, discount='log2'):
    ii_range = np.arange(num_labels) + 1

    if discount == 'log2':
        discount_factors = np.concatenate((np.array([1.0]), 1.0/np.log2(ii_range[1:])))
    elif discount == 'linear':
        discount_factors = -ii_range/float(num_labels) + 1.0
    elif discount == 'combination':
        l2 = np.concatenate((np.array([1.0]), 1.0/np.log2(ii_range[1:])))
        linear = -ii_range/float(num_labels) + 1.0
        discount_factors = np.max((l2,linear), axis=0)

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

def swap_perm_test_ndcg(preds1, preds2, true_labels, num_perm, method, k, normalize_from_below_too, seed = "78923"):
            
        # pVal is the probability that we would observe as big an AUC diff as we
        # did if the ROC curves were drawn from the null hypothesis (which is that 
        # one model does not perform better than the other)
        #
        # think of it this way: we want a null distribution which says that there
        # is no difference between the ROCs. Since an AUC difference
        # cannot arise from any entries in the ordered lists that match up, we can
        # ignore these (though we could include them as well, but it would fall out
        # in the wash). So instead, we assume (know) that all the information in the 
        # differences in AUCs is contained in the mismatched pairs, and we want to
        # destroy this info for the null, so we swap the values between the two
        # models. However, we want to keep the number of positive/negative samples
        # the same, so when we swap one pair, we must also swap another in the
        # other direction.            
        #
        # Adapted from an AUC test which one can show matches the analytical version
        # when there are no ties.
        #
        # see ndcg_at_k_ties for all but the first four parameters
        #
        # this is a two-sided test, but since it is a symmetric null distribution, one should
        # be able to divide the p-value by 2 to get the one-sided version (but think this through before using)
        
        ranks1 = sp.stats.mstats.rankdata(preds1)
        ranks2 = sp.stats.mstats.rankdata(preds2)

        ndcg1 = ndcg_at_k_ties(true_labels, ranks1, k=k, method=method, normalize_from_below_too=normalize_from_below_too)
        ndcg2 = ndcg_at_k_ties(true_labels, ranks2, k=k, method=method, normalize_from_below_too=normalize_from_below_too)

        real_ndcg_diff = np.abs(ndcg1 - ndcg2)

        perm_ndcg = np.nan*np.zeros(num_perm)
        
        sorted_ind = np.argsort(predictions)[::-1]
        predictions = predictions[sorted_ind]
               
       
            #if (orderedTargets1.Length == 0) throw new Exception("empty ROCs given as input");

            #DoubleArray targetDiff = orderedTargets1 - orderedTargets2;
            #IntArray posDiffInd = targetDiff.Find(v => v > 0);
            #IntArray negDiffInd = targetDiff.Find(v => v < 0);

            #int numPos = posDiffInd.Length;
            #int numNeg = negDiffInd.Length;
            #//Helper.CheckCondition(Math.Abs(numPos - numNeg) <= 1, "don't think this should happen when non-truncated ROCs are used");

            #double pVal;
            #if (numPos == 0 || numNeg == 0)
            #{//ROCs are identical
            #    pVal = 1;
            #    return pVal;
            #}

            #//bug checking:
            #int numPos1 = roc1._classLabels.ElementEQ(POS_LABEL).Sum();
            #int numNeg1 = roc1._classLabels.ElementEQ(NEG_LABEL).Sum();
            #int numPos2 = roc2._classLabels.ElementEQ(POS_LABEL).Sum();
            #int numNeg2 = roc2._classLabels.ElementEQ(NEG_LABEL).Sum();

            #//these won't be true if we're using a truncated ROC test
            #//Helper.CheckCondition(numPos1 == numPos2, "rocs must correspond to the same labelled data (i.e, # of 1/0 must be the same)");
            #//Helper.CheckCondition(numNeg1 == numNeg2, "rocs must correspond to the same labelled data (i.e, # of 1/0 must be the same)");

            #//for bug checking, keep track of avg number of swaps:
            #//DoubleArray numPairedSwaps = new DoubleArray(1, numTrial);
            #//DoubleArray auc1tmp= new DoubleArray(1, numTrial);
            #//DoubleArray auc2tmp = new DoubleArray(1, numTrial);

            #int numPairs = Math.Min(numPos, numNeg);

            #//for (int t = 0; t < numTrial; t++)
            #Parallel.For(0, numTrial, parallelOptions, t =>
            #{
            #    //Helper.CheckCondition((orderedTargets1 - orderedTargets2).Abs().Sum() != 0);

            #    Random myRand = SpecialFunctions.GetMachineInvariantRandomFromSeedAndNullIndex(randomStringSeed, t + 1);

            #    //randomly pair up each positive mismatch with each negative mismatch
            #    IntArray posIndRand = posDiffInd.RandomPermutation(myRand);
            #    IntArray negIndRand = negDiffInd.RandomPermutation(myRand);

            #    //throw new NotImplementedException("Change to GetSlice()");
            #    IntArray possiblePosPairs = posIndRand.GetColSlice(0, 1, numPairs - 1);//ColSlice(0, 1, numPairs - 1);
            #    IntArray possibleNegPairs = negIndRand.GetColSlice(0, 1, numPairs - 1); //ColSlice(0, 1, numPairs - 1);
            #    IntArray possiblePairs = ShoUtils.CatArrayCol(possiblePosPairs, possibleNegPairs).T;
            #    //Helper.CheckCondition(possiblePairs.size1 == numPairs, "something went wrong");

            #    //randomly pick each pair with prob=0.5 to include in the swap:
            #    DoubleArray randVec = (new DoubleArray(1, numPairs)).FillRandUseSeed(myRand);

            #    IntArray pairsOfPairsToBothSwapInd = randVec.Find(v => v >= 0.5);
            #    List<int> listInd = pairsOfPairsToBothSwapInd.T.ToListOrEmpty();
            #    //numPairedSwaps[t] = listInd.Count;

            #    DoubleArray newTarg1 = DoubleArray.From(orderedTargets1);
            #    DoubleArray newTarg2 = DoubleArray.From(orderedTargets2);

            #    if (listInd.Count > 0)
            #    {
            #        //throw new NotImplementedException("Change to GetSlice()");
            #        List<int> swapThesePairs = possiblePairs.GetRows(pairsOfPairsToBothSwapInd.T.ToList()).ToVector().ToList();

            #        //swap the chosen pairs with a 1-x
            #        //Helper.CheckCondition((newTarg1.GetColsE(swapThesePairs) - newTarg2.GetColsE(swapThesePairs)).Abs().Sum() == swapThesePairs.Count); 

            #        //throw new NotImplementedException("Change to SetSlice()");
            #        newTarg1.SetCols(swapThesePairs, 1 - newTarg1.GetCols(swapThesePairs));//.GetColsE(swapThesePairs));
            #        newTarg2.SetCols(swapThesePairs, 1 - newTarg2.GetCols(swapThesePairs));//GetColsE(swapThesePairs));

            #        //newTarg1.WriteToCSVNoDate("newTarg1Swapped");
            #        //newTarg2.WriteToCSVNoDate("newTarg2Swapped");

            #        //Helper.CheckCondition(newTarg1.Sum() == orderedTargets1.Sum());
            #        //Helper.CheckCondition((newTarg1 - newTarg2).Abs().Sum() == numPos + numNeg);
            #        //Helper.CheckCondition((newTarg1 - newTarg2).Find(v => v != 0).Length == numPos + numNeg);
            #        //Helper.CheckCondition((newTarg1 - orderedTargets1).Abs().Sum() == swapThesePairs.Count);
            #        //Helper.CheckCondition((newTarg2 - orderedTargets2).Abs().Sum() == swapThesePairs.Count);
            #        //Helper.CheckCondition((orderedTargets1 - orderedTargets2).Abs().Sum() != 0);
            #    }

            #    double AUC1, AUC2;

            #    if (maxFPR == 1)
            #    {
            #        //do it the cheap way
            #        AUC1 = ComputeAUCfromOrderedList(newTarg1);
            #        AUC2 = ComputeAUCfromOrderedList(newTarg2);
            #    }
            #    else
            #    {
            #        //do it with manual integration, the more expensive way
            #        AUC1 = new ROC(newTarg1, roc1._classProbs, roc1._lowerScoreIsMoreLikelyClass1, maxFPR, true)._AucAtMaxFpr;
            #        AUC2 = new ROC(newTarg2, roc2._classProbs, roc2._lowerScoreIsMoreLikelyClass1, maxFPR, true)._AucAtMaxFpr;
            #    }

            #    //auc1tmp[t] = AUC1;
            #    //auc2tmp[t] = AUC2;

            #    //permDiffs[t] = Math.Abs(AUC1 - AUC2);
            #    permDiffs[t] = (AUC1 - AUC2);

            #}
            #);

            #//double markerSize = 0.1;
            #//ShoUtils.MakePlotAndView(permDiffs, "permDiffs", false, markerSize, ".");
            #//ShoUtils.MakePlotAndView(numPairedSwaps, "numPairedSwaps", false, markerSize, "*");
            #permDiffs = permDiffs.Map(v => Math.Abs(v));
            #//debugging:
            #//permDiffs.WriteToCSVNoDate("permDiffs");
            #//numPairedSwaps.WriteToCSVNoDate("numPairedSwapsC#");


            #double pseudoCount = 1;
            #pVal = (pseudoCount + (double)(permDiffs >= realAUCdiff).Sum()) / (double)numTrial;
            #pVal = Math.Min(pVal, 1);

            #//ShoUtils.MakePlotAndView((auc1tmp-auc2tmp).Map(v=>Math.Abs(v)), "auc1-auc2", false, 0.2, ".");

            #//System.Console.WriteLine("Avg # swaps: " + numPairedSwaps.Mean() + " ( of " + numPairs + " total), numGreaterSwaps=" + (double)(permDiffs >= realAUCdiff).Sum() + ", p=" + pVal + ", realAUCdiff=" + String.Format("{0:0.00000}", realAUCdiff));


            #return pVal;


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
    
    truth3 = np.array([3, 4, 2, 1, 0, 0, 0])
    truth4 = np.zeros(7); truth4[0] = 1
    pred3 = np.array([2, 1, 3, 4, 5, 6, 7])*10
    k = len(pred3)

    #print ndcg_at_k_ties(truth, truth, k, method=0, normalize_from_below_too=True)
    #print ndcg_at_k_ties(truth, pred2, k, method=0, normalize_from_below_too=True)
    #print ndcg_at_k_ties(truth, pred3, k, method=0, normalize_from_below_too=True)
    #print ndcg_at_k_ties(truth3, pred3, k, method=3, normalize_from_below_too=True)
    print ndcg_at_k_ties(truth4, pred2, k, method=3, normalize_from_below_too=True)

    import ipdb; ipdb.set_trace()

    print ndcg_alt(truth[np.argsort(pred2)[::-1]], 5)
    print ndcg_at_k(truth[np.argsort(pred2)[::-1]], 5, method=1)
    print ndcg_at_k(truth[np.argsort(pred2)[::-1]], 5, method=0)
           
    print ndcg_at_k_ties(truth, pred2, 5, method=1)
    print ndcg_at_k_ties(truth, pred2, 5, method=0)

