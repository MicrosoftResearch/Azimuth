import pandas
import time
import sklearn
import numpy as np
import Bio.SeqUtils as SeqUtil
import Bio.Seq as Seq
import azimuth.util
import sys
import Bio.SeqUtils.MeltingTemp as Tm
import pickle
import itertools

def featurize_data(data, learn_options, Y, gene_position, pam_audit=True, length_audit=True):
    '''
    assumes that data contains the 30mer
    returns set of features from which one can make a kernel for each one
    '''    
    all_lens = data['30mer'].apply(len).values
    unique_lengths = np.unique(all_lens)
    num_lengths = len(unique_lengths)
    assert num_lengths == 1, "should only have sequences of a single length, but found %s: %s" % (num_lengths, str(unique_lengths))

    print "Constructing features..."
    t0 = time.time()

    feature_sets = {}

    if learn_options["nuc_features"]:
        # spectrum kernels (position-independent) and weighted degree kernels (position-dependent)
        get_all_order_nuc_features(data['30mer'], feature_sets, learn_options, learn_options["order"], max_index_to_use=30)

    check_feature_set(feature_sets)

    if learn_options["gc_features"]:
        gc_above_10, gc_below_10, gc_count = gc_features(data, length_audit)
        feature_sets['gc_above_10'] = pandas.DataFrame(gc_above_10)
        feature_sets['gc_below_10'] = pandas.DataFrame(gc_below_10)
        feature_sets['gc_count'] = pandas.DataFrame(gc_count)

    if learn_options["include_gene_position"]:
        # gene_position_columns = ["Amino Acid Cut position", "Percent Peptide", "Nucleotide cut position"]
        # gene_position_columns = ["Percent Peptide", "Nucleotide cut position"]

        for set in gene_position.columns:
            set_name = set
            feature_sets[set_name] = pandas.DataFrame(gene_position[set])
        feature_sets["Percent Peptide <50%"] = feature_sets["Percent Peptide"] < 50
        feature_sets["Percent Peptide <50%"]['Percent Peptide <50%'] = feature_sets["Percent Peptide <50%"].pop("Percent Peptide")

    if learn_options["include_gene_effect"]:
        print "including gene effect"
        gene_names = Y['Target gene']
        enc = sklearn.preprocessing.OneHotEncoder()
        label_encoder = sklearn.preprocessing.LabelEncoder()
        label_encoder.fit(gene_names)
        one_hot_genes = np.array(enc.fit_transform(label_encoder.transform(gene_names)[:, None]).todense())
        feature_sets["gene effect"] = pandas.DataFrame(one_hot_genes,
                                                       columns=["gene_%d" % i for i in range(one_hot_genes.shape[1])], index=gene_names.index)

    if learn_options['include_known_pairs']:
        feature_sets['known pairs'] = pandas.DataFrame(Y['test'])

    if learn_options["include_NGGX_interaction"]:
        feature_sets["NGGX"] = NGGX_interaction_feature(data, pam_audit)

    if learn_options["include_Tm"]:
        feature_sets["Tm"] = Tm_feature(data, pam_audit)

    if learn_options["include_sgRNAscore"]:
        feature_sets["sgRNA Score"] = pandas.DataFrame(data["sgRNA Score"])

    if learn_options["include_drug"]:
        # feature_sets["drug"] = pandas.DataFrame(data["drug"])
        drug_names = Y.index.get_level_values('drug').tolist()
        enc = sklearn.preprocessing.OneHotEncoder()
        label_encoder = sklearn.preprocessing.LabelEncoder()
        label_encoder.fit(drug_names)
        one_hot_drugs = np.array(enc.fit_transform(label_encoder.transform(drug_names)[:, None]).todense())
        feature_sets["drug"] = pandas.DataFrame(one_hot_drugs, columns=["drug_%d" % i for i in range(one_hot_drugs.shape[1])], index=drug_names)

    if learn_options['include_strand']:
        feature_sets['Strand effect'] = (pandas.DataFrame(data['Strand']) == 'sense')*1

    if learn_options["include_gene_feature"]:
        feature_sets["gene features"] = gene_feature(Y, data, learn_options)

    if learn_options["include_gene_guide_feature"] > 0:
        tmp_feature_sets = gene_guide_feature(Y, data, learn_options)
        for key in tmp_feature_sets:
            feature_sets[key] = tmp_feature_sets[key]

    if learn_options["include_microhomology"]:
        feature_sets["microhomology"] = get_micro_homology_features(Y['Target gene'], learn_options, data)

    t1 = time.time()
    print "\t\tElapsed time for constructing features is %.2f seconds" % (t1-t0)

    check_feature_set(feature_sets)

    if learn_options['normalize_features']:
        assert("should not be here as doesn't make sense when we make one-off predictions, but could make sense for internal model comparisons when using regularized models")
        feature_sets = normalize_feature_sets(feature_sets)
        check_feature_set(feature_sets)

    return feature_sets


def check_feature_set(feature_sets):
    '''
    Ensure the # of people is the same in each feature set
    '''
    assert feature_sets != {}, "no feature sets present"

    N = None
    for ft in feature_sets.keys():
        N2 = feature_sets[ft].shape[0]
        if N is None:
            N = N2
        else:
            assert N >= 1, "should be at least one individual"
            assert N == N2, "# of individuals do not match up across feature sets"

    for set in feature_sets.keys():
        if np.any(np.isnan(feature_sets[set])):
            raise Exception("found Nan in set %s" % set)


def NGGX_interaction_feature(data, pam_audit=True):
    '''
    assuming 30-mer, grab the NGGX _ _ positions, and make a one-hot
    encoding of the NX nucleotides yielding 4x4=16 features
    '''
    sequence = data['30mer'].values
    feat_NX = pandas.DataFrame()
    # check that GG is where we think
    for seq in sequence:
        if pam_audit and seq[25:27] != "GG":
            raise Exception("expected GG but found %s" % seq[25:27])
        NX = seq[24]+seq[27]
        NX_onehot = nucleotide_features(NX,order=2, feature_type='pos_dependent', max_index_to_use=2, prefix="NGGX")
        # NX_onehot[:] = np.random.rand(NX_onehot.shape[0]) ##TESTING RANDOM FEATURE
        feat_NX = pandas.concat([feat_NX, NX_onehot], axis=1)
    return feat_NX.T


def get_all_order_nuc_features(data, feature_sets, learn_options, maxorder, max_index_to_use, prefix=""):
    for order in range(1, maxorder+1):
        print "\t\tconstructing order %s features" % order
        nuc_features_pd, nuc_features_pi = apply_nucleotide_features(data, order, learn_options["num_proc"],
                                                                     include_pos_independent=True, max_index_to_use=max_index_to_use, prefix=prefix)
        feature_sets['%s_nuc_pd_Order%i' % (prefix, order)] = nuc_features_pd
        if learn_options['include_pi_nuc_feat']:
            feature_sets['%s_nuc_pi_Order%i' % (prefix, order)] = nuc_features_pi
        check_feature_set(feature_sets)
        print "\t\t\t\t\t\t\tdone"


def countGC(s, length_audit=True):
    '''
    GC content for only the 20mer, as per the Doench paper/code
    '''
    if length_audit:
        assert len(s) == 30, "seems to assume 30mer"
    return len(s[4:24].replace('A', '').replace('T', ''))


def SeqUtilFeatures(data):
    '''
    assuming '30-mer'is a key
    get melting temperature features from:
        0-the 30-mer ("global Tm")
        1-the Tm (melting temperature) of the DNA:RNA hybrid from positions 16 - 20 of the sgRNA, i.e. the 5nts immediately proximal of the NGG PAM
        2-the Tm of the DNA:RNA hybrid from position 8 - 15 (i.e. 8 nt)
        3-the Tm of the DNA:RNA hybrid from position 3 - 7  (i.e. 5 nt)
    '''
    sequence = data['30mer'].values
    num_features = 1
    featarray = np.ones((sequence.shape[0], num_features))
    for i, seq in enumerate(sequence):
        assert len(seq) == 30, "seems to assume 30mer"
        featarray[i, 0] = SeqUtil.molecular_weight(str(seq))

    feat = pandas.DataFrame(pandas.DataFrame(featarray))
    return feat


def organism_feature(data):
    '''
    Human vs. mouse
    '''
    organism = np.array(data['Organism'].values)
    feat = pandas.DataFrame(pandas.DataFrame(featarray))
    import ipdb; ipdb.set_trace()
    return feat


def get_micro_homology_features(gene_names, learn_options, X):
    # originally was flipping the guide itself as necessary, but now flipping the gene instead

    print "building microhomology features"
    feat = pandas.DataFrame(index=X.index)
    feat["mh_score"] = ""
    feat["oof_score"] = ""

    #with open(r"tmp\V%s_gene_mismatches.csv" % learn_options["V"],'wb') as f:
    if True:
        # number of nulceotides to take to the left and right of the guide
        k_mer_length_left = 9
        k_mer_length_right = 21
        for gene in gene_names.unique():
            gene_seq = Seq.Seq(util.get_gene_sequence(gene)).reverse_complement()
            guide_inds = np.where(gene_names.values == gene)[0]
            print "getting microhomology for all %d guides in gene %s" % (len(guide_inds), gene)
            for j, ps in enumerate(guide_inds):
                guide_seq = Seq.Seq(X['30mer'][ps])
                strand = X['Strand'][ps]
                if strand=='sense':
                    gene_seq = gene_seq.reverse_complement()
                # figure out the sequence to the left and right of this guide, in the gene
                ind = gene_seq.find(guide_seq)
                if ind==-1:
                    gene_seq = gene_seq.reverse_complement()
                    ind = gene_seq.find(guide_seq)
                    #assert ind != -1, "still didn't work"
                    #print "shouldn't get here"
                else:
                    #print "all good"
                    pass
                #assert ind != -1, "could not find guide in gene"
                if ind==-1:
                    #print "***could not find guide %s for gene %s" % (str(guide_seq), str(gene))
                    #if.write(str(gene) + "," + str(guide_seq))
                    mh_score = 0
                    oof_score = 0
                else:
                    #print "worked"

                    assert gene_seq[ind:(ind+len(guide_seq))]==guide_seq, "match not right"
                    left_win = gene_seq[(ind - k_mer_length_left):ind]
                    right_win = gene_seq[(ind + len(guide_seq)):(ind + len(guide_seq) + k_mer_length_right)]

                    #if strand=='antisense':
                    #    # it's arbitrary which of sense and anti-sense we flip, we just want
                    #    # to keep them in the same relative alphabet/direction
                    #    left_win = left_win.reverse_complement()
                    #    right_win = right_win.reverse_complement()
                    assert len(left_win.tostring())==k_mer_length_left
                    assert len(right_win.tostring())==k_mer_length_right

                    sixtymer = str(left_win) + str(guide_seq) + str(right_win)
                    assert len(sixtymer)==60, "should be of length 60"
                    mh_score, oof_score = microhomology.compute_score(sixtymer)

                feat.ix[ps,"mh_score"] = mh_score
                feat.ix[ps,"oof_score"] = oof_score
            print "computed microhomology of %s" % (str(gene))

    return pandas.DataFrame(feat, dtype='float')


def local_gene_seq_features(gene_names, learn_options, X):

    print "building local gene sequence features"
    feat = pandas.DataFrame(index=X.index)
    feat["gene_left_win"] = ""
    feat["gene_right_win"] = ""

    # number of nulceotides to take to the left and right of the guide
    k_mer_length = learn_options['include_gene_guide_feature']
    for gene in gene_names.unique():
        gene_seq = Seq.Seq(util.get_gene_sequence(gene)).reverse_complement()
        for ps in np.where(gene_names.values==gene)[0]:
            guide_seq = Seq.Seq(X['30mer'][ps])
            strand = X['Strand'][ps]
            if strand=='sense':
                guide_seq = guide_seq.reverse_complement()
                #gene_seq = gene_seq.reverse_complement()
            # figure out the sequence to the left and right of this guide, in the gene
            ind = gene_seq.find(guide_seq)
            if ind ==-1:
                #gene_seq = gene_seq.reverse_complement()
                #ind = gene_seq.find(guide_seq)
                assert ind != -1, "could not find guide in gene"
            assert gene_seq[ind:(ind+len(guide_seq))]==guide_seq, "match not right"
            left_win = gene_seq[(ind - k_mer_length):ind]
            right_win = gene_seq[(ind + len(guide_seq)):(ind + len(guide_seq) + k_mer_length)]

            if strand=='antisense':
                # it's arbitrary which of sense and anti-sense we flip, we just want
                # to keep them in the same relative alphabet/direction
                left_win = left_win.reverse_complement()
                right_win = right_win.reverse_complement()
            assert not left_win.tostring()=="", "k_mer_context, %s, is too large" % k_mer_length
            assert not left_win.tostring()=="", "k_mer_context, %s, is too large" % k_mer_length
            assert len(left_win)==len(right_win), "k_mer_context, %s, is too large" % k_mer_length
            feat.ix[ps,"gene_left_win"] = left_win.tostring()
            feat.ix[ps,"gene_right_win"] = right_win.tostring()
        print "featurizing local context of %s" % (gene)

    feature_sets = {}
    get_all_order_nuc_features(feat["gene_left_win"], feature_sets, learn_options, learn_options["order"], max_index_to_use=sys.maxint, prefix="gene_left_win")
    get_all_order_nuc_features(feat["gene_right_win"], feature_sets, learn_options, learn_options["order"], max_index_to_use=sys.maxint, prefix="gene_right_win")
    return feature_sets

def gene_feature(Y, X, learn_options):
    '''
    Things like the sequence of the gene, the DNA Tm of the gene, etc.
    '''

    gene_names = Y['Target gene']

    gene_length = np.zeros((gene_names.values.shape[0], 1))
    gc_content = np.zeros((gene_names.shape[0], 1))
    temperature = np.zeros((gene_names.shape[0], 1))
    molecular_weight = np.zeros((gene_names.shape[0], 1))

    for gene in gene_names.unique():
        seq = util.get_gene_sequence(gene)
        gene_length[gene_names.values==gene] = len(seq)
        gc_content[gene_names.values==gene] = SeqUtil.GC(seq)
        temperature[gene_names.values==gene] = Tm.Tm_staluc(seq, rna=False)
        molecular_weight[gene_names.values==gene] = SeqUtil.molecular_weight(seq, 'DNA')

    all = np.concatenate((gene_length, gc_content, temperature, molecular_weight), axis=1)
    df = pandas.DataFrame(data=all, index=gene_names.index, columns=['gene length',
                                                                     'gene GC content',
                                                                     'gene temperature',
                                                                     'gene molecular weight'])
    return df

def gene_guide_feature(Y, X, learn_options):
    #features, which are related to parts of the gene-local to the guide, and
    #possibly incorporating the guide or interactions with it

    #expensive, so pickle if necessary
    gene_file = r"..\data\gene_seq_feat_V%s_km%s.ord%s.pickle" % (learn_options['V'], learn_options['include_gene_guide_feature'], learn_options['order'])

    if False: #os.path.isfile(gene_file): #while debugging, comment out
        print "loading local gene seq feats from file %s" % gene_file
        with open(gene_file, "rb") as f: feature_sets = pickle.load(f)
    else:
        feature_sets = local_gene_seq_features(Y['Target gene'], learn_options, X)
        print "writing local gene seq feats to file %s" % gene_file
        with open(gene_file, "wb") as f: pickle.dump(feature_sets, f)

    return feature_sets


def gc_cont(seq):
    return (seq.count('G') + seq.count('C'))/float(len(seq))



def Tm_feature(data, pam_audit=True):
    '''
    assuming '30-mer'is a key
    get melting temperature features from:
        0-the 30-mer ("global Tm")
        1-the Tm (melting temperature) of the DNA:RNA hybrid from positions 16 - 20 of the sgRNA, i.e. the 5nts immediately proximal of the NGG PAM
        2-the Tm of the DNA:RNA hybrid from position 8 - 15 (i.e. 8 nt)
        3-the Tm of the DNA:RNA hybrid from position 3 - 7  (i.e. 5 nt)
    '''
    sequence = data['30mer'].values
    featarray = np.ones((sequence.shape[0],4))
    for i, seq in enumerate(sequence):
        if pam_audit and seq[25:27]!="GG":
            raise Exception("expected GG but found %s" % seq[25:27])
        rna = False
        featarray[i,0] = Tm.Tm_staluc(seq, rna=rna)        #30mer Tm
        featarray[i,1] = Tm.Tm_staluc(seq[19:24], rna=rna) #5nts immediately proximal of the NGG PAM
        featarray[i,2] = Tm.Tm_staluc(seq[11:19], rna=rna)   #8-mer
        featarray[i,3] = Tm.Tm_staluc(seq[6:11], rna=rna)      #5-mer


    feat = pandas.DataFrame(featarray, index=data.index, columns=["Tm global_%s" % rna, "5mer_end_%s" %rna, "8mer_middle_%s" %rna, "5mer_start_%s" %rna])

    return feat

def gc_features(data, audit=True):
    gc_count = data['30mer'].apply(lambda seq: countGC(seq, audit))
    gc_count.name = 'GC count'
    gc_above_10 = (gc_count > 10)*1
    gc_above_10.name = 'GC > 10'
    gc_below_10 = (gc_count < 10)*1
    gc_below_10.name = 'GC < 10'
    return gc_above_10, gc_below_10, gc_count



def normalize_features(data,axis):
    '''
    input: Pandas.DataFrame of dtype=np.float64 array, of dimensions
    mean-center, and unit variance each feature
    '''
    data -= data.mean(axis)
    data /= data.std(axis)
    # remove rows with NaNs
    data = data.dropna(1)
    if np.any(np.isnan(data.values)): raise Exception("found NaN in normalized features")
    return data

def apply_nucleotide_features(seq_data_frame, order, num_proc, include_pos_independent, max_index_to_use, prefix=""):

    fast = True
    if include_pos_independent:
        feat_pd = seq_data_frame.apply(nucleotide_features, args=(order, max_index_to_use, prefix, 'pos_dependent'))        
        feat_pi = seq_data_frame.apply(nucleotide_features, args=(order, max_index_to_use, prefix, 'pos_independent'))            
        assert not np.any(np.isnan(feat_pd)), "nans here can arise from sequences of different lengths"           
        assert not np.any(np.isnan(feat_pi)), "nans here can arise from sequences of different lengths"           
        return feat_pd, feat_pi
    else:
        feat_pd = seq_data_frame.apply(nucleotide_features, args=(order, max_index_to_use, prefix, 'pos_dependent'))
        assert not np.any(np.isnan(feat_pd)), "found nan in feat_pd"
        return feat_pd

def get_alphabet(order, raw_alphabet = ['A', 'T', 'C', 'G']):
    alphabet = ["".join(i) for i in itertools.product(raw_alphabet, repeat=order)]
    return alphabet

def nucleotide_features(s, order, max_index_to_use, prefix="", feature_type='all', raw_alphabet = ['A', 'T', 'C', 'G']):
    '''
    compute position-specific order-mer features for the 4-letter alphabet
    (e.g. for a sequence of length 30, there are 30*4 single nucleotide features
          and (30-1)*4^2=464 double nucleotide features
    '''
    assert feature_type in ['all', 'pos_independent', 'pos_dependent']
    if max_index_to_use <= len(s):
        #print "WARNING: trimming max_index_to use down to length of string=%s" % len(s)        
        max_index_to_use = len(s)
    
    if max_index_to_use is not None:
        s = s[:max_index_to_use]
    #assert(len(s)==30, "length not 30")
    #s = s[:30] #cut-off at thirty to clean up extra data that they accidentally left in, and were instructed to ignore in this way
    alphabet = get_alphabet(order, raw_alphabet = raw_alphabet)
    features_pos_dependent = np.zeros(len(alphabet)*(len(s)-(order-1)))
    features_pos_independent = np.zeros(np.power(len(raw_alphabet),order))

    for position in range(0, len(s)-order+1, 1):
        nucl = s[position:position+order]        
        features_pos_dependent[alphabet.index(nucl) + (position*len(alphabet))] = 1.0
        features_pos_independent[alphabet.index(nucl)] += 1.0
    index_dependent = ['%s_pd.Order%d_P%d' % (prefix, order, i) for i in range(len(features_pos_dependent))]

    if np.any(np.isnan(features_pos_dependent)):
        raise Exception("found nan features in features_pos_dependent")
    if np.any(np.isnan(features_pos_independent)):
        raise Exception("found nan features in features_pos_independent")

    if feature_type == 'all' or feature_type == 'pos_independent':
        index_independent = ['%s_pi.Order%d_P%d' % (prefix, order,i) for i in range(len(features_pos_independent))]
        if feature_type == 'all':
            res = pandas.Series(features_pos_dependent,index=index_dependent), pandas.Series(features_pos_independent,index=index_independent)
            assert not np.any(np.isnan(res.values))
            return res
        else:
            res = pandas.Series(features_pos_independent, index=index_independent)
            assert not np.any(np.isnan(res.values))
            return res

    res = pandas.Series(features_pos_dependent, index=index_dependent)
    assert not np.any(np.isnan(res.values))
    return res

def nucleotide_features_dictionary(prefix=''):
    seqname = ['-4', '-3', '-2', '-1']
    seqname.extend([str(i) for i in range(1,21)])
    seqname.extend(['N', 'G', 'G', '+1', '+2', '+3'])

    orders = [1, 2, 3]
    sequence = 30
    feature_names_dep = []
    feature_names_indep = []
    index_dependent = []
    index_independent = []

    for order in orders:
        raw_alphabet = ['A', 'T', 'C', 'G']
        alphabet = ["".join(i) for i in itertools.product(raw_alphabet, repeat=order)]
        features_pos_dependent = np.zeros(len(alphabet)*(sequence-(order-1)))
        features_pos_independent = np.zeros(np.power(len(raw_alphabet),order))

        index_dependent.extend(['%s_pd.Order%d_P%d' % (prefix, order, i) for i in range(len(features_pos_dependent))])
        index_independent.extend(['%s_pi.Order%d_P%d' % (prefix, order,i) for i in range(len(features_pos_independent))])

        for pos in range(sequence-(order-1)):
            for letter in alphabet:
                feature_names_dep.append('%s_%s' % (letter, seqname[pos]))

        for letter in alphabet:
            feature_names_indep.append('%s' % letter)

        assert len(feature_names_indep) == len(index_independent)
        assert len(feature_names_dep) == len(index_dependent)

    index_all = index_dependent + index_independent
    feature_all = feature_names_dep + feature_names_indep

    return dict(zip(index_all, feature_all))

def normalize_feature_sets(feature_sets):
    '''
    zero-mean, unit-variance each feature within each set
    '''

    print "Normalizing features..."
    t1 = time.time()

    new_feature_sets = {}
    for set in feature_sets:
         new_feature_sets[set] = normalize_features(feature_sets[set],axis=0)
         if np.any(np.isnan(new_feature_sets[set].values)):
             raise Exception("found Nan feature values in set=%s" % set)
         assert new_feature_sets[set].shape[1] > 0, "0 columns of features"
    t2 = time.time()
    print "\t\tElapsed time for normalizing features is %.2f seconds" % (t2-t1)

    return new_feature_sets
