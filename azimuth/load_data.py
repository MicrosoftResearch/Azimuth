import pandas
import util
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
import numpy as np
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))

def from_custom_file(data_file, learn_options):
    # use semantics of when we load V2 data
    print "Loading inputs to predict from %s" % data_file
    data = pandas.read_csv(data_file)

    mandatory_columns = ['30mer', 'Target gene', 'Percent Peptide', 'Amino Acid Cut position']
    for col in mandatory_columns:
        assert col in data.columns, "inputs for prediction must include these columns: %s" % mandatory_columns

    Xdf = pandas.DataFrame(data)
    Xdf['30mercopy'] = Xdf['30mer']
    Xdf = Xdf.set_index(['30mer', 'Target gene'])
    Xdf['30mer'] = Xdf['30mercopy']
    Xdf.index.names = ['Sequence', 'Target']
    Xdf['drug']= ['dummydrug%s' % i for i in range(Xdf.shape[0])]
    Xdf = Xdf.set_index('drug', append=True)

    Y = None
    gene_position = Xdf[['Percent Peptide', 'Amino Acid Cut position']]
    target_genes =  np.unique(Xdf.index.levels[1])

    learn_options = set_V2_target_names(learn_options)

    return Xdf, Y, gene_position, target_genes


def from_file(data_file, learn_options, data_file2=None, data_file3=None):
    if learn_options["V"] == 1:  # from Nature Biotech paper

        print "loading V%d data" % learn_options["V"]

        assert not learn_options["weighted"] is not None, "not supported for V1 data"
        annotations, gene_position, target_genes, Xdf, Y = read_V1_data(data_file, learn_options)

        learn_options['binary target name'] = 'average threshold'
        learn_options['rank-transformed target name'] = 'average rank'
        learn_options['raw target name'] = 'average activity'

        # NF: not sure why the line below was uncommented
        # gene_position, selected_ind, target_genes, Xdf, Y = extract_by_organism("mouse", Xdf, Y, gene_position)

    elif learn_options["V"] == 2:  # from Nov 2014, hot off the machines
        Xdf, drugs_to_genes, target_genes, Y, gene_position = read_V2_data(data_file, learn_options)

        # check that data is consistent with sgRNA score
        xx = Xdf['sgRNA Score'].values
        yy = Y['score_drug_gene_rank'].values
        rr,pp = sp.stats.pearsonr(xx, yy)
        assert rr > 0, "data processing has gone wrong as correlation with previous predictions is negative"

        learn_options = set_V2_target_names(learn_options)

    elif learn_options["V"] == 3:  # merge of V1 and V2--this is what is used for the final model
        # these are relative to the V2 data, and V1 will be made to automatically match
        learn_options['binary target name'] = 'score_drug_gene_threshold'
        learn_options['rank-transformed target name'] = 'score_drug_gene_rank'
        learn_options['raw target name'] = None

        Xdf, Y, gene_position, target_genes = mergeV1_V2(data_file, data_file2, learn_options)


    elif learn_options["V"] == 4:  # merge of V1 and V2 and the Xu et al data
        # these are relative to the V2 data, and V1 and Xu et al. will be made to automatically match
        learn_options['binary target name'] = 'score_drug_gene_threshold'
        learn_options['rank-transformed target name'] = 'score_drug_gene_rank'
        learn_options['raw target name'] = None

        Xdf, Y, gene_position, target_genes = merge_all(data_file, data_file2, data_file3, learn_options)


    elif learn_options['V'] == 5:
        learn_options['binary target name'] = 'score_drug_gene_threshold'
        learn_options['rank-transformed target name'] = 'score_drug_gene_rank'
        learn_options['raw target name'] = None

        gene_position, target_genes, Xdf, Y = read_xu_et_al(data_file3)


    # truncate down to 30--some data sets gave us more.
    Xdf["30mer"] = Xdf["30mer"].apply(lambda x: x[0:30])

    return Xdf, Y, gene_position, target_genes


def set_V2_target_names(learn_options):
    if 'binary target name' not in learn_options.keys():
        learn_options['binary target name'] = 'score_drug_gene_threshold'
    if 'rank-transformed target name' not in learn_options.keys():
        learn_options['rank-transformed target name'] = 'score_drug_gene_rank'
    learn_options['raw target name'] = 'score'
    return learn_options


def combine_organisms(human_data, mouse_data):
    # 'Target' is the column name, 'CD13' are some rows in that column
    # xs slices through the pandas data frame to return another one
    cd13 = human_data.xs('CD13', level='Target', drop_level=False)
    # y_names are column names, cd13 is a pandas object
    X_CD13, Y_CD13 = util.get_data(cd13, y_names=['NB4 CD13', 'TF1 CD13'])
    cd33 = human_data.xs('CD33', level='Target', drop_level=False)
    X_CD33, Y_CD33 = util.get_data(cd33, y_names=['MOLM13 CD33', 'TF1 CD33', 'NB4 CD33'])
    cd15 = human_data.xs('CD15', level='Target', drop_level=False)
    X_CD15, Y_CD15 = util.get_data(cd15, y_names=['MOLM13 CD15'])

    mouse_X = pandas.DataFrame()
    mouse_Y = pandas.DataFrame()
    for k in mouse_data.index.levels[1]:
        # is k the gene
        X, Y = util.get_data(mouse_data.xs(k, level='Target', drop_level=False), ["On-target Gene"], target_gene=k, organism='mouse')
        mouse_X = pandas.concat([mouse_X, X], axis=0)
        mouse_Y = pandas.concat([mouse_Y, Y], axis=0)

    X = pandas.concat([X_CD13, X_CD15, X_CD33, mouse_X], axis=0)
    Y = pandas.concat([Y_CD13, Y_CD15, Y_CD33, mouse_Y], axis=0)

    return X, Y


def read_V1_data(data_file, learn_options, AML_file=cur_dir + "/data/V1_suppl_data.txt"):
    if data_file is None:
        data_file = cur_dir + "/data/V1_data.xlsx"
    human_data = pandas.read_excel(data_file, sheetname=0, index_col=[0, 1])
    mouse_data = pandas.read_excel(data_file, sheetname=1, index_col=[0, 1])
    Xdf, Y = combine_organisms(human_data, mouse_data)

    # get position within each gene, then join and re-order
    # note that 11 missing guides we were told to ignore
    annotations = pandas.read_csv(AML_file, delimiter='\t', index_col=[0, 4])
    annotations.index.names = Xdf.index.names
    gene_position = pandas.merge(Xdf, annotations, how="inner", left_index=True, right_index=True)
    gene_position = util.impute_gene_position(gene_position)
    gene_position = gene_position[['Amino Acid Cut position', 'Nucleotide cut position', 'Percent Peptide']]
    Y = Y.loc[gene_position.index]
    Xdf = Xdf.loc[gene_position.index]

    Y['test'] = 1  # for bookeeping to keep consistent with V2 which uses this for "extra pairs"

    target_genes = Y['Target gene'].unique()

    Y.index.names = ['Sequence', 'Target gene']

    assert Xdf.index.equals(Y.index), "The index of Xdf is different from the index of Y (this can cause inconsistencies/random performance later on)"

    if learn_options is not None and learn_options["flipV1target"]:
        print "************************************************************************"
        print "*****************MATCHING DOENCH CODE (DEBUG MODE)**********************"
        print "************************************************************************"
        # normally it is: Y['average threshold'] = Y['average rank'] > 0.8, where 1s are good guides, 0s are not
        Y['average threshold'] = Y['average rank'] < 0.2  # 1s are bad guides
        print "press c to continue"
        import ipdb
        ipdb.set_trace()

    return annotations, gene_position, target_genes, Xdf, Y

def rank_transform(x):
    return 1.0 - sp.stats.mstats.rankdata(x)/sp.stats.mstats.rankdata(x).max()

def read_xu_et_al(data_file, learn_options=None, verbose=True, subsetting='ours'):
    if data_file is None:
        data_file = '../data/xu_et_al_data.xlsx'

    datasets = ['ribo', 'non_ribo', 'mESC']
    aggregated = None

    for d in datasets:
        data_efficient = pandas.read_excel(data_file, sheetname='%s_efficient_sgRNA' % d, skiprows=2)
        data_inefficient = pandas.read_excel(data_file, sheetname='%s_inefficient_sgRNA' % d, skiprows=2)

        data_efficient['threshold'] = 1.
        data_inefficient['threshold'] = 0.

        exp_data = pandas.concat((data_efficient, data_inefficient))
        exp_data['rank_KBM7'] = exp_data.groupby('Gene Symbol')['log2 fold change, KBM7'].transform(rank_transform)
        exp_data['rank_HL60'] = exp_data.groupby('Gene Symbol')['log2 fold change, HL60'].transform(rank_transform)

        if aggregated is None:
            aggregated = exp_data
        else:
            aggregated = pandas.concat((aggregated, exp_data))


    # go from 40mer to 30mer
    if subsetting == 'ours':
        aggregated["sequence(target+3'+5')"] = aggregated["sequence(target+3'+5')"].apply(lambda x: x[6:-4])
    else:
        aggregated["sequence(target+3'+5')"] = aggregated["sequence(target+3'+5')"].apply(lambda x: x[10:])

    # make sure EVEYTHING is uppercase
    aggregated["sequence(target+3'+5')"] = aggregated["sequence(target+3'+5')"].apply(lambda x: x.upper())

    # rename columns
    aggregated.rename(columns={"sequence(target+3'+5')": '30mer', 'Gene Symbol': 'Target gene', 'strand':'Strand'}, inplace=True)

    aggregated['Strand'].loc[aggregated['Strand']=='+'] = 'sense'
    aggregated['Strand'].loc[aggregated['Strand']=='-'] = 'antisense'

    aggregated['average rank'] = aggregated[['rank_HL60', 'rank_KBM7']].mean(axis=1)
    df = aggregated
    df = df.rename(columns={'30mer': 'Sequence', 'Target gene': 'Target'})
    df['drug'] = 'nodrug'
    df['test'] = 1
    df = df.set_index(['Sequence', 'Target', 'drug'])
    df['30mer'] = df.index.get_level_values(0)
    df['Target gene'] = df.index.get_level_values(1)
    df['Organism'] = 'unknown'
    df['score_drug_gene_rank'] = df['average rank']
    df['score_drug_gene_threshold'] = df['threshold']
    df['Nucleotide cut position'] = df['start of target']
    df['Percent Peptide'] = 0
    df['Amino Acid Cut position'] = 0
    target_genes = np.unique(df['Target gene'].values)

    return df[['Nucleotide cut position', 'Percent Peptide', 'Amino Acid Cut position']], target_genes, df[['30mer', 'Strand']], df[['score_drug_gene_rank', 'score_drug_gene_threshold', 'test', 'Target gene']]

def read_V2_data(data_file, learn_options=None, verbose=True):
    if data_file is None:
        data_file = cur_dir + "/data/V2_data.xlsx"

    # to compare
    # import predict as pr; a1, g1, t1, X1, Y1 = pr.data_setup()
    # a1.index.names

    data = pandas.read_excel(data_file, sheetname="ResultsFiltered", skiprows=range(0, 6+1), index_col=[0, 4])
    # grab data relevant to each of three drugs, which exludes some genes
    # note gene MED12 has two drugs, all others have at most one
    Xdf = pandas.DataFrame()

    # This comes from the "Pairs" tab in their excel sheet,
    # note HPRT/HPRT1 are same thing, and also PLX_2uM/PLcX_2uM
    known_pairs = {'AZD_200nM':  ['CCDC101', 'MED12', 'TADA2B', 'TADA1'],
                   '6TG_2ug/mL': ['HPRT1'],
                   'PLX_2uM':    ['CUL3', 'NF1', 'NF2', 'MED12']}

    drugs_to_genes = {'AZD_200nM':  ['CCDC101', 'MED12', 'TADA2B', 'TADA1'],
                      '6TG_2ug/mL': ['HPRT1'],
                      'PLX_2uM':    ['CUL3', 'NF1', 'NF2', 'MED12']}

    if learn_options is not None:
        assert not (learn_options['extra pairs'] and learn_options['all pairs']), "extra pairs and all pairs options (in learn_options) can't be active simultaneously."

        if learn_options['extra pairs']:
            drugs_to_genes['AZD_200nM'].extend(['CUL3', 'NF1', 'NF2'])
        elif learn_options['all pairs']:
            drugs_to_genes['AZD_200nM'].extend(['HPRT1', 'CUL3', 'NF1', 'NF2'])
            drugs_to_genes['PLX_2uM'].extend(['HPRT1', 'CCDC101', 'TADA2B', 'TADA1'])
            drugs_to_genes['6TG_2ug/mL'].extend(['CCDC101', 'MED12', 'TADA2B', 'TADA1', 'CUL3', 'NF1', 'NF2'])

    count = 0
    for drug in drugs_to_genes.keys():
        genes = drugs_to_genes[drug]
        for g in genes:
            Xtmp = data.copy().xs(g, level='Target gene', drop_level=False)
            Xtmp['drug'] = drug
            Xtmp['score'] = Xtmp[drug].copy()  # grab the drug results that are relevant for this gene

            if g in known_pairs[drug]:
                Xtmp['test'] = 1.
            else:
                Xtmp['test'] = 0.

            count = count + Xtmp.shape[0]
            Xdf = pandas.concat([Xdf, Xtmp], axis=0)
            if verbose:
                print "Loaded %d samples for gene %s \ttotal number of samples: %d" % (Xtmp.shape[0], g, count)

    # create new index that includes the drug
    Xdf = Xdf.set_index('drug', append=True)

    Y = pandas.DataFrame(Xdf.pop("score"))
    Y.columns.names = ["score"]

    test_gene = pandas.DataFrame(Xdf.pop('test'))
    target = pandas.DataFrame(Xdf.index.get_level_values('Target gene').values, index=Y.index, columns=["Target gene"])
    Y = pandas.concat((Y, target, test_gene), axis=1)
    target_genes = Y['Target gene'].unique()
    gene_position = Xdf[["Percent Peptide", "Amino Acid Cut position"]].copy()

    # convert to ranks for each (gene, drug combo)
    # flip = True
    y_rank = pandas.DataFrame()
    y_threshold = pandas.DataFrame()
    y_quant = pandas.DataFrame()
    for drug in drugs_to_genes.keys():
        gene_list = drugs_to_genes[drug]
        for gene in gene_list:
            ytmp = pandas.DataFrame(Y.xs((gene, drug), level=["Target gene", "drug"], drop_level=False)['score'])
            y_ranktmp, y_rank_raw, y_thresholdtmp, y_quanttmp = util.get_ranks(ytmp, thresh=0.8, prefix="score_drug_gene", flip=False)
            # np.unique(y_rank.values-y_rank_raw.values)
            y_rank = pandas.concat((y_rank, y_ranktmp), axis=0)
            y_threshold = pandas.concat((y_threshold, y_thresholdtmp), axis=0)
            y_quant = pandas.concat((y_quant, y_quanttmp), axis=0)

    yall = pandas.concat((y_rank, y_threshold, y_quant), axis=1)
    Y = pandas.merge(Y, yall, how='inner', left_index=True, right_index=True)

    # convert also by drug only, irrespective of gene
    y_rank = pandas.DataFrame()
    y_threshold = pandas.DataFrame()
    y_quant = pandas.DataFrame()
    for drug in drugs_to_genes.keys():
        ytmp = pandas.DataFrame(Y.xs(drug, level="drug", drop_level=False)['score'])
        y_ranktmp, y_rank_raw, y_thresholdtmp, y_quanttmp = util.get_ranks(ytmp, thresh=0.8, prefix="score_drug", flip=False)
        # np.unique(y_rank.values-y_rank_raw.values)
        y_rank = pandas.concat((y_rank, y_ranktmp), axis=0)
        y_threshold = pandas.concat((y_threshold, y_thresholdtmp), axis=0)
        y_quant = pandas.concat((y_quant, y_quanttmp), axis=0)

    yall = pandas.concat((y_rank, y_threshold, y_quant), axis=1)
    Y = pandas.merge(Y, yall, how='inner', left_index=True, right_index=True)

    PLOT = False
    if PLOT:
        # to better understand, try plotting something like:
        labels = ["score", "score_drug_gene_rank", "score_drug_rank", "score_drug_gene_threshold", "score_drug_threshold"]

        for label in labels:
            plt.figure()
            plt.plot(Xdf['sgRNA Score'].values, Y[label].values, '.')
            r, pearp = sp.stats.pearsonr(Xdf['sgRNA Score'].values.flatten(), Y[label].values.flatten())
            plt.title(label + ' VS pred. score, $r$=%0.2f (p=%0.2e)' % (r, pearp))
            plt.xlabel("sgRNA prediction score")
            plt.ylabel(label)

    gene_position = util.impute_gene_position(gene_position)

    if learn_options is not None and learn_options["weighted"] == "variance":
        print "computing weights from replicate variance..."
        # compute the variance across replicates so can use it as a weight
        data = pandas.read_excel(data_file, sheetname="Normalized", skiprows=range(0, 6+1), index_col=[0, 4])
        data.index.names = ["Sequence", "Target gene"]

        experiments = {}
        experiments['AZD_200nM'] = ['Deep 25', 'Deep 27', 'Deep 29 ', 'Deep 31']
        experiments['6TG_2ug/mL'] = ['Deep 33', 'Deep 35', 'Deep 37', 'Deep 39']
        experiments['PLX_2uM'] = ['Deep 49', 'Deep 51', 'Deep 53', 'Deep 55']

        variance = None
        for drug in drugs_to_genes.keys():
            data_tmp = data.iloc[data.index.get_level_values('Target gene').isin(drugs_to_genes[drug])][experiments[drug]]
            data_tmp["drug"] = drug
            data_tmp = data_tmp.set_index('drug', append=True)
            data_tmp["variance"] = np.var(data_tmp.values, axis=1)
            if variance is None:
                variance = data_tmp["variance"].copy()
            else:
                variance = pandas.concat((variance, data_tmp["variance"]), axis=0)

        orig_index = Y.index.copy()
        Y = pandas.merge(Y, pandas.DataFrame(variance), how="inner", left_index=True, right_index=True)
        Y = Y.ix[orig_index]
        print "done."

    # Make sure to keep this check last in this function
    assert Xdf.index.equals(Y.index), "The index of Xdf is different from the index of Y (this can cause inconsistencies/random performance later on)"

    return Xdf, drugs_to_genes, target_genes, Y, gene_position


def merge_all(data_file=None, data_file2=None, data_file3=None, learn_options=None):
    Xdf, Y, gene_position, target_genes = mergeV1_V2(data_file, data_file2, learn_options)
    gene_position_xu, target_genes_xu, Xdf_xu, Y_xu = read_xu_et_al(data_file3, learn_options)
    Xdf = pandas.concat((Xdf, Xdf_xu))
    Y = pandas.concat((Y, Y_xu))
    gene_position = pandas.concat((gene_position, gene_position_xu))
    target_genes = np.concatenate((target_genes, target_genes_xu))

    return Xdf, Y, gene_position, target_genes

def mergeV1_V2(data_file, data_file2, learn_options):
    '''
    ground_truth_label, etc. are taken to correspond to the V2 data, and then the V1 is appropriately matched
    based on semantics
    '''
    assert not learn_options['include_strand'], "don't currently have 'Strand' column in V1 data"

    annotations, gene_position1, target_genes1, Xdf1, Y1 = read_V1_data(data_file, learn_options)
    Xdf2, drugs_to_genes, target_genes2, Y2, gene_position2 = read_V2_data(data_file2)

    Y1.rename(columns={'average rank': learn_options["rank-transformed target name"]}, inplace=True)
    Y1.rename(columns={'average threshold': learn_options["binary target name"]}, inplace=True)

    # rename columns, and add a dummy "drug" to V1 so can join the data sets
    Y1["drug"] = ["nodrug" for x in range(Y1.shape[0])]
    Y1 = Y1.set_index('drug', append=True)
    Y1.index.names = ['Sequence', 'Target gene', 'drug']

    Y_cols_to_keep = np.unique(['Target gene', 'test', 'score_drug_gene_rank', 'score_drug_gene_threshold'])

    Y1 = Y1[Y_cols_to_keep]
    Y2 = Y2[Y_cols_to_keep]

    Xdf1["drug"] = ["nodrug" for x in range(Xdf1.shape[0])]
    Xdf1 = Xdf1.set_index('drug', append=True)

    X_cols_to_keep = ['30mer', 'Strand']
    Xdf1 = Xdf1[X_cols_to_keep]
    Xdf2 = Xdf2[X_cols_to_keep]

    gene_position1["drug"] = ["nodrug" for x in range(gene_position1.shape[0])]
    gene_position1 = gene_position1.set_index('drug', append=True)
    gene_position1.index.names = ['Sequence', 'Target gene', 'drug']
    cols_to_keep = [u'Percent Peptide', u'Amino Acid Cut position']
    gene_position1 = gene_position1[cols_to_keep]
    gene_position2 = gene_position2[cols_to_keep]

    Y = pandas.concat((Y1, Y2), axis=0)
    Xdf = pandas.concat((Xdf1, Xdf2), axis=0)
    gene_position = pandas.concat((gene_position1, gene_position2))

    # target_genes = target_genes1 + target_genes2
    target_genes = np.concatenate((target_genes1, target_genes2))

    save_to_file = False

    if save_to_file:
        Y.index.names = ['Sequence', 'Target', 'drug']
        assert np.all(Xdf.index.values==Y.index.values), "rows don't match up"

        onedupind = np.where(Y.index.duplicated())[0][0]
        alldupind = np.where(Y.index.get_level_values(0).values==Y.index[onedupind][0])[0]

        #arbitrarily set one of these to have "nodrug2" as the third level index
        #so that they are not repeated, and the joints therefore do not augment the data set
        assert len(alldupind)==2, "expected only duplicates"
        newindex = Y.index.tolist()
        newindex[onedupind] = (newindex[onedupind][0], newindex[onedupind][1], "nodrug2")
        Y.index = pandas.MultiIndex.from_tuples(newindex, names = Y.index.names)
        Xdf.index = pandas.MultiIndex.from_tuples(newindex, names = Y.index.names)

        # there seems to be a duplicate index, and thus this increases the data set size, so doing it the hacky way...
        XandY = pandas.merge(Xdf, Y, how="inner", left_index=True, right_index=True)
        gene_position_tmp = gene_position.copy()
        gene_position_tmp.index.names = ['Sequence', 'Target', 'drug']
        gene_position_tmp.index = pandas.MultiIndex.from_tuples(newindex, names = Y.index.names)
        XandY = pandas.merge(XandY, gene_position_tmp, how="inner", left_index=True, right_index=True)

        # truncate to 30mers
        XandY["30mer"] = XandY["30mer"].apply(lambda x: x[0:30])
        XandY.to_csv(r'D:\Source\CRISPR\data\tmp\V3.csv')

    return Xdf, Y, gene_position, target_genes


def get_V1_genes(data_file=None):
    annotations, gene_position, target_genes, Xdf, Y = read_V1_data(data_file, learn_options=None)
    return target_genes


def get_V2_genes(data_file=None):
    Xdf, drugs_to_genes, target_genes, Y, gene_position = read_V2_data(data_file, verbose=False)
    return target_genes


def get_V3_genes(data_fileV1=None, data_fileV2=None):
    target_genes = np.concatenate((get_V1_genes(data_fileV1), get_V2_genes(data_fileV2)))
    return target_genes

def get_xu_genes(data_file=None):
    return read_xu_et_al(data_file)[1]

def get_mouse_genes(data_file=None):
    annotations, gene_position, target_genes, Xdf, Y = read_V1_data(data_file, learn_options=None)
    return Xdf[Xdf['Organism'] == 'mouse']['Target gene'].unique()


def get_human_genes(data_file=None):
    annotations, gene_position, target_genes, Xdf, Y = read_V1_data(data_file, learn_options=None)
    mouse_genes = Xdf[Xdf['Organism'] == 'mouse']['Target gene'].unique()
    all_genes = get_V3_genes(None, None)  # TODO this needs to support specifying file names (!= 'None')
    return np.setdiff1d(all_genes, mouse_genes)
