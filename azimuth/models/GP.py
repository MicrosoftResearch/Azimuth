import numpy as np
import matplotlib.pyplot as plt


def gp_on_fold(feature_sets, train, test, y, y_all, inputs, dim, dimsum, learn_options):
    import GPy
    from models.gpy_ssk import WeightedDegree

    sequences = np.array([str(x) for x in y_all.index.get_level_values(0).tolist()])

    kern = WeightedDegree(1, sequences, d=learn_options['kernel degree'], active_dims=[0])
    X = np.arange(len(train))[:, None]

    current_dim = 1

    if 'gc_count' in feature_sets:
        kern += GPy.kern.RBF(1, active_dims=[current_dim], name='GC_rbf')
        X = np.concatenate((X, feature_sets['gc_count'].values), axis=1)
        current_dim += 1
        assert X.shape[1] == current_dim

    if 'drug' in feature_sets:
        Q = feature_sets['drug'].values.shape[1]
        kern += GPy.kern.Linear(Q, active_dims=range(current_dim, current_dim+Q), name='drug_lin')
        X = np.concatenate((X, feature_sets['drug'].values), axis=1)
        current_dim += Q
        assert X.shape[1] == current_dim

    if 'gene effect' in feature_sets:
        Q = feature_sets['gene effect'].values.shape[1]
        kern += GPy.kern.Linear(Q, active_dims=range(current_dim, current_dim+Q), name='gene_lin')
        X = np.concatenate((X, feature_sets['gene effect'].values), axis=1)
        current_dim += Q
        assert X.shape[1] == current_dim

    if "Percent Peptide" in feature_sets:
        Q = feature_sets['Percent Peptide'].values.shape[1]
        kern += GPy.kern.RBF(Q, active_dims=range(current_dim, current_dim+Q), name='percent_pept')
        X = np.concatenate((X, feature_sets['Percent Peptide'].values), axis=1)
        current_dim += Q
        assert X.shape[1] == current_dim

    if "Nucleotide cut position" in feature_sets:
        Q = feature_sets['Nucleotide cut position'].values.shape[1]
        kern += GPy.kern.RBF(Q, active_dims=range(current_dim, current_dim+Q), name='nucleo_cut')
        X = np.concatenate((X, feature_sets['Nucleotide cut position'].values), axis=1)
        current_dim += Q
        assert X.shape[1] == current_dim

    if "Strand effect" in feature_sets:
        Q = feature_sets['Strand effect'].values.shape[1]
        kern += GPy.kern.Linear(Q, active_dims=range(current_dim, current_dim+Q), name='strand')
        X = np.concatenate((X, feature_sets['Strand effect'].values), axis=1)
        current_dim += Q
        assert X.shape[1] == current_dim

    if "NGGX" in feature_sets:
        Q = feature_sets['NGGX'].values.shape[1]
        kern += GPy.kern.Linear(Q, active_dims=range(current_dim, current_dim+Q), name='NGGX')
        X = np.concatenate((X, feature_sets['NGGX'].values), axis=1)
        current_dim += Q
        assert X.shape[1] == current_dim

    if "TM" in feature_sets:
        Q = feature_sets['TM'].values.shape[1]
        kern += GPy.kern.RBF(Q, ARD=True, active_dims=range(current_dim, current_dim+Q), name='TM')
        X = np.concatenate((X, feature_sets['TM'].values), axis=1)
        current_dim += Q
        assert X.shape[1] == current_dim

    if "gene features" in feature_sets:
        Q = feature_sets['gene features'].values.shape[1]
        kern += GPy.kern.Linear(Q, ARD=True, active_dims=range(current_dim, current_dim+Q), name='genefeat')
        X = np.concatenate((X, feature_sets['gene features'].values), axis=1)
        current_dim += Q
        assert X.shape[1] == current_dim

    kern += GPy.kern.Bias(X.shape[1])

    if learn_options['warpedGP']:
        m = GPy.models.WarpedGP(X[train], y[train], kernel=kern)
    else:
        m = GPy.models.GPRegression(X[train], y[train], kernel=kern)


    if False: # debug plots
        m.optimize_restarts(3, messages=1)
        y_pred, y_uncert = m.predict(X[test])

        plt.figure('residuals')
        plt.title('residuals')
        plt.hist(y_pred.flatten()-y[test].flatten(), bins=50)

        plt.figure('predictions')
        plt.title('predictions')
        plt.errorbar(y_pred.flatten(), y[test].flatten(), yerr=np.sqrt(y_uncert.flatten()), fmt='o')
        plt.xlabel('prediction')
        plt.ylabel('truth')

        plt.figure('kernel')
        plt.title('kernel')
        plt.imshow(m.kern.K(X,X))
        print m
        print "%.3f variance explained" % (m.Gaussian_noise.variance/y[train].var())
        import ipdb; ipdb.set_trace()
        plt.close('all')
    else:
        m.optimize_restarts(3)
        # m.optimize(messages=1)
        y_pred, y_uncert = m.predict(X[test])

    # TODO add offset such that low scores are around 0 (not -4 or so)

    return y_pred, m[:]
